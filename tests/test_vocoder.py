"""Tests for src/vocoder.py — Griffin-Lim decode and parallel WAV export."""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.config import N_FRAMES, N_MELS, VAE_LATENT_DIM
from src.vae import AudioVAE
from src.vocoder import decode_batch, mel_to_wav

# Use few Griffin-Lim iters in tests so they run quickly
_TEST_ITERS = 8


def test_mel_to_wav_shape():
    """mel_to_wav([N_MELS, N_FRAMES]) returns a 1D numpy array of positive length."""
    mel = np.random.randn(N_MELS, N_FRAMES).astype(np.float32)
    wav = mel_to_wav(mel, n_iter=_TEST_ITERS)
    assert wav.ndim == 1, f"expected 1D, got shape {wav.shape}"
    assert len(wav) > 0, "wav is empty"


def test_mel_to_wav_finite():
    """mel_to_wav output contains no NaN or Inf values."""
    mel = np.random.randn(N_MELS, N_FRAMES).astype(np.float32)
    wav = mel_to_wav(mel, n_iter=_TEST_ITERS)
    assert np.isfinite(wav).all(), "wav contains NaN or Inf"


def test_decode_batch_returns_paths():
    """decode_batch with 4 latents returns 4 existing file paths."""
    vae = AudioVAE()
    vae.eval()
    latents = torch.randn(4, VAE_LATENT_DIM)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch Griffin-Lim iters via a wrapper
        import src.vocoder as _voc
        orig = _voc.GRIFFIN_LIM_ITERS
        _voc.GRIFFIN_LIM_ITERS = _TEST_ITERS
        try:
            paths = decode_batch(latents, vae.decode, output_dir=tmpdir, n_workers=4)
        finally:
            _voc.GRIFFIN_LIM_ITERS = orig

        assert len(paths) == 4, f"expected 4 paths, got {len(paths)}"
        for p in paths:
            assert os.path.isfile(p), f"file not created: {p}"


def test_decode_batch_parallel_faster():
    """4-worker ThreadPoolExecutor Griffin-Lim should be faster than sequential."""
    import concurrent.futures as _cf

    # 30 iters: each call ~1.5s on CPU so parallel benefit clearly outweighs overhead
    n_iter = 30

    vae = AudioVAE()
    vae.eval()
    latents = torch.randn(4, VAE_LATENT_DIM)
    with torch.no_grad():
        mels = vae.decode(latents)  # [4, 1, N_MELS, N_FRAMES]
    mel_list = [mels[i, 0].numpy() for i in range(4)]

    # Sequential
    t0 = time.perf_counter()
    for m in mel_list:
        mel_to_wav(m, n_iter=n_iter)
    t_seq = time.perf_counter() - t0

    # Parallel — same path as decode_batch internals
    def _run(m: np.ndarray) -> np.ndarray:
        return mel_to_wav(m, n_iter=n_iter)

    t0 = time.perf_counter()
    with _cf.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_run, mel_list))
    t_par = time.perf_counter() - t0

    assert t_par < t_seq, (
        f"parallel ({t_par:.2f}s) not faster than sequential ({t_seq:.2f}s)"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except NotImplementedError:
                print(f"SKIP  {name}  (not implemented)")
            except Exception as e:
                print(f"FAIL  {name}  {e}")
