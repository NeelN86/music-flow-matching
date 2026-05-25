# Vocoder: converts decoded mel spectrograms to WAV files.
#
# Primary path — HiFi-GAN (SpeechBrain tts-hifigan-libritts-16kHz):
#   denormalize_mel → dB → power → log-mel → HiFi-GAN generator → wav
#   HiFi-GAN is loaded once lazily on first call and cached.
#   The 16kHz LibriTTS model matches our mel config exactly:
#   sr=16000, n_mels=80, n_fft=1024, hop=256.
#
# Fallback path — Griffin-Lim (if HiFi-GAN is unavailable or fails):
#   denormalize_mel → dB → power → mel_to_stft → griffinlim → wav
#
# Mel format conversion (our dB-scale → HiFi-GAN log-scale):
#   mel_db  = mel_norm * MEL_STD + MEL_MEAN         (dB, roughly -100..+10)
#   power   = 10^(mel_db / 10)                      (linear power)
#   log_mel = log(power + 1e-9)                     (natural log, ~-20..0)
#   This is the format the SpeechBrain HiFi-GAN was trained with.

from __future__ import annotations

import concurrent.futures
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from src.config import (
    CHECKPOINT_DIR,
    GRIFFIN_LIM_ITERS,
    HOP_LENGTH,
    MEL_MEAN,
    MEL_STD,
    N_FFT,
    OUTPUT_DIR,
    SAMPLE_RATE,
)

_hifigan = None   # cached after first load
_hifigan_failed = False  # set True if load/inference ever fails, skips retries


def _load_hifigan():
    """Load SpeechBrain HiFi-GAN (16kHz) from local checkpoint once and cache it."""
    global _hifigan, _hifigan_failed
    if _hifigan is not None:
        return _hifigan
    if _hifigan_failed:
        return None
    savedir = os.path.join(CHECKPOINT_DIR, "hifigan")
    ckpt = os.path.join(savedir, "generator.ckpt")
    if not os.path.exists(ckpt):
        print("HiFi-GAN checkpoint not found, falling back to Griffin-Lim.")
        _hifigan_failed = True
        return None
    try:
        from speechbrain.inference.vocoders import HIFIGAN
        # Load from local directory — no network request needed
        _hifigan = HIFIGAN.from_hparams(
            source=savedir,
            savedir=savedir,
            run_opts={"device": "cpu"},
        )
        print("HiFi-GAN loaded successfully.")
        return _hifigan
    except Exception as e:
        print(f"HiFi-GAN load failed ({e}), falling back to Griffin-Lim.")
        _hifigan_failed = True
        return None


def _mel_norm_to_log(mel_normalized: np.ndarray) -> torch.Tensor:
    """Convert our dB-normalized mel to the natural-log scale HiFi-GAN expects.

    Returns a [1, N_MELS, T] float32 tensor ready for HiFi-GAN.
    """
    if mel_normalized.ndim == 3:
        mel_normalized = mel_normalized[0]      # [N_MELS, T]
    mel_db = mel_normalized * MEL_STD + MEL_MEAN
    power = 10.0 ** (mel_db / 10.0)          # db_to_power inline — avoids librosa lazy-import conflict with SpeechBrain
    log_mel = np.log(power + 1e-9).astype(np.float32)
    return torch.from_numpy(log_mel).unsqueeze(0)   # [1, N_MELS, T]


def mel_to_wav_hifigan(mels_normalized: list[np.ndarray]) -> list[np.ndarray]:
    """Decode a list of normalized mels to waveforms via HiFi-GAN (batched).

    Returns a list of float32 numpy arrays (one per input).
    Raises RuntimeError if HiFi-GAN is unavailable.
    """
    model = _load_hifigan()
    if model is None:
        raise RuntimeError("HiFi-GAN not available")

    # Stack into a batch [B, N_MELS, T]
    mel_tensors = torch.cat([_mel_norm_to_log(m) for m in mels_normalized], dim=0)

    with torch.no_grad():
        wavs = model.decode_batch(mel_tensors)   # [B, 1, T_wav] or [B, T_wav]

    wavs = wavs.squeeze(1) if wavs.ndim == 3 else wavs
    return [wavs[i].cpu().numpy().astype(np.float32) for i in range(wavs.shape[0])]


def mel_to_wav(
    mel_normalized: Tensor | np.ndarray,
    n_iter: int = GRIFFIN_LIM_ITERS,
) -> np.ndarray:
    """Reconstruct a waveform from a normalized mel via Griffin-Lim (fallback).

    Steps: denormalize → dB → power → mel_to_stft magnitude → griffinlim.
    """
    if isinstance(mel_normalized, torch.Tensor):
        mel_np = mel_normalized.detach().cpu().numpy()
    else:
        mel_np = np.asarray(mel_normalized)

    if mel_np.ndim == 3:
        mel_np = mel_np[0]

    mel_db = mel_np * MEL_STD + MEL_MEAN
    power = 10.0 ** (mel_db / 10.0)
    stft_mag = librosa.feature.inverse.mel_to_stft(power, sr=SAMPLE_RATE, n_fft=N_FFT)
    wav = librosa.griffinlim(stft_mag, n_iter=n_iter, hop_length=HOP_LENGTH)
    return wav.astype(np.float32)


def decode_batch(
    latents: Tensor,
    vae_decoder: callable,
    output_dir: str = OUTPUT_DIR,
    n_workers: int = 4,
) -> list[str]:
    """Decode a batch of latents to WAV files.

    Tries HiFi-GAN first (batched, single forward pass).
    Falls back to parallel Griffin-Lim if HiFi-GAN is unavailable.

    Args:
        latents:     [B, latent_dim] final positions from euler_integrate.
        vae_decoder: Callable [B, D] → [B, 1, N_MELS, N_FRAMES].
        output_dir:  Directory for output WAV files.
        n_workers:   Worker threads used only for the Griffin-Lim fallback.

    Returns:
        List of B absolute paths.
    """
    global _hifigan_failed

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        mels = vae_decoder(latents)             # [B, 1, N_MELS, N_FRAMES]

    B = mels.shape[0]
    mels_np = [mels[i, 0].cpu().numpy() for i in range(B)]
    paths = [
        os.path.abspath(os.path.join(output_dir, f"variation_{i}.wav"))
        for i in range(B)
    ]

    # ── Primary: HiFi-GAN (batched) ───────────────────────────────────────────
    if not _hifigan_failed:
        try:
            wavs = mel_to_wav_hifigan(mels_np)
            for wav, path in zip(wavs, paths):
                sf.write(path, wav, SAMPLE_RATE)
            return paths
        except Exception as e:
            print(f"HiFi-GAN inference failed ({e}), falling back to Griffin-Lim.")
            _hifigan_failed = True

    # ── Fallback: parallel Griffin-Lim ────────────────────────────────────────
    def _write_wav(args: tuple[np.ndarray, str]) -> str:
        mel_np, path = args
        wav = mel_to_wav(mel_np)
        sf.write(path, wav, SAMPLE_RATE)
        return path

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_write_wav, zip(mels_np, paths)))

    return paths
