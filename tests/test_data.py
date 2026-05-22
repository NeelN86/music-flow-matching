"""Tests for src/data.py — mel pipeline and NSynthDataset."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.config import CLIP_DURATION, N_FRAMES, N_MELS, SAMPLE_RATE
from src.data import audio_to_mel, denormalize_mel, load_audio, normalize_mel


def _make_wav(duration: float = CLIP_DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthetic sine wave at 440 Hz."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_audio_to_mel_shape():
    """Synthetic waveform → mel must be exactly [N_MELS, N_FRAMES]."""
    wav = _make_wav()
    mel = audio_to_mel(wav)
    assert mel.shape == (N_MELS, N_FRAMES), f"Expected ({N_MELS}, {N_FRAMES}), got {mel.shape}"


def test_audio_to_mel_short_clip():
    """Clip shorter than CLIP_DURATION is zero-padded to [N_MELS, N_FRAMES]."""
    wav = _make_wav(duration=CLIP_DURATION / 2)
    mel = audio_to_mel(wav)
    assert mel.shape == (N_MELS, N_FRAMES)


def test_audio_to_mel_long_clip():
    """Clip longer than CLIP_DURATION is truncated to [N_MELS, N_FRAMES]."""
    wav = _make_wav(duration=CLIP_DURATION * 2)
    mel = audio_to_mel(wav)
    assert mel.shape == (N_MELS, N_FRAMES)


def test_normalize_round_trip():
    """normalize_mel followed by denormalize_mel recovers the original values."""
    wav = _make_wav()
    mel = audio_to_mel(wav)
    normed = normalize_mel(mel)
    recovered = denormalize_mel(normed)
    # Should be very close (floating point only)
    assert torch.allclose(recovered, torch.from_numpy(mel), atol=1e-4), (
        f"Round-trip error: max diff = {(recovered - torch.from_numpy(mel)).abs().max()}"
    )


def test_normalize_output_type():
    """normalize_mel returns a torch.Tensor regardless of numpy or Tensor input."""
    wav = _make_wav()
    mel_np = audio_to_mel(wav)
    mel_t = torch.from_numpy(mel_np)

    out_from_np = normalize_mel(mel_np)
    out_from_t = normalize_mel(mel_t)

    assert isinstance(out_from_np, torch.Tensor), "Expected Tensor from numpy input"
    assert isinstance(out_from_t, torch.Tensor), "Expected Tensor from Tensor input"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                print(f"FAIL  {name}  {e}")
