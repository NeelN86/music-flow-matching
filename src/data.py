
# Audio preprocessing pipeline and NSynth dataset loader.
#
# mel pipeline: load_audio → audio_to_mel → normalize_mel
# inverse:      denormalize_mel → (passed to vocoder.mel_to_wav)

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.config import (
    CLIP_DURATION,
    HOP_LENGTH,
    MEL_MEAN,
    MEL_STD,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
)

# Silence floor in dB: used when padding short clips with silence
_SILENCE_DB: float = -80.0


def load_audio(
    path: str | Path,
    target_sr: int = SAMPLE_RATE,
    duration: float = CLIP_DURATION,
) -> np.ndarray:
    """Load an audio file, resample to target_sr, convert to mono, and
    fix-length to `duration` seconds (pad with zeros or truncate).

    Returns:
        wav: [target_sr * duration] float32 numpy array.
    """
    n_samples = int(target_sr * duration)
    wav, _ = librosa.load(str(path), sr=target_sr, mono=True, duration=duration)
    if len(wav) < n_samples:
        wav = np.pad(wav, (0, n_samples - len(wav)))
    else:
        wav = wav[:n_samples]
    return wav.astype(np.float32)


def audio_to_mel(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert a fixed-length waveform to a log-mel spectrogram.

    Uses librosa.feature.melspectrogram + librosa.power_to_db.
    Pads (with silence floor) or truncates to exactly [N_MELS, N_FRAMES].

    Returns:
        mel: [N_MELS, N_FRAMES] float32 array in dB scale.
    """
    power_mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(power_mel, ref=1.0, top_db=80.0)

    # Pad or truncate to exactly N_FRAMES columns
    if mel_db.shape[1] < N_FRAMES:
        pad_cols = N_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_cols)), constant_values=_SILENCE_DB)
    else:
        mel_db = mel_db[:, :N_FRAMES]

    assert mel_db.shape == (N_MELS, N_FRAMES), (
        f"Expected ({N_MELS}, {N_FRAMES}), got {mel_db.shape}"
    )
    return mel_db.astype(np.float32)


def normalize_mel(
    mel: np.ndarray | Tensor,
    mean: float = MEL_MEAN,
    std: float = MEL_STD,
) -> Tensor:
    """Standardize log-mel: (mel - mean) / std → approximately N(0,1)."""
    if not isinstance(mel, Tensor):
        mel = torch.from_numpy(mel)
    return (mel.float() - mean) / std


def denormalize_mel(
    mel: Tensor,
    mean: float = MEL_MEAN,
    std: float = MEL_STD,
) -> Tensor:
    """Invert normalize_mel: mel * std + mean → dB-scale log-mel."""
    return mel * std + mean


def compute_mel_stats(
    json_path: str,
    audio_dir: str,
    max_samples: int = 2000,
) -> tuple[float, float]:
    """Compute global (mean, std) over log-mel values from a random subset.

    Run once after downloading NSynth; paste the returned values into
    config.MEL_MEAN and config.MEL_STD.

    Args:
        json_path:   Path to NSynth examples.json.
        audio_dir:   Directory containing .wav files.
        max_samples: Cap on files to process (fast approximation).

    Returns:
        (mean, std) as Python floats.
    """
    validate_nsynth_dir(audio_dir, json_path)
    with open(json_path) as f:
        meta = json.load(f)

    note_ids = list(meta.keys())
    random.shuffle(note_ids)
    note_ids = note_ids[:max_samples]

    all_values: list[np.ndarray] = []
    for note_id in note_ids:
        wav_path = os.path.join(audio_dir, note_id + ".wav")
        if not os.path.exists(wav_path):
            continue
        wav = load_audio(wav_path)
        mel = audio_to_mel(wav)
        all_values.append(mel.ravel())

    concat = np.concatenate(all_values)
    return float(concat.mean()), float(concat.std())


def validate_nsynth_dir(audio_dir: str, json_path: str) -> None:
    """Raise FileNotFoundError with clear download instructions if NSynth is missing."""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"NSynth metadata not found: {json_path}\n"
            "Download NSynth from https://magenta.tensorflow.org/datasets/nsynth\n"
            "and extract so that 'examples.json' and 'audio/' are at the same level."
        )
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(
            f"NSynth audio directory not found: {audio_dir}\n"
            "Download NSynth from https://magenta.tensorflow.org/datasets/nsynth"
        )
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    if len(wav_files) == 0:
        raise FileNotFoundError(
            f"No .wav files found in {audio_dir}. "
            "The NSynth audio directory appears to be empty."
        )


class NSynthDataset(Dataset):
    """PyTorch Dataset over NSynth audio files.

    Each item is (mel [1, N_MELS, N_FRAMES], instrument_family_str, note_str).
    Loads WAV files lazily on __getitem__ to avoid loading ~2.5 GB into RAM.

    Args:
        json_path:   Path to NSynth examples.json.
        audio_dir:   Directory containing .wav files ({note_str}.wav).
        max_samples: Subsample the dataset — useful for fast dev iteration.
        normalize:   Apply normalize_mel using config MEL_MEAN / MEL_STD.
    """

    def __init__(
        self,
        json_path: str,
        audio_dir: str,
        max_samples: int | None = None,
        normalize: bool = True,
    ) -> None:
        validate_nsynth_dir(audio_dir, json_path)
        self.audio_dir = audio_dir
        self.normalize = normalize

        with open(json_path) as f:
            meta = json.load(f)

        # Keep only notes whose WAV file actually exists
        self.notes: list[tuple[str, str]] = []  # [(note_id, family_str), ...]
        for note_id, info in meta.items():
            wav_path = os.path.join(audio_dir, note_id + ".wav")
            if os.path.exists(wav_path):
                self.notes.append((note_id, info.get("instrument_family_str", "unknown")))

        if max_samples is not None and max_samples < len(self.notes):
            random.shuffle(self.notes)
            self.notes = self.notes[:max_samples]

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, idx: int) -> tuple[Tensor, str, str]:
        """Returns (mel [1, N_MELS, N_FRAMES], family_str, note_str)."""
        note_id, family = self.notes[idx]
        wav_path = os.path.join(self.audio_dir, note_id + ".wav")
        wav = load_audio(wav_path)
        mel = audio_to_mel(wav)

        mel_t = torch.from_numpy(mel).unsqueeze(0)  # [1, N_MELS, N_FRAMES]
        if self.normalize:
            mel_t = normalize_mel(mel_t)

        return mel_t, family, note_id


def make_dataloader(
    dataset: NSynthDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap NSynthDataset in a DataLoader.

    num_workers=0 default: on Windows, multiprocessing pickle overhead
    often exceeds the I/O benefit for small-to-medium NSynth subsets.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
