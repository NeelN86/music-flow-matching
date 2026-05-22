# Griffin-Lim vocoder: converts decoded mel spectrograms to WAV files.
#
# Pipeline per sample:
#   denormalize_mel → db_to_power → mel_to_stft → griffinlim → wav
#
# decode_batch decodes 4 latents in parallel via ThreadPoolExecutor.
# librosa/numpy release the GIL, so Python threads give true parallelism
# despite the GIL. The VAE decoder runs in the main thread first (torch
# is not thread-safe for concurrent forward passes); threads only receive
# numpy arrays.
#
# Timing target: 4 parallel Griffin-Lim calls (~60 iters each) finish
# in ~2.5s on CPU — well within the 3.5s animation window.

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from src.config import (
    GRIFFIN_LIM_ITERS,
    HOP_LENGTH,
    MEL_MEAN,
    MEL_STD,
    N_FFT,
    N_MELS,
    OUTPUT_DIR,
    SAMPLE_RATE,
)
from src.data import denormalize_mel


def mel_to_wav(
    mel_normalized: Tensor | np.ndarray,
    n_iter: int = GRIFFIN_LIM_ITERS,
) -> np.ndarray:
    """Reconstruct a waveform from a normalized log-mel spectrogram.

    Steps:
      1. Denormalize: dB-scale mel = mel_normalized * MEL_STD + MEL_MEAN
      2. dB → power: librosa.db_to_power
      3. mel → STFT magnitude: librosa.feature.inverse.mel_to_stft
      4. Phase recovery: librosa.griffinlim

    Args:
        mel_normalized: [N_MELS, N_FRAMES] or [1, N_MELS, N_FRAMES].
        n_iter:         Griffin-Lim iterations.

    Returns:
        wav: [T] float32 numpy array at SAMPLE_RATE.
    """
    raise NotImplementedError


def decode_batch(
    latents: Tensor,
    vae_decoder: callable,
    output_dir: str = OUTPUT_DIR,
    n_workers: int = 4,
) -> list[str]:
    """Decode a batch of 2D latents to WAV files using parallel Griffin-Lim.

    Args:
        latents:     [B, 2] final latent positions from euler_integrate.
        vae_decoder: Callable Tensor[B,2] → Tensor[B,1,N_MELS,N_FRAMES].
                     Called in the main thread before spawning workers.
        output_dir:  Directory for output files.
        n_workers:   Thread pool size.

    Returns:
        List of B absolute paths: ["outputs/variation_0.wav", ...].
    """
    raise NotImplementedError
