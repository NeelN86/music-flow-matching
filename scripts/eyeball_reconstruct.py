"""Milestone gate diagnostic: original vs reconstructed mel spectrogram + audio.

Run after VAE training. Shows a 3-panel figure (original | reconstructed | difference)
and saves original.wav and reconstructed.wav. Prints MSE between original and
reconstructed mel (should be < 1.5 in normalized space).

Usage:
    python scripts/eyeball_reconstruct.py --data_dir data/nsynth-valid [--idx 0]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import OUTPUT_DIR, VAE_CHECKPOINT
from src.vae import load_vae
from src.visualize import mel_thumbnail
from src.vocoder import mel_to_wav


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
