"""Standalone CLI for VAE training.

Usage:
    python scripts/train_vae.py --data_dir data/nsynth-valid [--epochs 30] [--max_samples 500]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import VAE_BATCH_SIZE, VAE_BETA, VAE_CHECKPOINT, VAE_EPOCHS, VAE_LR
from src.vae import train_vae


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
