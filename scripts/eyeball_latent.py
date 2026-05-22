"""Milestone gate diagnostic: scatter plot of 2D latent space colored by instrument family.

Run after VAE training. Pass if ≥3 instrument families form separable clusters
and the latent range is roughly [-4, 4]².

Usage:
    python scripts/eyeball_latent.py --data_dir data/nsynth-valid [--max_samples 1000]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import VAE_CHECKPOINT
from src.vae import load_vae
from src.visualize import latent_scatter


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
