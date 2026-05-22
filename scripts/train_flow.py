"""Standalone CLI for flow matching training (run after VAE training).

Usage:
    python scripts/train_flow.py --data_dir data/nsynth-valid [--steps 10000]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import FLOW_BATCH_SIZE, FLOW_CHECKPOINT, FLOW_LR, FLOW_STEPS, VAE_CHECKPOINT
from src.train import train_flow
from src.vae import load_vae


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
