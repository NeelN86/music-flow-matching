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
    parser = argparse.ArgumentParser(description="Train AudioVAE on NSynth mel spectrograms.")
    parser.add_argument("--data_dir", required=True, help="NSynth split dir (contains examples.json and audio/)")
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size for fast smoke tests")
    parser.add_argument("--batch_size", type=int, default=VAE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VAE_LR)
    parser.add_argument("--beta", type=float, default=VAE_BETA)
    parser.add_argument("--checkpoint", default=VAE_CHECKPOINT)
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, "examples.json")
    audio_dir = os.path.join(args.data_dir, "audio")

    print(f"Training VAE: epochs={args.epochs}  beta={args.beta}  lr={args.lr}  batch={args.batch_size}")
    if args.max_samples:
        print(f"Smoke-test mode: max_samples={args.max_samples}")

    train_vae(
        json_path=json_path,
        audio_dir=audio_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
