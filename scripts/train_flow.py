"""Standalone CLI for flow matching training (run after VAE training).

Usage:
    python scripts/train_flow.py
    python scripts/train_flow.py --steps 30000 --data_dir data/nsynth-valid
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import DataLoader

from src.config import (
    FLOW_BATCH_SIZE,
    FLOW_CHECKPOINT,
    FLOW_LR,
    FLOW_STEPS,
    VAE_CHECKPOINT,
)
from src.data import NSynthDataset
from src.train import train_flow
from src.vae import load_vae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/nsynth-valid")
    parser.add_argument("--steps", type=int, default=FLOW_STEPS)
    parser.add_argument("--lr", type=float, default=FLOW_LR)
    parser.add_argument("--batch_size", type=int, default=FLOW_BATCH_SIZE)
    parser.add_argument("--checkpoint_out", default=FLOW_CHECKPOINT)
    args = parser.parse_args()

    if not os.path.exists(VAE_CHECKPOINT):
        print(f"VAE checkpoint not found at {VAE_CHECKPOINT}. Train VAE first.")
        sys.exit(1)

    vae = load_vae(VAE_CHECKPOINT)

    dataset = NSynthDataset(
        os.path.join(args.data_dir, "examples.json"),
        os.path.join(args.data_dir, "audio"),
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} samples, batch_size={args.batch_size}")
    print(f"Training for {args.steps} steps -> saving to {args.checkpoint_out}")

    train_flow(
        vae=vae,
        dataloader=loader,
        steps=args.steps,
        lr=args.lr,
        checkpoint_path=args.checkpoint_out,
    )


if __name__ == "__main__":
    main()
