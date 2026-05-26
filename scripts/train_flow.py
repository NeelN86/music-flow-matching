"""Standalone CLI for flow matching training (run after VAE training).

Usage:
    # NSynth only (original)
    python scripts/train_flow.py --data_dir data/nsynth-valid

    # Mixed dataset
    python scripts/train_flow.py --data_dir data/nsynth-valid \
        --extra_audio_dirs data/vocalset --steps 30000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import ConcatDataset, DataLoader

from src.config import (
    FLOW_BATCH_SIZE,
    FLOW_CHECKPOINT,
    FLOW_LR,
    FLOW_STEPS,
    VAE_CHECKPOINT,
)
from src.data import FlatAudioDataset, NSynthDataset
from src.train import train_flow
from src.vae import load_vae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/nsynth-valid")
    parser.add_argument(
        "--extra_audio_dirs", nargs="*", default=[],
        help="Additional flat/nested WAV directories to mix in (e.g. data/vocalset)",
    )
    parser.add_argument("--steps", type=int, default=FLOW_STEPS)
    parser.add_argument("--lr", type=float, default=FLOW_LR)
    parser.add_argument("--batch_size", type=int, default=FLOW_BATCH_SIZE)
    parser.add_argument("--checkpoint_out", default=FLOW_CHECKPOINT)
    parser.add_argument("--vae_checkpoint", default=VAE_CHECKPOINT)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.vae_checkpoint):
        print(f"VAE checkpoint not found at {args.vae_checkpoint}. Train VAE first.")
        sys.exit(1)

    vae = load_vae(args.vae_checkpoint)

    nsynth_ds = NSynthDataset(
        os.path.join(args.data_dir, "examples.json"),
        os.path.join(args.data_dir, "audio"),
        normalize=True,
    )
    datasets = [nsynth_ds]

    for extra_dir in args.extra_audio_dirs:
        extra_ds = FlatAudioDataset(extra_dir, label="vocal", normalize=True)
        print(f"  Extra dir: {extra_dir}  ({len(extra_ds)} samples)")
        datasets.append(extra_ds)

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
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
