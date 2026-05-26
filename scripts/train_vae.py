"""Standalone CLI for VAE training.

Usage:
    # NSynth only (original)
    python scripts/train_vae.py --data_dir data/nsynth-valid

    # Mixed dataset (NSynth + VocalSet or any flat WAV directory)
    python scripts/train_vae.py --data_dir data/nsynth-valid \
        --extra_audio_dirs data/vocalset --epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import ConcatDataset

from src.config import VAE_BATCH_SIZE, VAE_BETA, VAE_CHECKPOINT, VAE_EPOCHS, VAE_LR
from src.data import FlatAudioDataset, NSynthDataset, make_dataloader
from src.vae import train_vae_from_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AudioVAE on NSynth + optional extra audio dirs.")
    parser.add_argument("--data_dir", required=True, help="NSynth split dir (contains examples.json and audio/)")
    parser.add_argument(
        "--extra_audio_dirs", nargs="*", default=[],
        help="Additional flat/nested WAV directories to mix in (e.g. data/vocalset)",
    )
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap NSynth size for fast smoke tests")
    parser.add_argument("--max_extra_per_dir", type=int, default=None, help="Cap samples per extra_audio_dir")
    parser.add_argument("--batch_size", type=int, default=VAE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VAE_LR)
    parser.add_argument("--beta", type=float, default=VAE_BETA)
    parser.add_argument("--checkpoint", default=VAE_CHECKPOINT)
    parser.add_argument("--beta_warmup", type=int, default=5, help="KL warmup epochs (0 = no warmup)")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, "examples.json")
    audio_dir = os.path.join(args.data_dir, "audio")

    nsynth_ds = NSynthDataset(json_path, audio_dir, max_samples=args.max_samples, normalize=True)
    datasets = [nsynth_ds]

    for extra_dir in args.extra_audio_dirs:
        extra_ds = FlatAudioDataset(extra_dir, label="vocal", max_samples=args.max_extra_per_dir, normalize=True)
        print(f"  Extra dir: {extra_dir}  ({len(extra_ds)} samples)")
        datasets.append(extra_ds)

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print(f"Training VAE: total={len(dataset)} samples  epochs={args.epochs}  beta={args.beta}  "
          f"beta_warmup={args.beta_warmup}  lr={args.lr}  batch={args.batch_size}")

    train_vae_from_loader(
        loader=loader,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        checkpoint_path=args.checkpoint,
        beta_warmup_epochs=args.beta_warmup,
    )


if __name__ == "__main__":
    main()
