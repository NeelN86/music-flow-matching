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

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import VAE_CHECKPOINT
from src.data import NSynthDataset
from src.vae import load_vae
from src.visualize import latent_scatter


def main() -> None:
    parser = argparse.ArgumentParser(description="Scatter plot of 2D VAE latent space colored by instrument family.")
    parser.add_argument("--data_dir", required=True, help="NSynth split directory")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--checkpoint", default=VAE_CHECKPOINT)
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, "examples.json")
    audio_dir = os.path.join(args.data_dir, "audio")

    model = load_vae(args.checkpoint)
    print(f"Loaded VAE from {args.checkpoint}")

    dataset = NSynthDataset(json_path, audio_dir, max_samples=args.max_samples, normalize=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Encoding {len(dataset)} samples...")

    all_mu: list[np.ndarray] = []
    all_labels: list[str] = []

    with torch.no_grad():
        for mels, families, _ in loader:
            mu, _ = model.encode(mels)
            all_mu.append(mu.numpy())
            all_labels.extend(families)

    latents = np.concatenate(all_mu, axis=0)
    print(
        f"Latent range: "
        f"x=[{latents[:, 0].min():.2f}, {latents[:, 0].max():.2f}]  "
        f"y=[{latents[:, 1].min():.2f}, {latents[:, 1].max():.2f}]"
    )
    print(f"Instrument families found: {sorted(set(all_labels))}")

    fig = latent_scatter(latents, all_labels)
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/latent_scatter.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}  ({len(latents)} points)")


if __name__ == "__main__":
    main()
