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

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import OUTPUT_DIR, SAMPLE_RATE, VAE_CHECKPOINT
from src.data import NSynthDataset, denormalize_mel
from src.vae import load_vae
from src.vocoder import mel_to_wav


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a NSynth sample through the VAE.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--idx", type=int, default=0, help="Dataset index to reconstruct")
    parser.add_argument("--checkpoint", default=VAE_CHECKPOINT)
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, "examples.json")
    audio_dir = os.path.join(args.data_dir, "audio")

    model = load_vae(args.checkpoint)
    print(f"Loaded VAE from {args.checkpoint}")

    dataset = NSynthDataset(json_path, audio_dir, normalize=True)
    mel, family, note_id = dataset[args.idx]
    print(f"Sample: {note_id}  family: {family}")

    mel_input = mel.unsqueeze(0)  # [1, 1, N_MELS, N_FRAMES]
    with torch.no_grad():
        mu, _ = model.encode(mel_input)
        recon = model.decode(mu)  # [1, 1, N_MELS, N_FRAMES]

    mel_recon = recon[0]  # [1, N_MELS, N_FRAMES]
    mse = F.mse_loss(mel_recon, mel).item()
    status = "PASS" if mse < 1.5 else "FAIL"
    print(f"MSE (normalized mel): {mse:.4f}  {status} (threshold < 1.5)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3-panel figure: original | reconstructed | |difference|
    orig_db = denormalize_mel(mel[0]).numpy()
    recon_db = denormalize_mel(mel_recon[0]).numpy()
    diff = np.abs(orig_db - recon_db)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, data, title in zip(axes, [orig_db, recon_db, diff], ["Original", "Reconstructed", "|Difference|"]):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{note_id} — {family}  |  MSE={mse:.4f}")
    fig.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, "reconstruction.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")

    # WAV files
    wav_orig = mel_to_wav(mel)
    wav_recon = mel_to_wav(mel_recon)
    orig_path = os.path.join(OUTPUT_DIR, "original.wav")
    recon_path = os.path.join(OUTPUT_DIR, "reconstructed.wav")
    sf.write(orig_path, wav_orig, SAMPLE_RATE)
    sf.write(recon_path, wav_recon, SAMPLE_RATE)
    print(f"Saved {orig_path}")
    print(f"Saved {recon_path}")


if __name__ == "__main__":
    main()
