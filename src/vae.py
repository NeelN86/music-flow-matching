# β-VAE with a 2D bottleneck trained on mel spectrograms.
#
# The 2D latent space is the linchpin of the whole project: it lets the
# VelocityMLP operate in the same space that we visualize with quiver+particles,
# with no PCA approximation. Every design decision here serves the goal of a
# well-structured 2D latent space that separates instrument families.
#
# Conv arithmetic (kernel=4, stride=2, padding=1 — doubles spatial dims exactly):
#   Encoder: [1,80,256]→[32,40,128]→[64,20,64]→[128,10,32]→[256,5,16] → flat 20480
#   Decoder: mirrors with ConvTranspose2d
#
# Why an intermediate FC(20480→512) before the 2D bottleneck:
#   Compressing 20480 dims to 2 in one shot is a gradient dead-end.
#   The 512-wide intermediate layer gives the optimizer a foothold.
#
# Why MSE loss (not BCE):
#   Mel values in dB are unbounded — BCE expects [0,1] outputs.
#
# Why β=0.5 (not 1.0):
#   Prioritizes reconstruction quality; the milestone gate will indicate
#   whether to adjust. β controls the KL/recon trade-off: lower β →
#   better reconstruction, less structured latent space.

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import (
    CHECKPOINT_DIR,
    VAE_BATCH_SIZE,
    VAE_BETA,
    VAE_BOTTLENECK_FC,
    VAE_CHECKPOINT,
    VAE_CHANNELS,
    VAE_EPOCHS,
    VAE_FLAT_DIM,
    VAE_LATENT_DIM,
    VAE_LR,
)


class AudioEncoder(nn.Module):
    """4-stage strided conv encoder. BatchNorm after each conv (encoder only).

    Spatial dims (kernel=4, stride=2, padding=1):
      [B, 1, 80, 256] → [B, 32, 40, 128] → [B, 64, 20, 64]
                      → [B, 128, 10, 32] → [B, 256, 5, 16]
      Flatten → 20480 → FC(512) → mu [B, 2], logvar [B, 2]
    """

    def __init__(self) -> None:
        super().__init__()
        in_ch, *out_chs = VAE_CHANNELS  # [1, 32, 64, 128, 256]
        layers: list[nn.Module] = []
        for out_ch in out_chs:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.fc_hidden = nn.Linear(VAE_FLAT_DIM, VAE_BOTTLENECK_FC)
        self.fc_mu = nn.Linear(VAE_BOTTLENECK_FC, VAE_LATENT_DIM)
        self.fc_logvar = nn.Linear(VAE_BOTTLENECK_FC, VAE_LATENT_DIM)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (mu [B, 2], logvar [B, 2])."""
        h = self.conv(x).flatten(1)         # [B, 20480]
        h = F.relu(self.fc_hidden(h))       # [B, 512]
        mu = self.fc_mu(h)
        # Clamp logvar to [-4, 4] so exp(logvar) stays in [~0.02, ~55].
        # Prevents KL term blowing up to NaN during early training.
        logvar = torch.clamp(self.fc_logvar(h), -4.0, 4.0)
        return mu, logvar


class AudioDecoder(nn.Module):
    """Mirror of AudioEncoder via ConvTranspose2d. No BN (avoids inference artifacts).

    [B, 2] → FC(512) → FC(20480) → reshape [B, 256, 5, 16]
           → ConvTranspose2d ×4 → [B, 1, 80, 256]
    No final activation — MSE loss works in unbounded log-mel space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc_in = nn.Linear(VAE_LATENT_DIM, VAE_BOTTLENECK_FC)
        self.fc_expand = nn.Linear(VAE_BOTTLENECK_FC, VAE_FLAT_DIM)

        # Build deconv stages in reverse channel order
        channels = list(reversed(VAE_CHANNELS))  # [256, 128, 64, 32, 1]
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            is_last = (i == len(channels) - 2)
            layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            if not is_last:
                layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """Returns reconstructed mel [B, 1, 80, 256]."""
        h = F.relu(self.fc_in(z))           # [B, 512]
        h = F.relu(self.fc_expand(h))       # [B, 20480]
        h = h.view(-1, 256, 5, 16)          # [B, 256, 5, 16]
        return self.deconv(h)               # [B, 1, 80, 256]


class AudioVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = AudioEncoder()
        self.decoder = AudioDecoder()

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (mu [B, 2], logvar [B, 2])."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Returns reconstructed mel [B, 1, 80, 256]."""
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """z = mu + σ*ε where ε ~ N(0,I). Returns mu at eval time."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (recon [B,1,80,256], mu [B,2], logvar [B,2])."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def audio_vae_loss(
    recon: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = VAE_BETA,
) -> tuple[Tensor, Tensor, Tensor]:
    """β-VAE loss = MSE_recon + β × KL.

    Both terms normalized per element so β is scale-independent.
    KL per dim: -0.5 * (1 + logvar - mu² - exp(logvar))

    Returns:
        (total_loss, recon_loss, kl_loss) — all scalar tensors for logging.
    """
    # MSE averaged over all elements (B × 1 × N_MELS × N_FRAMES)
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    # KL averaged per latent dim per sample: -0.5 * mean over B and dims
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def train_vae(
    json_path: str,
    audio_dir: str,
    epochs: int = VAE_EPOCHS,
    batch_size: int = VAE_BATCH_SIZE,
    lr: float = VAE_LR,
    beta: float = VAE_BETA,
    checkpoint_path: str = VAE_CHECKPOINT,
    max_samples: int | None = None,
    beta_warmup_epochs: int = 5,
    progress_cb: Callable[[int, int, float, float, float], None] | None = None,
) -> tuple[AudioVAE, list[dict]]:
    """Train AudioVAE on NSynth and return (model, per-step log dicts).

    Each log dict: {epoch, step, total, recon, kl}.

    Args:
        max_samples:         Cap dataset size for fast smoke tests (None = full dataset).
        beta_warmup_epochs:  Linearly ramp beta from 0 → target over this many epochs.
                             Prevents posterior collapse by letting the encoder develop
                             structure before the KL penalty kicks in fully.
        progress_cb:         Called as cb(epoch, step, total_loss, recon_loss, kl_loss).

    Returns:
        (model in eval() mode, history list).
    """
    from src.data import NSynthDataset, make_dataloader

    dataset = NSynthDataset(json_path, audio_dir, max_samples=max_samples, normalize=True)
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []

    model.train()
    global_step = 0
    for epoch in range(epochs):
        # KL warmup: linearly ramp beta from 0 to target over beta_warmup_epochs
        beta_eff = beta * min(1.0, (epoch + 1) / max(1, beta_warmup_epochs))

        for batch in loader:
            mel_batch = batch[0]  # [B, 1, N_MELS, N_FRAMES]

            optimizer.zero_grad()
            recon, mu, logvar = model(mel_batch)
            total, recon_loss, kl_loss = audio_vae_loss(recon, mel_batch, mu, logvar, beta_eff)
            total.backward()
            # Gradient clipping guards against NaN loss early in training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            log = {
                "epoch": epoch,
                "step": global_step,
                "total": total.item(),
                "recon": recon_loss.item(),
                "kl": kl_loss.item(),
            }
            history.append(log)

            if progress_cb is not None:
                progress_cb(epoch, global_step, total.item(), recon_loss.item(), kl_loss.item())

            global_step += 1

        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"total={log['total']:.4f}  recon={log['recon']:.4f}  kl={log['kl']:.4f}  "
            f"beta_eff={beta_eff:.3f}"
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved VAE checkpoint to {checkpoint_path}")

    model.eval()
    return model, history


def train_vae_from_loader(
    loader,
    epochs: int = VAE_EPOCHS,
    lr: float = VAE_LR,
    beta: float = VAE_BETA,
    checkpoint_path: str = VAE_CHECKPOINT,
    beta_warmup_epochs: int = 5,
    progress_cb: Callable[[int, int, float, float, float], None] | None = None,
) -> tuple["AudioVAE", list[dict]]:
    """Train AudioVAE from a pre-built DataLoader (accepts any dataset type).

    Identical training logic to train_vae but the caller constructs the loader,
    allowing ConcatDataset or any custom dataset to be passed in.
    Automatically uses CUDA if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = AudioVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []

    model.train()
    global_step = 0
    for epoch in range(epochs):
        beta_eff = beta * min(1.0, (epoch + 1) / max(1, beta_warmup_epochs))

        for batch in loader:
            mel_batch = batch[0].to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(mel_batch)
            total, recon_loss, kl_loss = audio_vae_loss(recon, mel_batch, mu, logvar, beta_eff)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            log = {
                "epoch": epoch,
                "step": global_step,
                "total": total.item(),
                "recon": recon_loss.item(),
                "kl": kl_loss.item(),
            }
            history.append(log)

            if progress_cb is not None:
                progress_cb(epoch, global_step, total.item(), recon_loss.item(), kl_loss.item())

            global_step += 1

        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"total={log['total']:.4f}  recon={log['recon']:.4f}  kl={log['kl']:.4f}  "
            f"beta_eff={beta_eff:.3f}"
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.cpu().state_dict(), checkpoint_path)
    print(f"Saved VAE checkpoint to {checkpoint_path}")

    model.eval()
    return model, history


def load_vae(path: str = VAE_CHECKPOINT) -> AudioVAE:
    """Load a saved AudioVAE from disk. Returns model in eval() mode."""
    model = AudioVAE()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
