# Flow matching training loop.
#
# Trains VelocityMLP to predict the velocity field that transports Gaussian
# noise x0 toward audio latent points x1, conditioned on the input audio's
# style (VAE mu).
#
# Training procedure per step:
#   1. Sample mel batch from NSynth dataloader.
#   2. Encode: mu, logvar = vae.encode(mel)  [VAE is frozen, no gradients]
#   3. x1 = mu + eps * exp(0.5 * logvar)     [reparameterized target]
#   4. style = mu                             [deterministic conditioning]
#   5. x0 ~ N(0, I)
#   6. t ~ U[0, 1]
#   7. x_t = (1-t)*x0 + t*x1                [linear interpolation — same as prior project]
#   8. loss = MSE(model(x_t, t, style), x1 - x0)

from __future__ import annotations

import os
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, FLOW_BATCH_SIZE, FLOW_CHECKPOINT, FLOW_LR, FLOW_STEPS
from src.model import VelocityMLP
from src.vae import AudioVAE


def train_flow(
    vae: AudioVAE,
    dataloader: DataLoader,
    steps: int = FLOW_STEPS,
    lr: float = FLOW_LR,
    checkpoint_path: str = FLOW_CHECKPOINT,
    progress_cb: Callable[[int, float], None] | None = None,
) -> tuple[VelocityMLP, list[float]]:
    """Train the style-conditioned VelocityMLP via flow matching.

    Args:
        vae:             Trained AudioVAE in eval() mode. Weights stay frozen.
        dataloader:      Yields mel batches [B, 1, N_MELS, N_FRAMES].
        steps:           Total gradient steps (dataloader cycles as needed).
        lr:              Adam learning rate.
        checkpoint_path: Destination for the saved model.
        progress_cb:     Called as cb(step, loss) every step.

    Returns:
        (model in eval() mode, loss history list).
    """
    model = VelocityMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    vae.eval()
    model.train()

    data_iter = _cycle(dataloader)

    for step in range(steps):
        batch = next(data_iter)
        mels = batch[0]  # [B, 1, N_MELS, N_FRAMES]

        # Encode with frozen VAE — no gradient through it
        with torch.no_grad():
            mu, logvar = vae.encode(mels)

        # x1: reparameterized latent; style: deterministic mu
        std = (0.5 * logvar).exp()
        x1 = mu + std * torch.randn_like(std)   # [B, 2]
        style = mu                               # [B, 2]

        B = x1.shape[0]
        x0 = torch.randn(B, 2)                  # noise source
        t = torch.rand(B)

        t_col = t.view(-1, 1)
        x_t = (1.0 - t_col) * x0 + t_col * x1  # linear interpolation
        v_target = x1 - x0                      # conditional flow target velocity

        v_pred = model(x_t, t, style)
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scalar = loss.item()
        loss_history.append(scalar)

        if progress_cb is not None:
            progress_cb(step, scalar)

        if (step + 1) % 100 == 0:
            print(f"step {step + 1:5d}/{steps}  loss={scalar:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved flow model to {checkpoint_path}")

    model.eval()
    return model, loss_history


def _cycle(dataloader: DataLoader) -> Iterator[Tensor]:
    """Infinite iterator over a DataLoader — restarts when exhausted."""
    while True:
        yield from dataloader
