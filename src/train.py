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

from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.config import FLOW_BATCH_SIZE, FLOW_CHECKPOINT, FLOW_LR, FLOW_STEPS
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
    raise NotImplementedError


def _cycle(dataloader: DataLoader) -> Iterator[Tensor]:
    """Infinite iterator over a DataLoader — restarts when exhausted."""
    raise NotImplementedError
