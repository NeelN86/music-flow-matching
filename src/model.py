# Style-conditioned VelocityMLP for audio flow matching.
#
# Direct port of the prior project's VelocityMLP (flow-matching-visualizer/src/model.py).
# The only architectural change: input_dim 3→5 to accommodate style conditioning.
#
# Input layout: [B, 5] = (x, y, t, style_x, style_y)
#   style = 2D VAE mu of the conditioning audio (the "learned embedding" is
#           learned by the VAE encoder — no separate style MLP needed)
#   style=None → zeros (unconditional mode, used in tests and visualization)
#
# The flow matching math is identical to the prior project:
#   x_t = (1-t)*x0 + t*x1,  target velocity = x1 - x0
#   model learns v(x,t,style) such that dx/dt = v(x,t,style)

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.config import FLOW_HIDDEN_WIDTH, FLOW_N_HIDDEN


class VelocityMLP(nn.Module):
    """Style-conditioned velocity field MLP.

    3 hidden layers, width 256, SiLU activation — identical structure to the
    prior toy flow matching project.

    Args:
        hidden_width: Width of each hidden layer.
        n_hidden:     Number of hidden layers.
    """

    def __init__(
        self,
        hidden_width: int = FLOW_HIDDEN_WIDTH,
        n_hidden: int = FLOW_N_HIDDEN,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        style: Tensor | None = None,
    ) -> Tensor:
        """Predict velocity at position x and time t, conditioned on style.

        Args:
            x:     [B, 2] spatial coordinates in 2D latent space.
            t:     [B] or scalar, times in [0, 1].
            style: [B, 2] or [1, 2] VAE mu of the conditioning audio.
                   Broadcast to batch dim if shape is [1, 2].
                   If None, uses zeros (unconditional).

        Returns:
            [B, 2] predicted velocity (vx, vy).
        """
        raise NotImplementedError
