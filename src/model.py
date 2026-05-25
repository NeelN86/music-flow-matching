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

from src.config import FLOW_HIDDEN_WIDTH, FLOW_N_HIDDEN, VAE_LATENT_DIM


class VelocityMLP(nn.Module):
    """Style-conditioned velocity field MLP.

    3 hidden layers, width 256, SiLU activation.

    Input:  [B, 2*latent_dim+1] = cat(position[latent_dim], t[1], style[latent_dim])
    Output: [B, latent_dim] predicted velocity

    Args:
        latent_dim:   Dimension of the latent space (defaults to VAE_LATENT_DIM).
        hidden_width: Width of each hidden layer.
        n_hidden:     Number of hidden layers.
    """

    def __init__(
        self,
        latent_dim: int = VAE_LATENT_DIM,
        hidden_width: int = FLOW_HIDDEN_WIDTH,
        n_hidden: int = FLOW_N_HIDDEN,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = 2 * latent_dim + 1  # position + time + style
        layers: list[nn.Module] = []
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_width), nn.SiLU()]
            in_dim = hidden_width
        layers.append(nn.Linear(hidden_width, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        style: Tensor | None = None,
    ) -> Tensor:
        """Predict velocity at position x and time t, conditioned on style.

        Args:
            x:     [B, latent_dim] coordinates in latent space.
            t:     [B] or scalar, times in [0, 1].
            style: [B, latent_dim] or [1, latent_dim] VAE mu of the conditioning audio.
                   Broadcast to batch dim if [1, latent_dim].
                   If None, uses zeros (unconditional).

        Returns:
            [B, latent_dim] predicted velocity.
        """
        B = x.shape[0]

        # Normalise t to [B, 1]
        t_col = t.view(-1, 1).expand(B, 1) if t.numel() > 1 else t.view(1, 1).expand(B, 1)

        if style is None:
            style = torch.zeros(B, self.latent_dim, device=x.device, dtype=x.dtype)
        else:
            style = style.expand(B, -1)

        inp = torch.cat([x, t_col, style], dim=1)  # [B, 2*latent_dim+1]
        return self.net(inp)
