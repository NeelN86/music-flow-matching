# ODE integration and velocity grid utilities.
#
# Port of flow-matching-visualizer/src/solvers.py with a `style` parameter added.
#
# Two roles:
#   1. euler_integrate — pre-computes the full particle trajectory before
#      animation starts. Must use uniform time steps (FuncAnimation needs frame i
#      at time i*dt). Custom Euler (not torchdiffeq) for this reason.
#
#   2. velocity_field_on_grid — evaluates the velocity field on a regular grid
#      for quiver plot rendering. Called once per animation frame.
#
# torchdiffeq.odeint is available in requirements for optional high-quality
# generation (dopri5 adaptive solver) but is not used here.

from __future__ import annotations

import torch
from torch import Tensor

from src.config import EULER_STEPS
from src.model import VelocityMLP


@torch.no_grad()
def euler_integrate(
    model: VelocityMLP,
    x0: Tensor,
    n_steps: int = EULER_STEPS,
    t_span: tuple[float, float] = (0.0, 1.0),
    style: Tensor | None = None,
) -> Tensor:
    """Euler ODE integration with optional style conditioning.

    x_{k+1} = x_k + dt * v(x_k, t_k, style)

    Args:
        model:   Trained VelocityMLP.
        x0:      [B, 2] initial positions (noise samples).
        n_steps: Number of Euler steps.
        t_span:  (t_start, t_end) integration interval.
        style:   [1, 2] or [B, 2] style conditioning. Broadcast if [1, 2].
                 None → unconditional (zeros).

    Returns:
        trajectory: [n_steps+1, B, 2]  (includes x0 as first frame).
    """
    model.eval()
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    B = x0.shape[0]

    x = x0.clone()
    trajectory = [x]

    for i in range(n_steps):
        t = t0 + i * dt
        t_batch = torch.full((B,), t, dtype=x.dtype)
        v = model(x, t_batch, style)
        x = x + dt * v
        trajectory.append(x)

    return torch.stack(trajectory, dim=0)  # [n_steps+1, B, 2]


@torch.no_grad()
def velocity_field_on_grid(
    model: VelocityMLP,
    t: float,
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    res: int = 20,
    style: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate the velocity field on a regular 2D grid.

    Args:
        model:  Trained VelocityMLP.
        t:      Time in [0, 1].
        bounds: (x_min, x_max, y_min, y_max).
        res:    Grid resolution (res × res points).
        style:  [1, 2] style vector, broadcast to all grid points. None -> zeros.

    Returns:
        (X, Y, U, V) each [res, res] — grid coordinates and velocity components.
    """
    model.eval()
    x_min, x_max, y_min, y_max = bounds
    xs = torch.linspace(x_min, x_max, res)
    ys = torch.linspace(y_min, y_max, res)

    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [res*res, 2]

    t_batch = torch.full((grid.shape[0],), t)

    # Broadcast style [1, 2] → [res*res, 2] if provided
    grid_style = style.expand(grid.shape[0], -1) if style is not None else None

    v = model(grid, t_batch, grid_style)  # [res*res, 2]

    U = v[:, 0].reshape(res, res)
    V = v[:, 1].reshape(res, res)
    return X, Y, U, V
