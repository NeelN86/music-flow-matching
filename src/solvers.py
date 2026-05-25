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
    pca: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate the velocity field on a regular 2D grid.

    When pca is provided the grid lives in PCA-projected 2D space: grid points are
    back-projected to the full latent space before the model forward pass, and the
    resulting velocities are projected back to 2D for display.

    Args:
        model:  Trained VelocityMLP.
        t:      Time in [0, 1].
        bounds: (x_min, x_max, y_min, y_max) in the 2D display space.
        res:    Grid resolution (res × res points).
        style:  [1, D] or [res*res, D] style vector. None -> zeros.
        pca:    (mean [D], components [2, D]) torch tensors for 16D↔2D projection.
                None assumes the model already operates in 2D.

    Returns:
        (X, Y, U, V) each [res, res] — 2D grid coordinates and velocity components.
    """
    model.eval()
    x_min, x_max, y_min, y_max = bounds
    xs = torch.linspace(x_min, x_max, res)
    ys = torch.linspace(y_min, y_max, res)

    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid_2d = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [G, 2]

    if pca is not None:
        pca_mean, pca_components = pca           # [D], [2, D]
        grid = grid_2d @ pca_components + pca_mean.unsqueeze(0)  # [G, D]
    else:
        grid = grid_2d                           # [G, 2]

    G = grid.shape[0]
    t_batch = torch.full((G,), t)
    grid_style = style.expand(G, -1) if style is not None else None

    v = model(grid, t_batch, grid_style)         # [G, D]

    if pca is not None:
        _, pca_components = pca
        v_2d = v @ pca_components.T              # [G, 2]
    else:
        v_2d = v

    U = v_2d[:, 0].reshape(res, res)
    V = v_2d[:, 1].reshape(res, res)
    return X, Y, U, V
