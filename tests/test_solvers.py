"""Tests for src/solvers.py — Euler integration and velocity grid."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from src.config import VAE_LATENT_DIM
from src.solvers import euler_integrate, velocity_field_on_grid
from src.model import VelocityMLP

D = VAE_LATENT_DIM


class _ConstantVelocity2D(nn.Module):
    """Toy 2D model that always returns velocity (1, 0) — for exact Euler tests."""
    def forward(self, x: torch.Tensor, t: torch.Tensor, style=None) -> torch.Tensor:
        return torch.zeros_like(x) + torch.tensor([1.0, 0.0])


def test_euler_trajectory_shape():
    """euler_integrate returns [n_steps+1, B, D]."""
    model = VelocityMLP()
    B, n_steps = 4, 50
    x0 = torch.randn(B, D)
    traj = euler_integrate(model, x0, n_steps=n_steps)
    assert traj.shape == (n_steps + 1, B, D), f"expected ({n_steps+1}, {B}, {D}), got {traj.shape}"


def test_euler_initial_condition():
    """trajectory[0] == x0 exactly."""
    model = VelocityMLP()
    x0 = torch.randn(4, D)
    traj = euler_integrate(model, x0, n_steps=10)
    assert torch.equal(traj[0], x0), "trajectory[0] does not match x0"


def test_euler_constant_field():
    """For a constant 2D velocity field v=(1,0), Euler is exact: x(t) = x0 + t*(1,0)."""
    model = _ConstantVelocity2D()
    x0 = torch.zeros(3, 2)
    traj = euler_integrate(model, x0, n_steps=50)
    expected = x0 + torch.tensor([1.0, 0.0])
    assert torch.allclose(traj[-1], expected, atol=1e-5), (
        f"constant-field Euler mismatch: got {traj[-1]}, expected {expected}"
    )


def test_euler_style_broadcast():
    """style [1,D] broadcasts to B particles without shape error."""
    model = VelocityMLP()
    B = 6
    x0 = torch.randn(B, D)
    style = torch.randn(1, D)
    traj = euler_integrate(model, x0, n_steps=10, style=style)
    assert traj.shape == (11, B, D)


def _mock_pca(latent_dim: int = D) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity-like PCA: maps first 2 latent dims to 2D display space."""
    mean = torch.zeros(latent_dim)
    components = torch.zeros(2, latent_dim)
    components[0, 0] = 1.0
    components[1, 1] = 1.0
    return mean, components


def test_quiver_grid_shape():
    """velocity_field_on_grid returns (X,Y,U,V) each [res, res]."""
    model = VelocityMLP()
    res = 15
    X, Y, U, V = velocity_field_on_grid(model, t=0.5, res=res, pca=_mock_pca())
    for name, arr in [("X", X), ("Y", Y), ("U", U), ("V", V)]:
        assert arr.shape == (res, res), f"{name}: expected ({res}, {res}), got {arr.shape}"


def test_quiver_grid_finite():
    """All quiver grid values must be finite."""
    model = VelocityMLP()
    X, Y, U, V = velocity_field_on_grid(model, t=0.3, res=10, pca=_mock_pca())
    for name, arr in [("X", X), ("Y", Y), ("U", U), ("V", V)]:
        assert torch.isfinite(arr).all(), f"{name} contains non-finite values"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except NotImplementedError:
                print(f"SKIP  {name}  (not implemented)")
            except Exception as e:
                print(f"FAIL  {name}  {e}")
