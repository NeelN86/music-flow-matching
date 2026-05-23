"""Tests for src/model.py — VelocityMLP."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from src.model import VelocityMLP


def test_output_shape_with_style():
    """VelocityMLP(x[B,2], t[B], style[B,2]) → [B,2]."""
    B = 8
    model = VelocityMLP()
    x = torch.randn(B, 2)
    t = torch.rand(B)
    style = torch.randn(B, 2)
    out = model(x, t, style)
    assert out.shape == (B, 2), f"expected ({B}, 2), got {out.shape}"


def test_output_shape_without_style():
    """VelocityMLP(x[B,2], t[B], style=None) → [B,2] (unconditional zeros)."""
    B = 5
    model = VelocityMLP()
    x = torch.randn(B, 2)
    t = torch.rand(B)
    out = model(x, t, style=None)
    assert out.shape == (B, 2), f"expected ({B}, 2), got {out.shape}"


def test_output_is_finite():
    """All output values must be finite (no NaN/Inf on random input)."""
    model = VelocityMLP()
    x = torch.randn(16, 2)
    t = torch.rand(16)
    style = torch.randn(16, 2)
    out = model(x, t, style)
    assert torch.isfinite(out).all(), "output contains NaN or Inf"


def test_gradients_flow():
    """All parameters receive gradients after a forward+backward pass."""
    model = VelocityMLP()
    x = torch.randn(4, 2)
    t = torch.rand(4)
    style = torch.randn(4, 2)
    loss = model(x, t, style).sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_time_sensitivity():
    """Different t values produce different velocity outputs (same x, same style)."""
    model = VelocityMLP()
    model.eval()
    x = torch.randn(1, 2).expand(2, -1)  # same position
    style = torch.zeros(2, 2)
    t_a = torch.tensor([0.1, 0.1])
    t_b = torch.tensor([0.9, 0.9])
    out_a = model(x, t_a, style)
    out_b = model(x, t_b, style)
    assert not torch.allclose(out_a, out_b), "output identical for different t — model ignores time"


def test_style_sensitivity():
    """Different style values produce different velocity outputs (same x, same t)."""
    model = VelocityMLP()
    model.eval()
    x = torch.randn(1, 2).expand(2, -1)
    t = torch.tensor([0.5, 0.5])
    style_a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    style_b = torch.tensor([[-1.0, 2.0], [-1.0, 2.0]])
    out_a = model(x, t, style_a)
    out_b = model(x, t, style_b)
    assert not torch.allclose(out_a, out_b), "output identical for different style — model ignores style"


def test_style_broadcast():
    """style [1,2] broadcasts correctly to batch size B."""
    B = 6
    model = VelocityMLP()
    x = torch.randn(B, 2)
    t = torch.rand(B)
    style_single = torch.randn(1, 2)         # [1, 2] — should broadcast
    style_expanded = style_single.expand(B, -1)  # [B, 2]
    out_broadcast = model(x, t, style_single)
    out_expanded = model(x, t, style_expanded)
    assert torch.allclose(out_broadcast, out_expanded), "broadcast style mismatch"


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
