"""Tests for src/model.py — VelocityMLP."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch


def test_output_shape_with_style():
    """VelocityMLP(x[B,2], t[B], style[B,2]) → [B,2]."""
    raise NotImplementedError


def test_output_shape_without_style():
    """VelocityMLP(x[B,2], t[B], style=None) → [B,2] (unconditional zeros)."""
    raise NotImplementedError


def test_output_is_finite():
    """All output values must be finite (no NaN/Inf on random input)."""
    raise NotImplementedError


def test_gradients_flow():
    """All parameters receive gradients after a forward+backward pass."""
    raise NotImplementedError


def test_time_sensitivity():
    """Different t values produce different velocity outputs (same x, same style)."""
    raise NotImplementedError


def test_style_sensitivity():
    """Different style values produce different velocity outputs (same x, same t)."""
    raise NotImplementedError


def test_style_broadcast():
    """style [1,2] broadcasts correctly to batch size B."""
    raise NotImplementedError


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
