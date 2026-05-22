"""Tests for src/solvers.py — Euler integration and velocity grid."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch


def test_euler_trajectory_shape():
    """euler_integrate returns [n_steps+1, B, 2]."""
    raise NotImplementedError


def test_euler_initial_condition():
    """trajectory[0] == x0 exactly."""
    raise NotImplementedError


def test_euler_constant_field():
    """For a constant velocity field v=(1,0), Euler is exact: x(t) = x0 + t*(1,0)."""
    raise NotImplementedError


def test_euler_style_broadcast():
    """style [1,2] broadcasts to B particles without shape error."""
    raise NotImplementedError


def test_quiver_grid_shape():
    """velocity_field_on_grid returns (X,Y,U,V) each [res, res]."""
    raise NotImplementedError


def test_quiver_grid_finite():
    """All quiver grid values must be finite."""
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
