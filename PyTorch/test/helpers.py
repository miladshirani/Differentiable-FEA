"""Shared helpers for the test-suite: tiny problems that solve in a fraction of a second."""
import torch
from scr.Problem import build_problem, default_spec

DTYPE = torch.float64


def small_problem(element_type="quad4", load=(60.0, 0.0), h=0.5, geometry="rectangle",
                  mesher="structured", **overrides):
    """A small cantilever-like problem for ``element_type`` (structured mesh, no Gmsh needed)."""
    spec = default_spec(geometry, element_type, h)
    spec.update(mesher=mesher, load_value=load, E=2600.0, nu=0.3)
    spec.update(overrides)
    params = build_problem(spec)
    theta = torch.ones(params["conn"].shape[0], dtype=DTYPE)
    return params, theta


def dense(op, n):
    """Assemble the dense matrix of a linear operator by applying it to unit vectors (tests only!)."""
    eye = torch.eye(n, dtype=DTYPE)
    return torch.stack([op(eye[i]) for i in range(n)], dim=1)
