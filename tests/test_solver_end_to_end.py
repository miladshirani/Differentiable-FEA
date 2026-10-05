"""End-to-end solves: finite-strain patch test, global equilibrium, adaptive load stepping."""
import pytest
import torch

from helpers import small_problem, DTYPE
from difffea.elements import element_names
from difffea.newton_krylov import newton_krylov_solve
from difffea.operators import global_residual
from difffea.postprocess import compute_fields, support_reactions


@pytest.mark.parametrize("name", element_names())
def test_finite_strain_patch_test(name):
    """
    Impose an AFFINE motion  x = F0 X  on the whole boundary.  A correct element
    must then reproduce it EXACTLY in the interior (F = F0 everywhere), on a
    distorted mesh, for every element type.
    """
    pytest.importorskip("gmsh")
    p, theta = small_problem(name, h=0.45, geometry="plate_hole", mesher="gmsh")
    nodes = p["nodes"]
    F0 = torch.tensor([[1.10, 0.06], [-0.04, 0.95]], dtype=DTYPE)
    u_exact = (nodes @ (F0 - torch.eye(2, dtype=DTYPE)).T).reshape(-1)

    # find boundary nodes: those of boundary edges
    from difffea.boundary_conditions import boundary_edges
    from difffea.elements import get_element
    bnodes = torch.unique(boundary_edges(p["conn"], get_element(name)["edges"]).reshape(-1))
    fixed = torch.zeros(nodes.shape[0], dtype=torch.bool)
    fixed[bnodes] = True
    fixed = fixed.repeat_interleave(2)
    q = dict(p, fixed=fixed, m=torch.where(fixed, 0.0, 1.0).to(DTYPE), u_D=u_exact,
             f=torch.zeros_like(p["f"]))

    u, info = newton_krylov_solve(q, theta, n_load_steps=2, tol=1e-10, verbose=False)
    assert info["converged"]
    assert (u - u_exact).abs().max() < 1e-8


@pytest.mark.parametrize("name", element_names())
def test_global_equilibrium_and_convergence(name):
    p, theta = small_problem(name, load=(0.0, -20.0))
    u, info = newton_krylov_solve(p, theta, n_load_steps=3, tol=1e-9, verbose=False)
    assert info["converged"] and info["load_factor"] == 1.0
    assert torch.norm(global_residual(u, theta, p)) < 1e-9
    reaction = support_reactions(u, theta, p)
    applied = p["f"].reshape(-1, 2).sum(0)
    assert torch.allclose(reaction + applied, torch.zeros(2, dtype=DTYPE), atol=1e-7)
    fields = compute_fields(u, theta, p)
    assert fields["min_J"] > 0 and torch.isfinite(fields["mises"]).all()


def test_adaptive_load_stepping_recovers_from_too_large_steps():
    """A slender L-bracket (6 N tip load) loaded with ONE nominal step needs automatic cut-backs."""
    pytest.importorskip("gmsh")
    p, theta = small_problem("quad4", load=(0.0, -6.0), h=0.3, geometry="l_bracket", mesher="gmsh")
    u, info = newton_krylov_solve(p, theta, n_load_steps=1, tol=1e-8, max_newton=25, verbose=False)
    assert info["converged"]
    assert info["n_cutbacks"] >= 1


def test_displacement_controlled_loading():
    p, theta = small_problem("quad4", load_mode="displacement", load=(0.2, None))
    u, info = newton_krylov_solve(p, theta, n_load_steps=2, tol=1e-9, verbose=False)
    assert info["converged"]
    right = p["nodes"][:, 0] > 2.0 - 1e-9
    assert torch.allclose(u.reshape(-1, 2)[right, 0], torch.full((int(right.sum()),), 0.2, dtype=DTYPE))
