"""
Invariants of the matrix-free operators, checked for EVERY element type.

These identities are the "unit tests of the mathematics": if any of them fails,
a plot made with the code cannot be trusted.
"""
import math
import pytest
import torch
from torch.func import jvp

from helpers import small_problem, dense, DTYPE
from difffea.elements import element_names
from difffea.materials import psi_2d
from difffea.operators import (global_residual, elem_residual, elem_residual_precomputed,
                           precompute_geometry, make_A_operator, make_AT_operator,
                           compute_jacobi_diagonal, make_jacobi_preconditioner)
from difffea.kinematics import Coordinate_Interpolation
from difffea.newton_krylov import pcg, solve_newton_krylov


@pytest.mark.parametrize("name", element_names())
def test_precomputed_element_force_equals_reference(name):
    """The fast (cached-geometry) element force equals the virtual-work reference."""
    p, _ = small_problem(name, h=1.0)
    e = 0
    Xe = p["nodes"][p["conn"][e]]
    g = torch.Generator().manual_seed(1)
    ue = 0.05 * torch.randn(Xe.shape, generator=g, dtype=DTYPE)
    ref = elem_residual(ue, Xe, 1000.0, 2000.0, p["shape_fn"], p["GP"], p["GW"])
    fast = elem_residual_precomputed(ue, p["dN_dX"][e], p["w_detJ"][e],
                                     torch.tensor([1000.0, 2000.0], dtype=DTYPE), psi_2d("neo_hookean"))
    assert torch.allclose(ref, fast, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("name", element_names())
def test_rigid_body_invariance(name):
    """Rigid translations and rotations produce no internal force."""
    p, theta = small_problem(name)
    p0 = dict(p, f=torch.zeros_like(p["f"]))
    n = p["nodes"].shape[0]
    t = torch.zeros(2 * n, dtype=DTYPE)
    t[0::2], t[1::2] = 0.3, -0.2
    assert global_residual(t, theta, p0).abs().max() < 1e-9
    a = 0.7
    Q = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=DTYPE)
    X = p["nodes"]
    u_rot = (X @ Q.T - X).reshape(-1)
    # the mask zeroes the clamped rows, so compare on free DOFs only (they carry all forces)
    assert global_residual(u_rot, theta, p0).abs().max() < 1e-8


@pytest.mark.parametrize("name", element_names())
def test_tangent_operator_properties(name):
    """A is symmetric, SPD at equilibrium; AT = A^T; jvp = finite difference; diag and pcg exact."""
    p, theta = small_problem(name)
    u = solve_newton_krylov(p, theta, n_load_steps=2, tol=1e-10, verbose=False)
    n = u.numel()
    assert torch.norm(global_residual(u, theta, p)) < 1e-9          # it IS an equilibrium

    A = dense(make_A_operator(u, theta, p), n)
    AT = dense(make_AT_operator(u, theta, p), n)
    assert (A - A.T).abs().max() < 1e-8                              # symmetry of hyperelastic tangent
    assert (AT - A.T).abs().max() < 1e-8                             # vjp-based transpose operator
    assert torch.linalg.eigvalsh(A).min() > 0                        # SPD: stable equilibrium
    d = compute_jacobi_diagonal(u, theta, p)
    assert (d - torch.diag(A)).abs().max() < 1e-8                    # matrix-free diagonal

    b = torch.randn(n, dtype=DTYPE) * p["m"]
    x, _ = pcg(make_A_operator(u, theta, p), b, make_jacobi_preconditioner(u, theta, p),
               rtol=1e-12, max_iter=2000)
    assert (x - torch.linalg.solve(A, b)).abs().max() < 1e-8         # pcg == dense solve

    v = torch.randn(n, dtype=DTYPE) * p["m"]                         # jvp vs central difference
    _, Jv = jvp(lambda w: global_residual(w, theta, p), (u,), (v,))
    eps = 1e-6
    fd = (global_residual(u + eps * v, theta, p) - global_residual(u - eps * v, theta, p)) / (2 * eps)
    assert (Jv - fd).abs().max() < 1e-4 * max(1.0, Jv.abs().max())
