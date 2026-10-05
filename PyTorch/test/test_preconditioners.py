"""Preconditioners: correctness of the blocks / assembled matrix, and the iteration savings."""
import pytest
import torch

from helpers import small_problem, dense, DTYPE
from scr.Elements import element_names
from scr.Operators import (PRECONDITIONERS, make_A_operator, make_preconditioner,
                           make_block_jacobi_preconditioner, assemble_tangent_sparse)
from scr.Newton_Krylov import pcg, newton_krylov_solve
from scr.Problem import to_device


@pytest.mark.parametrize("name", element_names())
def test_assembled_sparse_tangent_equals_matrix_free_operator(name):
    p, theta = small_problem(name, h=1.0)
    u = 0.01 * torch.randn(2 * p["nodes"].shape[0], dtype=DTYPE, generator=torch.Generator().manual_seed(2))
    A = dense(make_A_operator(u, theta, p), u.numel())
    K = torch.tensor(assemble_tangent_sparse(u, theta, p).toarray())
    assert (A - K).abs().max() < 1e-8 * A.abs().max()


@pytest.mark.parametrize("name", ["quad4", "tri6"])
def test_block_jacobi_applies_inverse_of_nodal_blocks(name):
    p, theta = small_problem(name, h=1.0)
    n = 2 * p["nodes"].shape[0]
    u = torch.zeros(n, dtype=DTYPE)
    A = dense(make_A_operator(u, theta, p), n)
    M = make_block_jacobi_preconditioner(u, theta, p)
    r = torch.randn(n, dtype=DTYPE, generator=torch.Generator().manual_seed(5))
    # reference: invert the 2x2 diagonal blocks of the dense matrix
    expected = torch.zeros_like(r)
    for a in range(n // 2):
        B = A[2 * a:2 * a + 2, 2 * a:2 * a + 2]
        expected[2 * a:2 * a + 2] = torch.linalg.solve(B, r[2 * a:2 * a + 2])
    assert torch.allclose(M(r), expected, atol=1e-8, rtol=1e-8)


def test_all_preconditioners_give_the_same_solution_and_ilu_needs_far_fewer_iterations():
    pytest.importorskip("gmsh")
    p, theta = small_problem("quad4", load=(300.0, 0.0), h=0.2, geometry="plate_hole", mesher="gmsh")
    tol = 1e-9 * float(torch.norm(p["f"]))
    out = {k: newton_krylov_solve(p, theta, 3, tol, verbose=False, preconditioner=k) for k in PRECONDITIONERS}
    ref = out["jacobi"][0]
    for k, (u, info) in out.items():
        assert info["converged"], k
        assert (u - ref).abs().max() < 1e-6, k
    assert out["ilu"][1]["total_cg"] * 10 < out["jacobi"][1]["total_cg"]


def test_unknown_preconditioner_is_rejected():
    p, theta = small_problem("quad4")
    with pytest.raises(ValueError):
        make_preconditioner("magic", torch.zeros(2 * p["nodes"].shape[0], dtype=DTYPE), theta, p)


def test_to_device_roundtrip_keeps_values():
    p, _ = small_problem("quad4")
    q = to_device(p, "cpu")
    assert q is not p and torch.equal(q["conn"], p["conn"]) and q["shape_fn"] is p["shape_fn"]
