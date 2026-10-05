"""
The three evaluations of the matrix-free tangent product ("jvp", "linearize", "qp") must be the SAME linear
operator: same result for any vector, symmetric, and the same Newton-Krylov solution.  "jvp" is the
reference (forward-mode AD of the residual); "linearize" caches its primal; "qp" stores the Gauss-point
tangent moduli (see operators.py).
"""
import pytest
import torch

from helpers import small_problem, DTYPE
from difffea.elements import element_names
from difffea.materials import material_names
from difffea.operators import make_tangent_operator, TANGENT_OPERATORS
from difffea.newton_krylov import newton_krylov_solve


def _state(params, seed=0):
    """A non-trivial linearisation point (a few per cent strain) and a random direction."""
    g = torch.Generator().manual_seed(seed)
    n = params["f"].shape[0]
    u = 0.02 * torch.randn(n, dtype=DTYPE, generator=g) * params["m"]
    v = torch.randn(n, dtype=DTYPE, generator=g)
    return u, v


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("element_type", element_names())
def test_operators_agree_for_every_element(element_type):
    params, theta = small_problem(element_type, h=0.5)
    u, v = _state(params)
    ref = make_tangent_operator("jvp", u, theta, params)(v)
    for kind in ("linearize", "qp"):
        assert _rel(make_tangent_operator(kind, u, theta, params)(v), ref) < 1e-11, kind


@pytest.mark.parametrize("material", material_names())
def test_operators_agree_for_every_material(material):
    params, theta = small_problem("quad4", material=material, h=1.0)
    u, v = _state(params, seed=1)
    ref = make_tangent_operator("jvp", u, theta, params)(v)
    for kind in ("linearize", "qp"):
        assert _rel(make_tangent_operator(kind, u, theta, params)(v), ref) < 1e-10, kind


def test_operators_respect_nonuniform_theta():
    params, theta = small_problem("quad4", h=0.5)
    theta = 0.5 + torch.rand(theta.shape[0], dtype=DTYPE, generator=torch.Generator().manual_seed(2))
    u, v = _state(params, seed=3)
    ref = make_tangent_operator("jvp", u, theta, params)(v)
    assert _rel(make_tangent_operator("qp", u, theta, params)(v), ref) < 1e-11


@pytest.mark.parametrize("kind", TANGENT_OPERATORS)
def test_every_operator_is_symmetric(kind):
    params, theta = small_problem("tri6", h=0.7)
    u, v = _state(params, seed=4)
    w = torch.randn_like(v)
    A = make_tangent_operator(kind, u, theta, params)
    assert float(w @ A(v)) == pytest.approx(float(v @ A(w)), rel=1e-10)


def test_unknown_operator_name_is_rejected():
    params, theta = small_problem("quad4", h=1.0)
    u, _ = _state(params)
    with pytest.raises(ValueError):
        make_tangent_operator("nope", u, theta, params)


@pytest.mark.parametrize("kind", ["linearize", "qp"])
def test_solver_reaches_the_same_solution(kind):
    params, theta = small_problem("quad9", load=(0.0, -40.0), h=0.5)
    tol = 1e-9 * float(torch.norm(params["f"]))
    u_ref, info_ref = newton_krylov_solve(params, theta, 3, tol, verbose=False, preconditioner="ilu")
    u, info = newton_krylov_solve(params, theta, 3, tol, verbose=False, preconditioner="ilu", tangent=kind)
    assert info["converged"] and info_ref["converged"]
    assert _rel(u, u_ref) < 1e-7
