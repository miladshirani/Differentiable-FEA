"""Material library: energies, objectivity, consistency of force with energy, fibre behaviour."""
import math
import pytest
import torch
from torch.func import grad

from helpers import small_problem, dense, DTYPE
from difffea.materials import (MATERIAL_LIBRARY, material_names, psi_2d, material_vector,
                           embed_plane_strain)
from difffea.constitutive_relations import neo_Hookean_energy
from difffea.operators import global_residual, make_A_operator
from difffea.newton_krylov import newton_krylov_solve, solve_newton_krylov
from difffea.postprocess import compute_fields, support_reactions
from difffea.sensitivity import compute_sensitivity

F_SAMPLE = torch.tensor([[1.12, 0.07], [-0.04, 0.93]], dtype=DTYPE)


def rotation(a):
    return torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=DTYPE)


def test_library_neo_hookean_equals_reference_energy():
    p, _ = material_vector("neo_hookean", {"mu": 1.3, "lmbda": 0.7})
    assert abs(psi_2d("neo_hookean")(F_SAMPLE, p) - neo_Hookean_energy(F_SAMPLE, 1.3, 0.7)) < 1e-12


@pytest.mark.parametrize("name", material_names())
def test_reference_state_is_stress_free(name):
    p, _ = material_vector(name, {"d": 0.2} if name == "fiber_reinforced" else None)
    psi = psi_2d(name)
    I = torch.eye(2, dtype=DTYPE)
    assert abs(psi(I, p)) < 1e-12
    assert grad(psi, argnums=0)(I, p).abs().max() < 1e-10


@pytest.mark.parametrize("name", material_names())
def test_objectivity(name):
    """psi(Q F) = psi(F) for every rotation Q."""
    p, _ = material_vector(name)
    psi = psi_2d(name)
    assert abs(psi(rotation(0.9) @ F_SAMPLE, p) - psi(F_SAMPLE, p)) < 1e-10


@pytest.mark.parametrize("name", material_names())
def test_internal_force_is_gradient_of_total_energy(name):
    """R_int(u) = d(total strain energy)/du : checks stress, quadrature and scatter together."""
    p_, theta = small_problem("quad4", material=name, h=1.0)
    p = dict(p_, m=torch.ones_like(p_["m"]), f=torch.zeros_like(p_["f"]))
    psi = psi_2d(name)
    pv = p["mat_p"]
    conn, dN, w = p["conn"], p["dN_dX"], p["w_detJ"]
    N = p["nodes"].shape[0]

    def total_energy(u):
        ue = u.reshape(N, 2)[conn]                                    # (n_el, nen, 2)
        gu = torch.einsum("eai,eqaj->eqij", ue, dN)                    # (n_el, n_q, 2, 2)
        F = torch.eye(2, dtype=DTYPE) + gu
        e = torch.vmap(torch.vmap(lambda F_: psi(F_, pv)))(F)          # (n_el, n_q)
        return torch.sum(w * e)

    g = torch.Generator().manual_seed(3)
    u = 0.04 * torch.randn(2 * N, generator=g, dtype=DTYPE)
    assert torch.allclose(grad(total_energy)(u), global_residual(u, theta, p), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("name", material_names())
def test_solve_equilibrium_symmetry_and_stress_fields(name):
    p, theta = small_problem("quad4", material=name, load=(0.0, -15.0))
    u, info = newton_krylov_solve(p, theta, n_load_steps=3, tol=1e-9, verbose=False)
    assert info["converged"]
    A = dense(make_A_operator(u, theta, p), u.numel())
    assert (A - A.T).abs().max() < 1e-7
    f = compute_fields(u, theta, p)
    assert torch.isfinite(f["mises"]).all() and f["min_J"] > 0
    assert torch.allclose(support_reactions(u, theta, p) + p["f"].reshape(-1, 2).sum(0),
                          torch.zeros(2, dtype=DTYPE), atol=1e-7)


def test_fibres_carry_load_only_in_tension():
    """Fibres along x: tension adds energy to the matrix, compression adds none."""
    mat = {"beta_deg": 0.0, "d": 0.0}
    p_fib, _ = material_vector("fiber_reinforced", mat)
    p_matrix = p_fib.clone()
    p_matrix[2] = 1e-12                                              # k1 -> 0: matrix only
    psi = psi_2d("fiber_reinforced")
    tension, compression = torch.diag(torch.tensor([1.10, 1.0], dtype=DTYPE)), torch.diag(torch.tensor([0.90, 1.0], dtype=DTYPE))
    assert psi(tension, p_fib) > psi(tension, p_matrix) + 1e-3
    assert abs(psi(compression, p_fib) - psi(compression, p_matrix)) < 1e-9


def test_fibre_direction_makes_the_response_anisotropic():
    """Stretching along the fibres stores much more energy than across them."""
    psi = psi_2d("fiber_reinforced")
    p, _ = material_vector("fiber_reinforced", {"beta_deg": 0.0})
    along = psi(torch.diag(torch.tensor([1.10, 1.0], dtype=DTYPE)), p)
    across = psi(torch.diag(torch.tensor([1.0, 1.10], dtype=DTYPE)), p)
    assert along > 2.0 * across


def test_adjoint_sensitivity_with_fibre_material():
    p, _ = small_problem("quad4", material="fiber_reinforced", load=(60.0, 0.0), h=1.0)
    n_el = p["conn"].shape[0]
    theta = 1.0 + 0.2 * torch.sin(torch.arange(n_el, dtype=DTYPE))
    loss_fn = lambda u: torch.dot(p["f"], u)
    _, g = compute_sensitivity(p, theta, loss_fn, n_load_steps=2, tol=1e-12)
    eps, g_fd = 1e-6, torch.zeros_like(theta)
    for i in range(n_el):
        tp, tm = theta.clone(), theta.clone()
        tp[i] += eps
        tm[i] -= eps
        g_fd[i] = (loss_fn(solve_newton_krylov(p, tp, 2, 1e-12, verbose=False)) -
                   loss_fn(solve_newton_krylov(p, tm, 2, 1e-12, verbose=False))) / (2 * eps)
    assert torch.norm(g - g_fd) / torch.norm(g_fd) < 1e-6


def test_unknown_material_parameter_is_rejected():
    with pytest.raises(ValueError):
        material_vector("fiber_reinforced", {"k3": 1.0})
