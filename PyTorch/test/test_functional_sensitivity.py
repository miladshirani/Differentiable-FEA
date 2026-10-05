"""Adjoint (implicit-function-theorem) sensitivities match central finite differences."""
import pytest
import torch

from helpers import small_problem, DTYPE
from scr.Newton_Krylov import solve_newton_krylov
from scr.Sensitivity import compute_sensitivity


@pytest.mark.parametrize("name", ["quad4", "tri3", "quad8"])
def test_adjoint_gradient_matches_finite_differences(name):
    p, _ = small_problem(name, load=(50.0, 0.0), h=1.0)
    n_el = p["conn"].shape[0]
    theta = 1.0 + 0.2 * torch.sin(torch.arange(n_el, dtype=DTYPE))
    loss_fn = lambda u: torch.dot(p["f"], u)                 # compliance

    _, g = compute_sensitivity(p, theta, loss_fn, n_load_steps=2, tol=1e-12)

    eps, g_fd = 1e-6, torch.zeros_like(theta)
    for i in range(n_el):
        tp, tm = theta.clone(), theta.clone()
        tp[i] += eps
        tm[i] -= eps
        Lp = loss_fn(solve_newton_krylov(p, tp, 2, 1e-12, verbose=False))
        Lm = loss_fn(solve_newton_krylov(p, tm, 2, 1e-12, verbose=False))
        g_fd[i] = (Lp - Lm) / (2 * eps)

    assert torch.norm(g - g_fd) / torch.norm(g_fd) < 1e-6
