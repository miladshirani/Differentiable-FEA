import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.func import jvp
from scr.Mesh import create_2d_quad_mesh
from scr.shape_functions_2D import shape_fn_quad4
from scr.Gauss_Quadratures import Gauss_Quadratures_2D
from scr.Operators import (global_residual, make_A_operator, make_AT_operator,
                           compute_jacobi_diagonal, make_jacobi_preconditioner)
from scr.Newton_Krylov import pcg, solve_newton_krylov

dtype = torch.float64


def build_problem(nx=3, ny=2, load=50.0):
    nodes, conn = create_2d_quad_mesh(1.5, 1.0, nx, ny, dtype=dtype)
    n = nodes.shape[0]
    GP, GW = Gauss_Quadratures_2D(2, dtype=dtype)
    left = torch.where(nodes[:, 0] == 0)[0]
    right = torch.where(nodes[:, 0] == 1.5)[0]
    fixed = torch.zeros(2 * n, dtype=torch.bool)
    fixed[2 * left] = True
    fixed[2 * left + 1] = True
    f = torch.zeros(2 * n, dtype=dtype)
    f[2 * right] = load / len(right)
    return {"nodes": nodes, "conn": conn, "shape_fn": shape_fn_quad4, "GP": GP, "GW": GW,
            "mu0": 1000.0, "lmbda0": 2000.0, "f": f,
            "m": torch.where(fixed, 0.0, 1.0).to(dtype), "fixed": fixed,
            "u_D": torch.zeros(2 * n, dtype=dtype)}


def dense(op, n):
    eye = torch.eye(n, dtype=dtype)
    return torch.stack([op(eye[i]) for i in range(n)], dim=1)


def test_operator_invariants():
    p = build_problem()
    theta = torch.ones(p["conn"].shape[0], dtype=dtype)
    u = solve_newton_krylov(p, theta, n_load_steps=3, tol=1e-10, verbose=False)
    n = u.numel()
    assert torch.norm(global_residual(u, theta, p)) < 1e-9

    A = dense(make_A_operator(u, theta, p), n)
    AT = dense(make_AT_operator(u, theta, p), n)
    assert (A - A.T).abs().max() < 1e-9                      # symmetry
    assert (AT - A.T).abs().max() < 1e-9                     # transpose operator
    assert torch.linalg.eigvalsh(A).min() > 0                # SPD
    d = compute_jacobi_diagonal(u, theta, p)
    assert (d - torch.diag(A)).abs().max() < 1e-9            # matrix-free diagonal

    b = torch.randn(n, dtype=dtype) * p["m"]
    x, _ = pcg(make_A_operator(u, theta, p), b, make_jacobi_preconditioner(u, theta, p), rtol=1e-12)
    assert (x - torch.linalg.solve(A, b)).abs().max() < 1e-9  # pcg vs dense

    v = torch.randn(n, dtype=dtype) * p["m"]                  # jvp vs finite difference
    _, Jv = jvp(lambda w: global_residual(w, theta, p), (u,), (v,))
    eps = 1e-6
    fd = (global_residual(u + eps * v, theta, p) - global_residual(u - eps * v, theta, p)) / (2 * eps)
    assert (Jv - fd).abs().max() < 1e-4


def test_rigid_body_invariance():
    p = build_problem()
    theta = torch.ones(p["conn"].shape[0], dtype=dtype)
    p0 = dict(p)
    p0["f"] = torch.zeros_like(p["f"])
    n = p["nodes"].shape[0]
    t = torch.zeros(2 * n, dtype=dtype)
    t[0::2], t[1::2] = 0.3, -0.2
    assert global_residual(t, theta, p0).abs().max() < 1e-10   # translation
    a = 0.7
    Q = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=dtype)
    X = p["nodes"]
    ur = (X @ Q.T - X).reshape(-1)
    assert global_residual(ur, theta, p0).abs().max() < 1e-9   # rotation


if __name__ == "__main__":
    test_operator_invariants()
    test_rigid_body_invariance()
    print("ALL INVARIANT TESTS PASSED")
