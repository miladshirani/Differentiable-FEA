"""
Adjoint.py
==========

Exact DESIGN SENSITIVITIES of a converged nonlinear solution, without
differentiating through the Newton iterations.

Setting: u*(theta) solves R(u*, theta) = 0 and we care about a scalar L(u*).
Differentiating the equilibrium condition (implicit function theorem):

        dR/du  du*/dtheta + dR/dtheta = 0
    =>  dL/dtheta = - lambda^T dR/dtheta ,    with the ADJOINT solve   J^T lambda = dL/du

Cost: ONE extra linear solve (with J^T) and ONE vector-Jacobian product, no
matter how many design variables theta there are, and O(1) memory in the
number of Newton iterations.

The class-based ``torch.autograd.Function`` bridge that used to live here was
removed: the project is purely functional.  ``Sensitivity.compute_sensitivity``
is its function-only replacement.
"""

import torch
from torch.func import vjp
from scr.Operators import global_residual, make_AT_operator, make_jacobi_preconditioner
from scr.Newton_Krylov import pcg

# =============================================================================
# 1. MATRIX-FREE ADJOINT LINEAR SOLVER
# =============================================================================
def solve_adjoint(u_star, theta, params, dL_du, rtol=1e-6, max_iter=500):
    """
    Solves the adjoint linear equation:
        A^T * lambda = m * (dL / du)
    matrix-free using Preconditioned Conjugate Gradient (or BiCGStab).

    Parameters
    ----------
    u_star   : Tensor of shape (2 * num_nodes,)
        Converged equilibrium displacement vector R(u_star, theta) = 0
    theta    : Tensor of shape (num_elements,)
        Element design / stiffness parameters
    params   : dict
        Mesh, connectivity, and boundary condition dictionary
    dL_du    : Tensor of shape (2 * num_nodes,)
        Direct partial derivative of the scalar loss w.r.t displacements
    rtol     : float
        Relative Krylov convergence tolerance

    Returns
    -------
    lambda_adj : Tensor of shape (2 * num_nodes,)
        Adjoint multiplier vector lambda
    cg_iters   : int
        Number of Krylov iterations consumed
    """
    m = params["m"]

    # 1. Build matrix-free transpose tangent operator and Jacobi preconditioner
    AT_fn = make_AT_operator(u_star, theta, params)
    Minv_fn = make_jacobi_preconditioner(u_star, theta, params)

    # 2. Right-hand side of adjoint system: project out Dirichlet boundaries
    b_adj = m * dL_du

    # 3. Solve A^T * lambda = b_adj matrix-free
    lambda_adj, cg_iters = pcg(AT_fn, b_adj, Minv_fn, rtol=rtol, max_iter=max_iter)

    return lambda_adj, cg_iters


# =============================================================================
# 2. PARAMETER SENSITIVITY VIA IMPLICIT FUNCTION THEOREM (VJP)
# =============================================================================
def compute_implicit_vjp(u_star, theta, params, lambda_adj):
    """
    Computes total design sensitivity dL/d(theta) using the adjoint solution:
        dL / d(theta) = -lambda^T * (dR / d(theta))

    Parameters
    ----------
    u_star     : Tensor (2 * num_nodes,)
        Converged equilibrium displacement state
    theta      : Tensor (num_elements,)
        Design parameter vector
    params     : dict
        Problem dictionary
    lambda_adj : Tensor (2 * num_nodes,)
        Solved adjoint multiplier vector

    Returns
    -------
    dL_dtheta : Tensor of shape (num_elements,)
        Exact gradient of the objective with respect to theta.
    """
    # Define residual strictly as a function of theta with u_star frozen
    def res_theta(th):
        return global_residual(u_star, th, params, load_factor=1.0)

    # Reverse-mode AD evaluates the Vector-Jacobian Product (VJP)
    _, vjp_fn = vjp(res_theta, theta)

    # Contraction: -lambda^T * (dR / d(theta))
    (dL_dtheta,) = vjp_fn(-lambda_adj)

    return dL_dtheta
