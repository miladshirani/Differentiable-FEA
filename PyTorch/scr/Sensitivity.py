import torch
from torch.func import grad
from scr.Newton_Krylov import solve_newton_krylov
from scr.Adjoint import solve_adjoint, compute_implicit_vjp


def compute_sensitivity(params, theta, loss_fn, n_load_steps=5, tol=1e-8,
                        max_newton=25, adjoint_rtol=1e-10):
    """
    Purely functional replacement for the autograd bridge `DifferentiableFEASolver`
    (a torch.autograd.Function subclass) in Adjoint.py.

    Solves the forward problem R(u*, theta) = 0, then evaluates the exact design
    gradient of a scalar objective L(u*) through the adjoint / implicit function theorem:

        1. u*        = solve_newton_krylov(params, theta)        (no autograd graph)
        2. dL_du     = grad(loss_fn)(u*)                          (explicit partial derivative)
        3. A^T lam   = m * dL_du                                  (matrix-free adjoint solve)
        4. dL_dtheta = -lam^T dR/dtheta                           (one vjp, no matrix)

    Parameters
    ----------
    params  : dict   Problem Pytree (mesh, connectivity, BC masks, material, ...)
    theta   : (num_elements,) design / stiffness-scaling field
    loss_fn : Callable u -> scalar, e.g. lambda u: torch.dot(params["f"], u)

    Returns
    -------
    u_star     : (2 * num_nodes,)
    dL_dtheta  : (num_elements,)
    """
    theta = theta.detach()
    u_star = solve_newton_krylov(params, theta, n_load_steps=n_load_steps, tol=tol,
                                 max_newton=max_newton, verbose=False)
    dL_du = grad(loss_fn)(u_star)
    lam, _ = solve_adjoint(u_star, theta, params, dL_du, rtol=adjoint_rtol)
    dL_dtheta = compute_implicit_vjp(u_star, theta, params, lam)
    return u_star, dL_dtheta
