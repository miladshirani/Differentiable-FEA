"""
newton_krylov.py
================

The NONLINEAR SOLVER, assembled from three nested loops:

    LEVEL 1  load continuation   ramp the load factor 0 -> 1; the increment is
                                 HALVED automatically whenever a step fails
    LEVEL 2  Newton iteration    linearise:  J(u) du = -R(u)
    LEVEL 3  line search         shrink the step until the residual decreases

and, inside every Newton iteration, a matrix-free Krylov solve of the linear
system with the Preconditioned Conjugate Gradient method (PCG).  The Jacobian
J is NEVER assembled: PCG only needs the product  v -> J v  (``make_A_operator``).

"Inexact" Newton: the linear system is solved only to a loose, adaptive
tolerance (the Eisenstat-Walker forcing term) -- accurate solves far from the
solution are wasted work.
"""

import torch
from difffea.operators import global_residual, make_tangent_operator, make_preconditioner


# =============================================================================
# 1. MATRIX-FREE PRECONDITIONED CONJUGATE GRADIENT (PCG)
# =============================================================================
def pcg(A_fn, b, Minv_fn, rtol=1e-5, max_iter=500):
    """
    Solve the symmetric positive-definite system  A x = b  using only the
    action  v -> A(v)  of the matrix.

    Algorithm (textbook PCG, Saad Alg. 9.1):
        r0 = b,  z0 = M^-1 r0,  p0 = z0
        repeat:  alpha = (r.z)/(p.Ap);  x += alpha p;  r -= alpha Ap
                 z = M^-1 r;  beta = (r.z)_new/(r.z)_old;  p = z + beta p

    Parameters
    ----------
    A_fn     : v -> A(v)      matrix-free operator (e.g. ``make_A_operator``)
    b        : (n,)           right-hand side
    Minv_fn  : r -> M^-1 r    preconditioner (e.g. Jacobi)
    rtol     : stop when ||r|| <= rtol * ||b||   (the "forcing term" eta)
    max_iter : iteration cap

    Returns
    -------
    x       : (n,) approximate solution
    n_iters : number of Krylov iterations used
    """
    x = torch.zeros_like(b)        # initial guess x0 = 0, so the first residual is b
    r = b.clone()                  # residual r = b - A x
    z = Minv_fn(r)                 # preconditioned residual z = M^-1 r
    p = z.clone()                  # first search direction
    rz = torch.dot(r, z)           # inner product r.z, reused by alpha and beta
    norm_b = torch.norm(b)

    # Nothing to do if the right-hand side is zero (e.g. already in equilibrium)
    if norm_b == 0.0:
        return x, 0

    for k in range(max_iter):
        # 1. the ONLY use of the operator: one matrix-vector product
        Ap = A_fn(p)
        curv = torch.dot(p, Ap)    # curvature p^T A p  (> 0 for SPD A)

        # 2. non-positive curvature => A is not SPD along p (buckling / inverted
        #    element).  Stop with the best information we have.
        if curv <= 0.0:
            return (z if k == 0 else x), k

        # 3. exact line minimisation along p
        alpha = rz / curv
        x = x + alpha * p
        r = r - alpha * Ap

        # 4. converged?  (relative residual test)
        if torch.norm(r) <= rtol * norm_b:
            return x, k + 1

        # 5. new preconditioned residual and A-conjugate direction update
        z = Minv_fn(r)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, max_iter


# =============================================================================
# 2. LEVEL 3: ARMIJO BACKTRACKING LINE SEARCH
# =============================================================================
def armijo_line_search(u, du, rn, theta, params, load_factor, c1=1e-4, min_alpha=2.0 ** -5):
    """
    Find a step length alpha in (0, 1] with sufficient decrease of ||R||:

            ||R(u + alpha du)||  <  (1 - c1 alpha) ||R(u)||

    Start with the full Newton step alpha = 1 and halve it until the condition
    holds.  The test also rejects steps that invert an element: there
    det F <= 0, log J is NaN and every comparison with NaN is False.

    Returns
    -------
    u_new, R_new : accepted state and its residual   (u, None if rejected)
    rn_new       : ||R_new||
    alpha        : step length used (the last tried one if rejected)
    accepted     : False if alpha fell below ``min_alpha`` (the step is hopeless:
                   Newton is outside its convergence basin, cut the load step)
    """
    alpha = 1.0
    while alpha >= min_alpha:
        u_candidate = u + alpha * du
        R_candidate = global_residual(u_candidate, theta, params, load_factor)
        rn_candidate = torch.norm(R_candidate)
        if rn_candidate < (1.0 - c1 * alpha) * rn:
            return u_candidate, R_candidate, rn_candidate, alpha, True
        alpha *= 0.5                       # halve the step and try again
    return u, None, rn, alpha, False


# =============================================================================
# 3. LEVEL 2: NEWTON ITERATION AT ONE FIXED LOAD FACTOR
# =============================================================================
def newton_at_load(u0, theta, params, load_factor, tol, max_newton, verbose, callback, step_id,
                   preconditioner="jacobi", tangent="qp"):
    """
    Drive the residual to zero at one fixed load factor with inexact Newton-Krylov.

    Returns
    -------
    u         : final iterate
    converged : bool
    records   : list of per-iteration dicts (step, load_factor, iteration,
                residual, cg_iters, alpha)
    """
    # impose the prescribed displacements of this load level (functional "u[fixed] = g")
    u = torch.where(params["fixed"], load_factor * params["u_D"], u0)

    R = global_residual(u, theta, params, load_factor)
    r0 = torch.norm(R)       # residual at the start (reference size for the forcing term)
    rn = r0                  # current residual norm
    records = []

    for it in range(max_newton + 1):
        if rn < tol:
            if verbose:
                print(f"  converged in {it} Newton iterations (||R|| = {rn:.3e})")
            return u, True, records
        if it == max_newton:
            break

        # Eisenstat-Walker forcing term: loose (0.5) far away, tight near the
        # solution, because Newton's error only shrinks as fast as the linear
        # solve is accurate.
        eta = min(0.5, float((rn / (r0 + 1e-14)) ** 0.5))

        # linearisation at the current state: closures, no matrices
        A_fn = make_tangent_operator(tangent, u, theta, params)
        Minv_fn = make_preconditioner(preconditioner, u, theta, params)

        # solve  J du = -R  matrix-free, to relative accuracy eta
        du, cg_iters = pcg(A_fn, -R, Minv_fn, rtol=eta)

        # globalise with the line search
        u, R_new, rn, alpha, accepted = armijo_line_search(u, du, rn, theta, params, load_factor)
        record = dict(step=step_id, load_factor=load_factor, iteration=it,
                      residual=float(rn), cg_iters=int(cg_iters), alpha=alpha)
        records.append(record)
        if callback is not None:
            callback(record)
        if verbose:
            print(f"  iter {it:2d} | ||R|| = {rn:.4e} | CG = {cg_iters:3d} | alpha = {alpha:.4f}")
        if not accepted:
            return u, False, records          # stalled: let the caller reduce the load step
        R = R_new

    return u, False, records                  # max_newton reached without convergence


# =============================================================================
# 4. LEVEL 1: ADAPTIVE LOAD CONTINUATION
# =============================================================================
@torch.no_grad()
def newton_krylov_solve(params, theta, n_load_steps=5, tol=1e-8, max_newton=25,
                        verbose=True, callback=None, max_cutbacks=8, preconditioner="jacobi",
                        tangent="qp"):
    """
    Solve the nonlinear equilibrium equations  R(u, theta) = 0  and report how
    the iteration went.

    ``@torch.no_grad()`` switches autograd recording OFF for the whole solve:
    we do not want a graph of every Newton iteration.  (Sensitivities are
    obtained later by the adjoint method, see adjoint.py.)

    Load continuation: the load factor goes 0 -> 1 in increments of 1/n_load_steps.
    If Newton fails at some increment (stalled line search or too many
    iterations), the state is restored and the increment is HALVED, up to
    ``max_cutbacks`` times; after a success the increment may grow back.

    Parameters
    ----------
    params       : problem dictionary (see operators.py)
    theta        : (n_el,) element stiffness scaling (1.0 = nominal material)
    n_load_steps : nominal number of load increments
    tol          : absolute convergence tolerance on ||R||
    max_newton   : maximum Newton iterations per load increment
    verbose      : print a table of iterations
    callback     : optional function(dict) called after every Newton iteration
    max_cutbacks : how many times the increment may be halved
    preconditioner : "jacobi", "block_jacobi" or "ilu" (see Operators.PRECONDITIONERS)
    tangent      : how the matrix-free tangent product is evaluated: "qp" (default: tangent stored at
                   the Gauss points, fastest measured), "linearize" or "jvp" (reference); see
                   Operators.TANGENT_OPERATORS.  All three apply the same linear operator.

    Returns
    -------
    u    : (2N,) displacement at the last converged load factor
    info : dict with  converged (bool), load_factor (reached), history (list of
           per-iteration dicts), total_newton, total_cg, n_cutbacks, snapshots
           (list of (load factor, u) at every converged load level, for animations)
    """
    num_dofs = 2 * params["nodes"].shape[0]
    u = torch.zeros(num_dofs, dtype=params["nodes"].dtype, device=params["nodes"].device)
    info = {"converged": False, "load_factor": 0.0, "history": [],
            "total_newton": 0, "total_cg": 0, "n_cutbacks": 0,
            "snapshots": [(0.0, u.clone())]}       # (load factor, u) of every converged level

    d_nominal = 1.0 / n_load_steps       # nominal load increment
    d_lf = d_nominal                     # current load increment
    lf = 0.0                             # last converged load factor
    step_id, cutbacks_in_row = 0, 0

    while lf < 1.0 - 1e-12:
        lf_try = min(1.0, lf + d_lf)
        step_id += 1
        if verbose:
            print(f"\n=== Load step {step_id}: load factor {lf:.3f} -> {lf_try:.3f} ===")

        u_try, ok, records = newton_at_load(u, theta, params, lf_try, tol, max_newton,
                                            verbose, callback, step_id, preconditioner, tangent)
        info["history"] += records
        info["total_newton"] += len(records)
        info["total_cg"] += sum(r["cg_iters"] for r in records)

        if ok:
            u, lf = u_try, lf_try
            info["snapshots"].append((lf, u.clone()))
            cutbacks_in_row = 0
            d_lf = min(d_nominal, 1.5 * d_lf)          # recover the nominal increment
        else:
            cutbacks_in_row += 1
            info["n_cutbacks"] += 1
            if cutbacks_in_row > max_cutbacks:
                info["load_factor"] = lf
                return u, info                          # give up: report the last converged state
            d_lf *= 0.5                                 # retry the SAME state with a smaller step
            if verbose:
                print(f"  Newton failed -> load increment reduced to {d_lf:.4f}")

    info["converged"] = True
    info["load_factor"] = 1.0
    return u, info


def solve_newton_krylov(params, theta, n_load_steps=5, tol=1e-8, max_newton=25, verbose=True):
    """
    Convenience wrapper around ``newton_krylov_solve`` that returns ONLY the
    displacement vector u (2N,).  Used by the examples, tests and sensitivities.
    """
    u, _ = newton_krylov_solve(params, theta, n_load_steps=n_load_steps, tol=tol,
                               max_newton=max_newton, verbose=verbose)
    return u
