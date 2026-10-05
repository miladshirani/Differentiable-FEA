"""
gauss_quadratures.py
====================

Numerical integration rules on the *reference* elements.

Finite element integrals are never computed on the physical element.  We pull
every integral back to a fixed reference element (the square [-1,1]^2 for
quadrilaterals, the unit right triangle for triangles) and replace it by a
weighted sum over a few well-chosen points:

        integral_ref  f(xi, eta) d(xi) d(eta)   ~=   sum_q  w_q * f(xi_q, eta_q)

Everything here returns plain tensors ``(points, weights)``:

    points  : (n_q, 2)   columns are [xi, eta]
    weights : (n_q,)

so that the rest of the code can just ``vmap`` over the first axis.
"""

import numpy as np
import torch
from typing import Tuple


# =============================================================================
# 1D Gauss-Legendre rule on [-1, 1]
# =============================================================================
def Gauss_Quadratures_1D(n_points_per_dim: int,
                         dtype: torch.dtype = torch.float64,
                         device: str = "cpu"
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gauss-Legendre points and weights on the interval [-1, 1].

    An n-point rule integrates every polynomial of degree <= 2n - 1 EXACTLY.
    (That is why 2 points are enough for a bilinear Q4 element on an
    undistorted mesh: the integrand is at most cubic along each direction.)

    Parameters
    ----------
    n_points_per_dim : number of Gauss points n
    dtype, device    : torch dtype / device of the returned tensors

    Returns
    -------
    points  : (n,)  abscissae, roots of the Legendre polynomial P_n
    weights : (n,)  weights, they sum to 2 (the length of [-1, 1])
    """
    # numpy computes the roots of P_n with the Golub-Welsch algorithm
    # (eigenvalues of a small symmetric tridiagonal matrix).
    pts_1d, wts_1d = np.polynomial.legendre.leggauss(n_points_per_dim)

    # Move the numbers into torch with the requested precision / device.
    points = torch.tensor(pts_1d, dtype=dtype, device=device)    # (n,)
    weights = torch.tensor(wts_1d, dtype=dtype, device=device)   # (n,)
    return points, weights


# =============================================================================
# 2D tensor-product rule on the square [-1, 1]^2   (for Q4, Q8, Q9)
# =============================================================================
def Gauss_Quadratures_2D(n_points_per_dim: int,
                         dtype: torch.dtype = torch.float64,
                         device: str = "cpu"
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tensor-product Gauss rule on [-1, 1]^2: every pairing of two 1D points.

    Parameters
    ----------
    n_points_per_dim : n, giving n*n points in total

    Returns
    -------
    points  : (n*n, 2)  columns [xi, eta]
    weights : (n*n,)    products w_i * w_j, they sum to 4 (the area of the square)
    """
    # 1D rule that is going to be "multiplied with itself"
    pts_1d, wts_1d = Gauss_Quadratures_1D(n_points_per_dim, dtype=dtype, device=device)

    # Build all (xi, eta) pairs with two nested vmaps (no Python loops):
    #   inner vmap sweeps xi   (x) for a fixed eta (y)   -> (n, 2)
    #   outer vmap sweeps eta  (y)                        -> (n, n, 2)
    # and reshape(-1, 2) flattens the two sweep axes into one list of n*n points.
    points = torch.func.vmap(
        lambda y: torch.func.vmap(lambda x: torch.stack([x, y]))(pts_1d)
    )(pts_1d).reshape(-1, 2)

    # The weight of a pair is the product of the two 1D weights (outer product),
    # flattened in the same (eta-major, xi-minor) order as the points above.
    weights = torch.einsum("i,j->ij", wts_1d, wts_1d).reshape(-1)
    return points, weights


# =============================================================================
# Rules on the reference triangle  {xi >= 0, eta >= 0, xi + eta <= 1}   (Tri3, Tri6)
# =============================================================================
def Gauss_Quadratures_Triangle(degree: int,
                               dtype: torch.dtype = torch.float64,
                               device: str = "cpu"
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quadrature on the reference triangle that is EXACT for all polynomials of
    total degree <= ``degree``.

    A triangle is not a product of two intervals, so the tensor-product rule
    above does not apply.  We use classical *symmetric* rules for the degrees
    that finite elements need in practice, and a *collapsed* (Duffy) rule built
    from 1D Gauss points for everything else.

        degree 1 : 1 point  (centroid)           exact for Tri3 with constant F
        degree 2 : 3 points                      mass matrices of Tri3
        degree 4 : 6 points (Dunavant)           neo-Hookean Tri6 (F is linear in x)
        other    : collapsed Gauss, n*n points   n = ceil((degree + 2) / 2)

    Returns
    -------
    points  : (n_q, 2)  columns [xi, eta]
    weights : (n_q,)    they sum to 1/2 (the area of the reference triangle)
    """
    if degree <= 1:
        # centroid rule: f(1/3, 1/3) * area
        pts = [[1.0 / 3.0, 1.0 / 3.0]]
        wts = [0.5]

    elif degree == 2:
        # three "inner mid-edge" points; each carries one third of the area
        pts = [[1.0 / 6.0, 1.0 / 6.0],
               [2.0 / 3.0, 1.0 / 6.0],
               [1.0 / 6.0, 2.0 / 3.0]]
        wts = [1.0 / 6.0] * 3

    elif degree <= 4:
        # Dunavant degree-4 rule: two orbits of three points each.  The
        # barycentric points are (a, a, 1-2a) and its permutations.
        a1, w1 = 0.445948490915965, 0.223381589678011
        a2, w2 = 0.091576213509771, 0.109951743655322
        b1, b2 = 1.0 - 2.0 * a1, 1.0 - 2.0 * a2
        pts = [[a1, a1], [b1, a1], [a1, b1],
               [a2, a2], [b2, a2], [a2, b2]]
        # the published weights add to 1 (unit area): scale by 1/2
        wts = [0.5 * w1] * 3 + [0.5 * w2] * 3

    else:
        # Collapsed (Duffy) rule.  Map the unit square onto the triangle:
        #       xi = u,   eta = v * (1 - u),     d(xi) d(eta) = (1 - u) du dv
        # then apply a tensor Gauss rule in (u, v).  The Jacobian factor (1-u)
        # raises the polynomial degree in u by one, hence n >= (degree + 2) / 2.
        n = int(np.ceil((degree + 2) / 2))
        g, w = np.polynomial.legendre.leggauss(n)
        g, w = 0.5 * (g + 1.0), 0.5 * w            # map [-1, 1] -> [0, 1]
        pts, wts = [], []
        for ui, wu in zip(g, w):
            for vj, wv in zip(g, w):
                pts.append([ui, vj * (1.0 - ui)])
                wts.append(wu * wv * (1.0 - ui))

    points = torch.tensor(pts, dtype=dtype, device=device)    # (n_q, 2)
    weights = torch.tensor(wts, dtype=dtype, device=device)   # (n_q,)
    return points, weights
