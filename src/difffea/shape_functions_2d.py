"""
shape_functions_2d.py
=====================

Lagrange / serendipity SHAPE FUNCTIONS on the reference elements.

A shape function N_a(xi, eta) is the "hat" attached to node a of the element.
It equals 1 at its own node, 0 at every other node, and the N_a always sum to 1
(partition of unity).  A field is interpolated from nodal values v_a as

        v(xi, eta) = sum_a  N_a(xi, eta) * v_a

Every function here takes the two SCALAR reference coordinates (xi, eta)
and returns a vector ``(n_nodes,)`` holding N_1 ... N_n at that point.
They are written for ONE point on purpose: derivatives come from autograd
(see grad_shape_functions_2d.py) and loops over points come from ``vmap``.

Reference domains
-----------------
    quadrilaterals : the square  [-1, 1] x [-1, 1]
    triangles      : the unit right triangle  xi >= 0, eta >= 0, xi + eta <= 1
    edges (1D)     : the interval [-1, 1]
"""

import torch


# =============================================================================
# 1D edge shape functions (reference interval s in [-1, 1])
# Used to integrate tractions (surface loads) along element edges.
# =============================================================================

def shape_fn_1d_linear(s: torch.Tensor) -> torch.Tensor:
    """
    2-node linear Lagrange shape functions on an edge.

        node 1 (s = -1) ------------- node 2 (s = +1)

    Parameters
    ----------
    s : scalar tensor, coordinate along the edge

    Returns
    -------
    (2,)  [N1, N2]
    """
    N1 = 0.5 * (1.0 - s)    # 1 at s=-1, 0 at s=+1
    N2 = 0.5 * (1.0 + s)    # 0 at s=-1, 1 at s=+1
    return torch.stack([N1, N2])


def shape_fn_1d_quadratic(s: torch.Tensor) -> torch.Tensor:
    """
    3-node quadratic Lagrange shape functions on an edge.

        node 1 (s = -1) ---- node 3 (s = 0) ---- node 2 (s = +1)

    NOTE the ordering: the two END nodes come first, the MID node last.  This
    matches how the element edges are listed in elements.py
    (start vertex, end vertex, mid-side node).

    Returns
    -------
    (3,)  [N1, N2, N3]
    """
    N1 = 0.5 * s * (s - 1.0)   # 1 at s=-1
    N2 = 0.5 * s * (s + 1.0)   # 1 at s=+1
    N3 = 1.0 - s ** 2          # 1 at s=0  (the bubble)
    return torch.stack([N1, N2, N3])


# =============================================================================
# 2D quadrilateral elements (reference square [-1, 1]^2)
# Nodes are numbered counter-clockwise (CCW): corners first, then mid-sides.
# =============================================================================

def shape_fn_quad4(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    4-node bilinear quadrilateral (Q4).

        4 (-1, 1) ------- 3 (1, 1)
            |                |
            |   (xi, eta)    |
            |                |
        1 (-1,-1) ------- 2 (1,-1)

    Each N_a is the product of two 1D linear "hats", one in xi and one in eta.

    Returns
    -------
    (4,)  [N1, N2, N3, N4]
    """
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return torch.stack([N1, N2, N3, N4])


def shape_fn_quad8(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    8-node quadratic SERENDIPITY quadrilateral (Q8): no centre node.

        4 (-1, 1) ---- 7 (0, 1) ---- 3 (1, 1)
            |                           |
        8 (-1, 0)     (xi, eta)      6 (1, 0)
            |                           |
        1 (-1,-1) ---- 5 (0,-1) ---- 2 (1,-1)

    Corner functions are bilinear hats corrected by -(xi^2...) terms so that they
    vanish at the mid-side nodes; mid-side functions are (parabola) x (linear hat).

    Returns
    -------
    (8,)  [N1 ... N8]
    """
    # --- corner nodes --------------------------------------------------------
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)

    # --- mid-side nodes ------------------------------------------------------
    N5 = 0.5 * (1.0 - xi ** 2) * (1.0 - eta)    # bottom
    N6 = 0.5 * (1.0 + xi) * (1.0 - eta ** 2)    # right
    N7 = 0.5 * (1.0 - xi ** 2) * (1.0 + eta)    # top
    N8 = 0.5 * (1.0 - xi) * (1.0 - eta ** 2)    # left

    return torch.stack([N1, N2, N3, N4, N5, N6, N7, N8])


def shape_fn_quad9(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    9-node biquadratic LAGRANGE quadrilateral (Q9).

        4 (-1, 1) ---- 7 (0, 1) ---- 3 (1, 1)
            |             |             |
        8 (-1, 0) ---- 9 (0, 0) ---- 6 (1, 0)
            |             |             |
        1 (-1,-1) ---- 5 (0,-1) ---- 2 (1,-1)

    Built as a tensor product of the three 1D quadratic Lagrange polynomials
    (L1 at -1, L2 at 0, L3 at +1) in each direction.

    Returns
    -------
    (9,)  [N1 ... N9]
    """
    # 1D quadratic Lagrange polynomials in xi ...
    L1_xi = 0.5 * xi * (xi - 1.0)     # 1 at xi = -1
    L2_xi = 1.0 - xi ** 2             # 1 at xi =  0
    L3_xi = 0.5 * xi * (xi + 1.0)     # 1 at xi = +1
    # ... and in eta
    L1_eta = 0.5 * eta * (eta - 1.0)
    L2_eta = 1.0 - eta ** 2
    L3_eta = 0.5 * eta * (eta + 1.0)

    # The node at (xi_i, eta_j) gets the product L_i(xi) * L_j(eta)
    N1 = L1_xi * L1_eta   # corner (-1,-1)
    N2 = L3_xi * L1_eta   # corner (+1,-1)
    N3 = L3_xi * L3_eta   # corner (+1,+1)
    N4 = L1_xi * L3_eta   # corner (-1,+1)
    N5 = L2_xi * L1_eta   # bottom mid
    N6 = L3_xi * L2_eta   # right mid
    N7 = L2_xi * L3_eta   # top mid
    N8 = L1_xi * L2_eta   # left mid
    N9 = L2_xi * L2_eta   # centre

    return torch.stack([N1, N2, N3, N4, N5, N6, N7, N8, N9])


# =============================================================================
# 2D triangular elements (reference triangle xi, eta >= 0, xi + eta <= 1)
# Written with the barycentric (area) coordinates  lam1 + lam2 + lam3 = 1.
# =============================================================================

def shape_fn_tri3(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    3-node linear triangle (Tri3, "constant strain triangle").

        3 (0, 1)
        |  \\
        |    \\
        1 (0, 0) ---- 2 (1, 0)

    The shape functions ARE the barycentric coordinates.

    Returns
    -------
    (3,)  [N1, N2, N3]
    """
    N1 = 1.0 - xi - eta
    N2 = xi
    N3 = eta
    return torch.stack([N1, N2, N3])


def shape_fn_tri6(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    6-node quadratic triangle (Tri6).

        3 (0, 1)
        |  \\
        6    5          mid-side nodes: 4 on edge 1-2,
        |      \\                       5 on edge 2-3,
        1 --4-- 2 (1,0)                 6 on edge 3-1

    Returns
    -------
    (6,)  [N1 ... N6]
    """
    # barycentric coordinates
    lam1 = 1.0 - xi - eta
    lam2 = xi
    lam3 = eta

    # corner nodes:  lam (2 lam - 1)   -> 1 at own corner, 0 at all other nodes
    N1 = lam1 * (2.0 * lam1 - 1.0)
    N2 = lam2 * (2.0 * lam2 - 1.0)
    N3 = lam3 * (2.0 * lam3 - 1.0)

    # mid-side nodes: 4 lam_i lam_j     -> 1 at the mid-point of edge i-j
    N4 = 4.0 * lam1 * lam2
    N5 = 4.0 * lam2 * lam3
    N6 = 4.0 * lam3 * lam1

    return torch.stack([N1, N2, N3, N4, N5, N6])
