import torch


# =============================================================================
# 1D Boundary Edge Shape Functions (Reference interval: s in [-1, 1])
# =============================================================================



def shape_fn_1d_linear(s: torch.Tensor) -> torch.Tensor:
    """
    1D 2-node linear Lagrange shape functions on [-1, 1].
    Node 1 (-1) ------- Node 2 (+1)

    Standard linear 1D shape functions on reference interval [-1, 1] for element edges.

    Node 1 (s = -1) ------- Node 2 (s = 1)

    Parameters
    ----------
    s : torch.Tensor
        Scalar coordinate along the 1D edge.

    Returns
    -------
    torch.Tensor
        Shape function values [N1, N2] of shape (2,).
    """

    N1 = 0.5 * (1.0 - s)
    N2 = 0.5 * (1.0 + s)

    return torch.stack([N1, N2])


def shape_fn_1d_quadratic(s: torch.Tensor) -> torch.Tensor:
    """
    1D 3-node quadratic Lagrange shape functions on [-1, 1].
    Node 1 (-1) ---- Node 3 (0) ---- Node 2 (+1)
    """
    N1 = 0.5 * s * (s - 1.0)
    N2 = 0.5 * s * (s + 1.0)
    N3 = 1.0 - s**2

    return torch.stack([N1, N2, N3])



# =============================================================================
# 2D Quadrilateral Elements (Reference domain: [-1, 1] x [-1, 1])
# =============================================================================



def shape_fn_quad4(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:

    """
    Standard bilinear Lagrange shape functions for a 4-node quadrilateral (Quad4)
    on the parent reference domain [-1, 1] x [-1, 1].

    Node numbering:
      4 (-1, 1) ------- 3 (1, 1)
          |                |
          |     (xi,eta)   |
          |                |
      1 (-1,-1) ------- 2 (1,-1)

    Parameters
    ----------
    xi : torch.Tensor
        Scalar coordinate in parent xi-direction.
    eta : torch.Tensor
        Scalar coordinate in parent eta-direction.

    Returns
    -------
    torch.Tensor
        Shape function values [N1, N2, N3, N4] of shape (4,).
    """
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)

    return torch.stack([N1, N2, N3, N4])


def shape_fn_quad8(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
     """
    8-node quadratic serendipity quadrilateral element (Quad8).

    4 (-1, 1) ---- 7 (0, 1) ---- 3 (1, 1)
        |                          |
    8 (-1, 0)      (xi,eta)      6 (1, 0)
        |                          |
    1 (-1,-1) ---- 5 (0,-1) ---- 2 (1,-1)
    """
     
     # Corner Nodes
     N1 = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0)
     N2 = 0.25 * (1.0 + xi) * (1.0 - eta) * ( xi - eta - 1.0)
     N3 = 0.25 * (1.0 + xi) * (1.0 + eta) * ( xi + eta - 1.0)
     N4 = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)

     # Mid-edges Nodes
     N5 = 0.5 * (1.0 - xi**2) * (1.0 - eta)
     N6 = 0.5 * (1.0 + xi) * (1.0 - eta**2)
     N7 = 0.5 * (1.0 - xi**2) * (1.0 + eta)
     N8 = 0.5 * (1.0 - xi) * (1.0 - eta**2)

     return torch.stack([N1, N2, N3, N4, N5, N6, N7, N8])


def shape_fn_quad9(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:

    """
    9-node biquadratic Lagrangian quadrilateral element (Quad9).

    4 (-1, 1) ---- 7 (0, 1) ---- 3 (1, 1)
        |            |             |
    8 (-1, 0) ---- 9 (0, 0) ---- 6 (1, 0)
        |            |             |
    1 (-1,-1) ---- 5 (0,-1) ---- 2 (1,-1)
    """

    L1_xi = 0.5 * xi * (xi - 1.0)
    L2_xi = 1.0 - xi**2
    L3_xi = 0.5 * xi * (xi + 1.0)

    L1_eta = 0.5 * eta * (eta - 1.0)
    L2_eta = 1.0 - eta**2
    L3_eta = 0.5 * eta * (eta + 1.0)

    N1 = L1_xi * L1_eta
    N2 = L3_xi * L1_eta
    N3 = L3_xi * L3_eta
    N4 = L1_xi * L3_eta
    N5 = L2_xi * L1_eta
    N6 = L3_xi * L2_eta
    N7 = L2_xi * L3_eta
    N8 = L1_xi * L2_eta
    N9 = L2_xi * L2_eta

    return torch.stack([N1, N2, N3, N4, N5, N6, N7, N8, N9])


# =============================================================================
# 2D Triangular Elements (Reference domain: xi >= 0, eta >= 0, xi + eta <= 1)
# =============================================================================

def shape_fn_tri3(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    3-node linear triangular element (Tri3).

    3 (0, 1)
    | \
    |   \
    1 (0, 0) -- 2 (1, 0)
    """
    N1 = 1.0 - xi - eta
    N2 = xi
    N3 = eta
    return torch.stack([N1, N2, N3])

def shape_fn_tri6(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    6-node quadratic triangular element (Tri6).

    3 (0, 1)
    | \
    6   5
    |     \
    1 -- 4 -- 2 (1, 0)
    """
    lam1 = 1.0 - xi - eta
    lam2 = xi
    lam3 = eta

    # Corner Nodes
    N1 = lam1 * (2.0 * lam1 - 1.0)
    N2 = lam2 * (2.0 * lam2 - 1.0)
    N3 = lam3 * (2.0 * lam3 - 1.0)

    # Mid-side Nodes
    N4 = 4.0 * lam1 * lam2
    N5 = 4.0 * lam2 * lam3
    N6 = 4.0 * lam3 * lam1

    return torch.stack([N1, N2, N3, N4, N5, N6])












