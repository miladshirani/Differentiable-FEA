"""
shape_functions.py - Definition of reference shape functions and their gradients.
Supports both AD-based derivation and exact analytical derivatives.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple




def shape_fn_tri3(xi: float, eta: float) -> jnp.ndarray:
    """
    Linear 3-node triangle shape functions on reference simplex.
    Domain: xi >= 0, eta >= 0, xi + eta <= 1.
    """
    N1 = 1.0 - xi - eta
    N2 = xi
    N3 = eta
    return jnp.array([N1, N2, N3])

def shape_fn_tri6(xi: float, eta: float) -> jnp.ndarray:
    zeta = 1.0 - xi - eta
    
    # Corners
    N1 = zeta * (2.0 * zeta - 1.0)
    N2 = xi * (2.0 * xi - 1.0)
    N3 = eta * (2.0 * eta - 1.0)
    
    # Mid-edges
    N4 = 4.0 * xi * zeta
    N5 = 4.0 * xi * eta
    N6 = 4.0 * eta * zeta
    
    return jnp.array([N1, N2, N3, N4, N5, N6])


####################################################
####################################################
####################################################


def shape_fn_quad4(xi: float, eta: float) -> jnp.ndarray:
    """
    Standard bilinear 4-node quad shape functions.
    Node ordering: 1:(-1,-1), 2:(1,-1), 3:(1,1), 4:(-1,1)
    """
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)  # Corrected
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)  # Corrected

    return jnp.array([N1, N2, N3, N4])


def shape_fn_quad9(xi: float, eta: float) -> jnp.ndarray:
    """
    9-node biquadratic Lagrangian quadrilateral (Quad9) shape functions.
    
    Nodes 1-4: Corners (-1,-1), (1,-1), (1,1), (-1,1)
    Nodes 5-8: Mid-edges (0,-1), (1,0), (0,1), (-1,0)
    Node 9:    Center (0,0)
    
    Returns:
        jnp.ndarray of shape (9,) containing [N1, ..., N9]
    """
    # 1D Lagrange polynomials in xi
    l1_xi = 0.5 * xi * (xi - 1.0)
    l2_xi = 1.0 - xi**2
    l3_xi = 0.5 * xi * (xi + 1.0)

    # 1D Lagrange polynomials in eta
    l1_eta = 0.5 * eta * (eta - 1.0)
    l2_eta = 1.0 - eta**2
    l3_eta = 0.5 * eta * (eta + 1.0)

    # Corner nodes
    N1 = l1_xi * l1_eta
    N2 = l3_xi * l1_eta
    N3 = l3_xi * l3_eta
    N4 = l1_xi * l3_eta

    # Mid-edge nodes
    N5 = l2_xi * l1_eta
    N6 = l3_xi * l2_eta
    N7 = l2_xi * l3_eta
    N8 = l1_xi * l2_eta

    # Center node
    N9 = l2_xi * l2_eta

    return jnp.array([N1, N2, N3, N4, N5, N6, N7, N8, N9])


def shape_fn_quad8(xi: float, eta: float) -> jnp.ndarray:
    """
    8-node serendipity quadrilateral (Quad8) shape functions.

    Nodes 1-4: Corners (-1,-1), (1,-1), (1,1), (-1,1)
    Nodes 5-8: Mid-edges (0,-1), (1,0), (0,1), (-1,0)
    """
    # Corner nodes
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta) * ( xi - eta - 1.0)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta) * ( xi + eta - 1.0)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)

    # Mid-edge nodes
    N5 = 0.5 * (1.0 - xi**2) * (1.0 - eta)
    N6 = 0.5 * (1.0 + xi) * (1.0 - eta**2)
    N7 = 0.5 * (1.0 - xi**2) * (1.0 + eta)
    N8 = 0.5 * (1.0 - xi) * (1.0 - eta**2)

    return jnp.array([N1, N2, N3, N4, N5, N6, N7, N8])
    
