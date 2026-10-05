"""
kinematics.py
=============

Geometry and deformation of ONE element, all built from the shape functions.

The isoparametric idea
----------------------
The same shape functions interpolate the geometry and the displacement:

        X(xi, eta) = sum_a N_a(xi, eta) X_a          (reference position)
        u(xi, eta) = sum_a N_a(xi, eta) u_a          (displacement)

Chain rule:  the deformation gradient needs  du/dX, but we can only
differentiate with respect to the PARENT coordinates (xi, eta):

        du/dX = (du/dxi) (dX/dxi)^{-1}               F = I + du/dX

Everything below is a pure function of its arguments; closures
(``X_fn``, ``u_fn``) carry the nodal data of one element.
"""

import torch
from torch.func import jacfwd
from typing import Callable, Tuple


# =============================================================================
# Interpolation closures
# =============================================================================
def Interpolation_Fn(shape_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                     nodal_values: torch.Tensor  # (n_nodes, 2)
                     ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Build the closure  (xi, eta) -> v(xi, eta) = sum_a N_a(xi, eta) v_a.

    Parameters
    ----------
    shape_fn     : (xi, eta) -> (n_nodes,)
    nodal_values : (n_nodes, 2)  the values v_a at the nodes (coordinates or displacements)

    Returns
    -------
    closure : (xi, eta) -> (2,)   (or (n_pts, 2) if xi, eta are batched)
    """
    # "...i,ij->...j" contracts the node index i:
    #   (n_nodes,) x (n_nodes, 2) -> (2,)        for one point
    #   (n_pts, n_nodes) x (n_nodes, 2) -> (n_pts, 2)   for a batch of points
    return lambda xi, eta: torch.einsum("...i,ij->...j", shape_fn(xi, eta), nodal_values)


def Coordinate_Interpolation(shape_fn, nodal_values):
    """
    Geometry map X(xi, eta): parent coordinates -> reference (undeformed) position.

    Parameters
    ----------
    shape_fn     : shape functions of the element
    nodal_values : (n_nodes, 2) undeformed node coordinates X_a of the element
    """
    return Interpolation_Fn(shape_fn, nodal_values)


def Displacement_Interpolation(shape_fn, nodal_values):
    """
    Displacement field u(xi, eta).

    Parameters
    ----------
    nodal_values : (n_nodes, 2) nodal displacements u_a of the element
    """
    return Interpolation_Fn(shape_fn, nodal_values)


# =============================================================================
# Mesh Jacobian
# =============================================================================
def Mesh_Jacobian(X_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                  xi: torch.Tensor,
                  eta: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Jacobian of the map parent -> reference geometry at one point:

            J = dX/d(xi, eta) = [ dX/dxi | dX/deta ]      (2 x 2)

    det(J) is the local area scale factor: dOmega = det(J) dxi deta.
    det(J) <= 0 means the element is inverted or badly distorted.

    Returns
    -------
    Grad_X     : (2, 2)
    det_Grad_X : ()
    inv_Grad_X : (2, 2)
    """
    dX_dxi = jacfwd(X_fn, argnums=0)(xi, eta)    # (2,)  derivative of X w.r.t. xi
    dX_deta = jacfwd(X_fn, argnums=1)(xi, eta)   # (2,)  derivative of X w.r.t. eta

    # columns are the two tangent vectors of the parent axes: (2, 2)
    Grad_X = torch.stack([dX_dxi, dX_deta], dim=-1)

    det_Grad_X = torch.linalg.det(Grad_X)
    inv_Grad_X = torch.linalg.inv(Grad_X)
    return Grad_X, det_Grad_X, inv_Grad_X


# =============================================================================
# Deformation gradients
# =============================================================================
def Local_Deformation_Gradient(u_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                               xi: torch.Tensor,
                               eta: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Displacement gradient with respect to the PARENT coordinates:

            Grad_xi(u) = [ du/dxi | du/deta ]      F_local = I + Grad_xi(u)

    ("local" = not yet mapped to the physical element.)

    Returns
    -------
    Local_F      : (2, 2)
    Local_Grad_u : (2, 2)
    """
    du_dxi = jacfwd(u_fn, argnums=0)(xi, eta)     # (2,)
    du_deta = jacfwd(u_fn, argnums=1)(xi, eta)    # (2,)

    Local_Grad_u = torch.stack([du_dxi, du_deta], dim=-1)             # (2, 2)
    I = torch.eye(2, dtype=Local_Grad_u.dtype, device=Local_Grad_u.device)
    return I + Local_Grad_u, Local_Grad_u


def Global_Deformation_Gradient(u_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                X_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                xi: torch.Tensor,
                                eta: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PHYSICAL deformation gradient at one Gauss point (chain rule):

            Grad_X(u) = Grad_xi(u) @ (dX/dxi)^{-1}          F = I + Grad_X(u)

    Returns
    -------
    Global_F      : (2, 2)  deformation gradient F
    det_Grad_X    : ()      det of the mesh Jacobian (volume scale factor)
    Global_Grad_u : (2, 2)  displacement gradient du/dX
    """
    # du/d(xi, eta)
    _, Local_Grad_u = Local_Deformation_Gradient(u_fn, xi, eta)
    # dX/d(xi, eta), its determinant and inverse
    _, det_Grad_X, inv_Grad_X = Mesh_Jacobian(X_fn, xi, eta)

    # (du/dxi) (dX/dxi)^{-1}  -> du/dX      indices: ij,jk->ik
    Global_Grad_u = torch.einsum("ij,jk->ik", Local_Grad_u, inv_Grad_X)

    I = torch.eye(2, dtype=Global_Grad_u.dtype, device=Global_Grad_u.device)
    Global_F = I + Global_Grad_u
    return Global_F, det_Grad_X, Global_Grad_u
