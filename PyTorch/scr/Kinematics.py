import torch
from torch.func import grad, vmap, jacfwd
from typing import Callable, Tuple



def Interpolation_Fn(shape_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                        nodal_values: torch.Tensor # shape (n_nodes, 2)
                        ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    
    """
    Creates an isoparametric field interpolation closure:
        (xi, eta) -> v(xi, eta) = shape_fn(xi, eta) @ nodal_values
    """
    # Dot/Matrix product: (n_nodes,) @ (n_nodes, 2) -> (2,)
    # Batch product:      (n_pts, n_nodes) @ (n_nodes, 2) -> (n_pts, 2)
    return  lambda xi, eta: shape_fn(xi, eta) @ nodal_values


def Coordinate_Interpolation(shape_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                            nodal_values: torch.Tensor # shape (n_nodes, 2)
                            ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Creates a mapping function X(xi, eta) from reference domain to physical domain.

    Parameters
    ----------
    shape_fn : Callable
        Evaluates shape functions N(xi, eta) -> shape (n_nodes,).
    X_nodes : torch.Tensor
        Undeformed nodal coordinates of element, shape (n_nodes, 2).

    Returns
    -------
    X_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function returning physical position X(xi, eta) of shape (2,).
    """
    
    return Interpolation_Fn(shape_fn, nodal_values)


def Displacement_Interpolation(shape_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                               nodal_values: torch.Tensor # shape (n_nodes, 2)
                                ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    
    """
    Creates a mapping function u(xi, eta) for the interpolated displacement field.

    Parameters
    ----------
    shape_fn : Callable
        Evaluates shape functions N(xi, eta) -> shape (n_nodes,).
    u_nodes : torch.Tensor
        Nodal displacements of element, shape (n_nodes, 2).

    Returns
    -------
    u_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function returning displacement vector u(xi, eta) of shape (2,).
    """
    
    return Interpolation_Fn(shape_fn, nodal_values)

    
    

def Mesh_Jacobian(
    X_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xi: torch.Tensor,
    eta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reference-to-physical mesh Jacobian dX/d(xi, eta),
    its determinant det(J), and its inverse J^{-1}.

    Returns
    -------
    Grad_X : torch.Tensor
        Mesh Jacobian matrix of shape (2, 2).
    det_Grad_X : torch.Tensor
        Determinant of mesh Jacobian of shape ().
    inv_Grad_X : torch.Tensor
        Inverse of mesh Jacobian matrix of shape (2, 2).
        """
    
    dX_dxi = jacfwd(X_fn, argnums = 0)(xi, eta)       # Shape: (2,)
    dX_eta = jacfwd(X_fn, argnums = 1)(xi, eta)       # Shape: (2,)

    Grad_X = torch.stack([dX_dxi, dX_eta], dim = -1) # Shape: (2,2)

    det_Grad_X = torch.linalg.det(Grad_X)
    inv_Grad_X = torch.linalg.inv(Grad_X)

    return Grad_X, det_Grad_X, inv_Grad_X


def Local_Deformation_Gradient(
    u_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xi: torch.Tensor,
    eta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the displacement gradient with respect to reference parent coordinates:
        grad_xi(u) = du / d(xi, eta)
    and the local deformation gradient F_local = I + grad_xi(u).

    Returns
    -------
    Local_F : torch.Tensor, shape (2, 2)
    Local_Grad_u : torch.Tensor, shape (2, 2)
    """
    
    
    du_dxi = jacfwd(u_fn, argnums = 0)(xi, eta)             # Shape: (2,)
    du_eta = jacfwd(u_fn, argnums = 1)(xi, eta)             # Shape: (2,)

    Local_Grad_u = torch.stack([du_dxi, du_eta], dim = -1)  # Shape: (2,2)
    I = torch.eye(2, dtype = Local_Grad_u.dtype, device=Local_Grad_u.device)
    Local_F = I + Local_Grad_u

    return Local_F, Local_Grad_u



def Global_Deformation_Gradient(
    u_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xi: torch.Tensor,
    eta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes physical deformation gradient F, mesh determinant det(J),
    and physical displacement gradient du/dX using the chain rule:
        grad_X(u) = grad_xi(u) @ (dX/dxi)^{-1}
        F = I + grad_X(u)

    Returns
    -------
    Global_F : torch.Tensor, shape (2, 2)
    det_Grad_X : torch.Tensor, shape ()
    Global_Grad_u : torch.Tensor, shape (2, 2)
    """

    _, Local_Grad_u           = Local_Deformation_Gradient(u_fn, xi, eta)
    _, det_Grad_X, inv_Grad_X = Mesh_Jacobian(X_fn, xi, eta)

    Global_Grad_u = Local_Grad_u @ inv_Grad_X

    I = torch.eye(2, dtype = Global_Grad_u.dtype, device=Global_Grad_u.device)

    Global_F = I + Global_Grad_u

    return Global_F, det_Grad_X, Global_Grad_u

    





