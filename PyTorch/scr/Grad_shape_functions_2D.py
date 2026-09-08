import torch
from torch.func import grad, vmap, jacfwd
from typing import Callable


def Grad_Shape_Functions_1D(shape_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            s: torch.Tensor
                            ) -> torch.Tensor:
    
    """
    Computes the 1D reference gradient along an element boundary edge:
        dN/ds = d(shape_fn_1d)/ds   (shape: (n_edge_nodes,))

    Parameters
    ----------
    shape_fn_1d : Callable
        1D shape function returning N(s) of shape (n_edge_nodes,).
    s : torch.Tensor
        Reference coordinate s in [-1, 1] (0D scalar tensor).

    Returns
    -------
    torch.Tensor
        Gradient tensor dN_ds of shape (n_edge_nodes,).
    """
    return jacfwd(shape_fn)(s)




def Grad_Shape_Functions_2D(shape_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            xi: torch.Tensor,
                            eta: torch.Tensor
                            ) -> torch.Tensor:

    """
    Computes the reference-coordinate gradients of 2D shape functions:
        dN/dxi  = d(shape_fn)/dxi   (shape: (n_nodes,))
        dN/deta = d(shape_fn)/deta  (shape: (n_nodes,))

    Parameters
    ----------
    shape_fn : Callable
        Function returning shape functions N(xi, eta) of shape (n_nodes,).
    xi : torch.Tensor
        Reference coordinate xi (0D scalar tensor).
    eta : torch.Tensor
        Reference coordinate eta (0D scalar tensor).

    Returns
    -------
    torch.Tensor
        Gradient matrix dN_dxi_eta of shape (n_nodes, 2), where:
        - Column 0: dN / dxi
        - Column 1: dN / deta
    """

    dN_dxi  = jacfwd(shape_fn, argnums = 0)(xi, eta)
    dN_deta = jacfwd(shape_fn, argnums = 1)(xi, eta)

    # Combine into a single (n_nodes, 2) gradient matrix
    return torch.stack([dN_dxi, dN_deta], dim = -1)



# Vectorized Gradient Functions across Gauss Points


"""
Vectorized 1D edge shape function gradients over 1D Gauss points.
Inputs:
    shape_fn_1d : Callable
    s_pts       : torch.Tensor of shape (n_gp_1d,)
Returns:
    dN_ds : torch.Tensor of shape (n_gp_1d, n_edge_nodes)
"""
Grad_Shape_Functions_1D_Batched = vmap(Grad_Shape_Functions_1D, in_dims=(None, 0))


"""
Vectorized 2D shape function gradients over Gauss points.
Inputs:
    shape_fn : Callable
    xi_pts   : torch.Tensor of shape (n_gp,)
    eta_pts  : torch.Tensor of shape (n_gp,)
Returns:
    dN_dxi_eta : torch.Tensor of shape (n_gp, n_nodes, 2)
"""
Grad_Shape_Functions_2D_Batched = vmap(Grad_Shape_Functions_2D, in_dims=(None, 0, 0))
