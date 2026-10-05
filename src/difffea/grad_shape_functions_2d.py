"""
grad_shape_functions_2d.py
==========================

Derivatives of the shape functions with respect to the REFERENCE coordinates,
obtained by forward-mode automatic differentiation.

We never type  dN/dxi  by hand.  Because ``shape_fn`` is a pure function of
(xi, eta), ``jacfwd`` differentiates it exactly, for every element type, with
zero extra code.  This is the first appearance of the book's recurring idea:

        "derivatives are PRODUCED by autodiff, never typed."
"""

import torch
from torch.func import vmap, jacfwd
from typing import Callable


def Grad_Shape_Functions_1D(shape_fn: Callable[[torch.Tensor], torch.Tensor],
                            s: torch.Tensor
                            ) -> torch.Tensor:
    """
    Derivative of 1D edge shape functions:  dN/ds  at one point.

    Parameters
    ----------
    shape_fn : s -> (n_edge_nodes,)   e.g. ``shape_fn_1d_linear``
    s        : scalar tensor, reference coordinate in [-1, 1]

    Returns
    -------
    dN_ds : (n_edge_nodes,)
    """
    # jacfwd of a function (scalar -> vector) is the vector of derivatives
    return jacfwd(shape_fn)(s)


def Grad_Shape_Functions_2D(shape_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            xi: torch.Tensor,
                            eta: torch.Tensor
                            ) -> torch.Tensor:
    """
    Reference gradients of all 2D shape functions at ONE point (xi, eta).

    Parameters
    ----------
    shape_fn : (xi, eta) -> (n_nodes,)
    xi, eta  : scalar tensors

    Returns
    -------
    dN : (n_nodes, 2)   column 0 = dN/dxi,  column 1 = dN/deta
    """
    # derivative w.r.t. the 1st argument (xi)  -> (n_nodes,)
    dN_dxi = jacfwd(shape_fn, argnums=0)(xi, eta)
    # derivative w.r.t. the 2nd argument (eta) -> (n_nodes,)
    dN_deta = jacfwd(shape_fn, argnums=1)(xi, eta)

    # put the two derivative columns side by side: (n_nodes, 2)
    return torch.stack([dN_dxi, dN_deta], dim=-1)


# -----------------------------------------------------------------------------
# Batched versions: evaluate at many points at once with vmap.
#   in_dims = (None, 0)    -> the function is shared, the point is batched
#   in_dims = (None, 0, 0) -> the function is shared, xi and eta are batched
# -----------------------------------------------------------------------------

# s : (n_pts,)  ->  (n_pts, n_edge_nodes)
Grad_Shape_Functions_1D_Batched = vmap(Grad_Shape_Functions_1D, in_dims=(None, 0))

# xi, eta : (n_pts,)  ->  (n_pts, n_nodes, 2)
Grad_Shape_Functions_2D_Batched = vmap(Grad_Shape_Functions_2D, in_dims=(None, 0, 0))
