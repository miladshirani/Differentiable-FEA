"""
Elements.py
===========

The ELEMENT LIBRARY: one dictionary that says everything the rest of the code
needs to know about an element type.

Why a table instead of classes?  Polymorphism in this code base is done by
*passing functions and data*: ``params["shape_fn"]`` is simply the shape
function, and the quadrature rule is simply two tensors.  Adding a new element
type means adding one row to ``ELEMENT_LIBRARY`` (and one shape function) --
nothing else changes.

Fields of each entry
--------------------
shape_fn   : (xi, eta) -> (n_nodes,)  shape functions of the element
n_nodes    : number of nodes per element
family     : "quad" or "tri"
order      : polynomial order of the displacement field (1 or 2)
n_corners  : number of corner (vertex) nodes; they come FIRST in the numbering
edges      : local node ids of every edge, ordered [start, end] for linear
             elements and [start, end, mid] for quadratic ones, CCW around the
             element.  Used to find the boundary and to integrate tractions.
gauss      : default quadrature recipe, ("tensor", n) or ("triangle", degree)
gmsh_type  : integer element-type code of the Gmsh mesher
"""

import torch
from typing import Callable, Dict, List, Tuple

from scr.shape_functions_2D import (shape_fn_quad4, shape_fn_quad8, shape_fn_quad9,
                                    shape_fn_tri3, shape_fn_tri6,
                                    shape_fn_1d_linear, shape_fn_1d_quadratic)
from scr.Gauss_Quadratures import (Gauss_Quadratures_1D, Gauss_Quadratures_2D,
                                   Gauss_Quadratures_Triangle)


ELEMENT_LIBRARY: Dict[str, dict] = {
    "quad4": dict(shape_fn=shape_fn_quad4, n_nodes=4, family="quad", order=1, n_corners=4,
                  edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
                  gauss=("tensor", 2), gmsh_type=3),

    "quad8": dict(shape_fn=shape_fn_quad8, n_nodes=8, family="quad", order=2, n_corners=4,
                  edges=[(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
                  gauss=("tensor", 3), gmsh_type=16),

    "quad9": dict(shape_fn=shape_fn_quad9, n_nodes=9, family="quad", order=2, n_corners=4,
                  edges=[(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
                  gauss=("tensor", 3), gmsh_type=10),

    "tri3": dict(shape_fn=shape_fn_tri3, n_nodes=3, family="tri", order=1, n_corners=3,
                 edges=[(0, 1), (1, 2), (2, 0)],
                 gauss=("triangle", 1), gmsh_type=2),

    "tri6": dict(shape_fn=shape_fn_tri6, n_nodes=6, family="tri", order=2, n_corners=3,
                 edges=[(0, 1, 3), (1, 2, 4), (2, 0, 5)],
                 gauss=("triangle", 4), gmsh_type=9),
}


def element_names() -> List[str]:
    """Names of all supported element types, e.g. ['quad4', 'quad8', ...]."""
    return list(ELEMENT_LIBRARY.keys())


def get_element(name: str) -> dict:
    """Look up an element type; raises a helpful error for a typo."""
    if name not in ELEMENT_LIBRARY:
        raise ValueError(f"Unknown element type '{name}'. Choose one of {element_names()}.")
    return ELEMENT_LIBRARY[name]


def element_quadrature(name: str,
                       dtype: torch.dtype = torch.float64,
                       device: str = "cpu"
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Default Gauss rule (points, weights) of an element type.

    The rule is chosen so that the neo-Hookean integrand is integrated
    accurately on undistorted elements:
        Q4  : 2x2 points       Q8/Q9 : 3x3 points
        Tri3: 1 point (F is constant)   Tri6 : 6 points (degree 4)

    Returns
    -------
    GP : (n_q, 2)   GW : (n_q,)
    """
    kind, n = get_element(name)["gauss"]
    if kind == "tensor":
        return Gauss_Quadratures_2D(n, dtype=dtype, device=device)
    return Gauss_Quadratures_Triangle(n, dtype=dtype, device=device)


def edge_shape_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    1D shape function of the EDGES of an element type: linear (2 nodes) for
    first-order elements, quadratic (3 nodes) for second-order ones.
    """
    return shape_fn_1d_linear if get_element(name)["order"] == 1 else shape_fn_1d_quadratic


def reference_node_coordinates(name: str,
                               dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Coordinates (xi, eta) of the nodes on the reference element, shape (n_nodes, 2).
    Handy for tests ("N_a(node_b) = delta_ab") and for plotting.
    """
    coords = {
        "quad4": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
        "quad8": [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0)],
        "quad9": [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)],
        "tri3": [(0, 0), (1, 0), (0, 1)],
        "tri6": [(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)],
    }[name]
    return torch.tensor(coords, dtype=dtype)
