"""
Gmsh_Mesh.py
============

Unstructured meshes of arbitrary 2D geometries with the Gmsh mesher (run in a child process), for ANY
element type of the element library (Tri3, Tri6, Q4, Q8, Q9).

Every geometry is described by a small pure function (in Gmsh_Worker.py) that adds
points, lines and arcs to Gmsh and returns the closed boundary loops.  To add a new shape
(a bracket, a plate with two holes, ...) write one such function and register
it in ``GEOMETRIES`` -- the meshing, boundary conditions, solver and GUI work
without any other change.

Node numbering convention of Gmsh == convention of this code base
-----------------------------------------------------------------
Gmsh lists the corners first, counter-clockwise, then the mid-side nodes of
edges (1-2), (2-3), ..., then (for Q9) the centre node.  This is exactly the
order of ``shape_functions_2D.py`` so the connectivity can be used unchanged.
"""

import functools
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import torch
from typing import Dict, List, Tuple

from scr.Elements import get_element

# Gmsh is NOT imported here: it runs in a child process (see Gmsh_Worker.py) because its
# OpenMP runtime clashes with PyTorch's.  We only check that it is installed.
GMSH_AVAILABLE = importlib.util.find_spec("gmsh") is not None


# =============================================================================
# Small mesh-clean-up utilities
# =============================================================================
def filter_unused_nodes(nodes: torch.Tensor, elements: torch.Tensor):
    """
    Drop nodes that no element uses and renumber the connectivity 0..N'-1.

    (Gmsh also returns the points and curve nodes of the geometry; the solver
    must not see nodes without elements, they would give singular rows.)

    Parameters
    ----------
    nodes    : (N, 2)       elements : (n_el, nen) integer node ids
    Returns  : clean_nodes (N', 2), clean_elements (n_el, nen)
    """
    # `used` = sorted list of node ids that occur in the connectivity,
    # `inverse` = for every entry of `elements`, its position inside `used`
    used, inverse = torch.unique(elements, return_inverse=True)
    return nodes[used], inverse.reshape(elements.shape)


def signed_corner_area(nodes: torch.Tensor, elements: torch.Tensor, n_corners: int) -> torch.Tensor:
    """
    Signed area (shoelace formula) of the polygon formed by the CORNER nodes of
    every element.  Positive = counter-clockwise, negative = clockwise.

    Returns (n_el,)
    """
    xy = nodes[elements[:, :n_corners]]                    # (n_el, n_corners, 2)
    x, y = xy[..., 0], xy[..., 1]
    # shoelace: 2A = sum_i (x_i y_{i+1} - x_{i+1} y_i), indices taken cyclically
    x_next, y_next = torch.roll(x, -1, dims=1), torch.roll(y, -1, dims=1)
    return 0.5 * torch.sum(x * y_next - x_next * y, dim=1)


# permutation of the local nodes that REVERSES the orientation of an element
_FLIP_PERMUTATION = {
    "tri3": [0, 2, 1],
    "tri6": [0, 2, 1, 5, 4, 3],
    "quad4": [0, 3, 2, 1],
    "quad8": [0, 3, 2, 1, 7, 6, 5, 4],
    "quad9": [0, 3, 2, 1, 7, 6, 5, 4, 8],
}


def check_and_fix_ccw(nodes: torch.Tensor, elements: torch.Tensor,
                      element_type: str = "quad4") -> torch.Tensor:
    """
    Make every element counter-clockwise (positive Jacobian).

    A clockwise element has det(J) < 0 and gives a negative volume; we repair
    it by applying the orientation-reversing permutation of its local nodes.

    Returns (n_el, nen) connectivity.
    """
    area = signed_corner_area(nodes, elements, get_element(element_type)["n_corners"])
    flipped = elements[:, _FLIP_PERMUTATION[element_type]]       # all elements, reordered
    # keep the flipped version only where the area is negative
    return torch.where((area < 0.0)[:, None], flipped, elements)


# name -> description of every available geometry
GEOMETRIES: Dict[str, dict] = {
    "rectangle": dict(
        label="Rectangular plate / beam",
        params=dict(width=2.0, height=1.0),
        area=lambda p: p["width"] * p["height"],
        extent=lambda p: (p["width"], p["height"])),
    "l_bracket": dict(
        label="L-shaped bracket",
        params=dict(L=2.0, H=2.0, t=0.6),
        area=lambda p: p["L"] * p["t"] + p["t"] * (p["H"] - p["t"]),
        extent=lambda p: (p["L"], p["H"])),
    "plate_hole": dict(
        label="Plate with circular hole",
        params=dict(width=3.0, height=2.0, radius=0.4),
        area=lambda p: p["width"] * p["height"] - np.pi * p["radius"] ** 2,
        extent=lambda p: (p["width"], p["height"])),
    "notched_plate": dict(
        label="Plate with two semicircular notches",
        params=dict(width=3.0, height=2.0, radius=0.5),
        area=lambda p: p["width"] * p["height"] - np.pi * p["radius"] ** 2,
        extent=lambda p: (p["width"], p["height"])),
}


def geometry_names() -> List[str]:
    return list(GEOMETRIES.keys())


# =============================================================================
# The mesher
# =============================================================================
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gmsh_Worker.py")


@functools.lru_cache(maxsize=64)
def _mesh_in_subprocess(request_json: str):
    """
    Run ``Gmsh_Worker.py`` in a child process and return (xy, conn) as NumPy arrays.
    Results are cached by request, so a GUI that re-runs on every widget change does
    not re-mesh an unchanged geometry.
    """
    proc = subprocess.run([sys.executable, WORKER], input=request_json, capture_output=True,
                          text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError("Gmsh meshing failed:\n" + proc.stderr.strip().splitlines()[-1]
                           if proc.stderr.strip() else "Gmsh meshing failed")
    path = proc.stdout.strip().splitlines()[-1]
    try:
        with np.load(path) as data:
            return data["xy"].copy(), data["conn"].copy()
    finally:
        os.remove(path)


def generate_gmsh_mesh(geometry: str,
                       geom_params: dict,
                       element_type: str = "quad4",
                       h: float = 0.2,
                       dtype=torch.float64,
                       device: str = "cpu"
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mesh a geometry with elements of the requested type and size.

    Parameters
    ----------
    geometry     : key of ``GEOMETRIES``
    geom_params  : dict of dimensions, e.g. {"width": 3, "height": 2, "radius": 0.4}
    element_type : "tri3", "tri6", "quad4", "quad8" or "quad9"
    h            : target element edge length (smaller = denser mesh)

    Returns
    -------
    nodes    : (N, 2) coordinates          elements : (n_el, nen) connectivity
    """
    if not GMSH_AVAILABLE:
        raise RuntimeError("The Gmsh Python API is not installed (pip install gmsh).")
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry '{geometry}'. Choose one of {geometry_names()}.")

    spec = get_element(element_type)
    request = dict(geometry=geometry, params={k: float(v) for k, v in geom_params.items()},
                   element_family=spec["family"], order=spec["order"], gmsh_type=spec["gmsh_type"],
                   n_nodes=spec["n_nodes"], incomplete=(element_type == "quad8"), h=float(h))
    xy, conn = _mesh_in_subprocess(json.dumps(request, sort_keys=True))

    nodes_t = torch.tensor(xy, dtype=dtype, device=device)
    elems_t = torch.tensor(conn, dtype=torch.int64, device=device)

    nodes_t, elems_t = filter_unused_nodes(nodes_t, elems_t)
    elems_t = check_and_fix_ccw(nodes_t, elems_t, element_type)
    return nodes_t, elems_t
