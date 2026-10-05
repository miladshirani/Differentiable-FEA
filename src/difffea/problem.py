"""
problem.py
==========

PROBLEM SET-UP: turns a plain-dictionary SPECIFICATION of a boundary value
problem (geometry, mesh, element type, material, supports, load) into the
``params`` dictionary that every solver function consumes.

    spec  --build_problem-->  params   --newton_krylov_solve-->  u

``spec`` keys
-------------
geometry      : key of Gmsh_Mesh.GEOMETRIES ("rectangle", "l_bracket", ...)
geom_params   : dimensions of the geometry, e.g. {"width": 3, "height": 2, "radius": 0.4}
element_type  : "quad4" | "quad8" | "quad9" | "tri3" | "tri6"
h             : target element size (smaller = denser mesh)
mesher        : "gmsh" (any geometry) or "structured" (rectangle only, no Gmsh needed)
material      : key of Materials.MATERIAL_LIBRARY (default "neo_hookean")
material_params : dict of parameter values overriding the library defaults, e.g.
                {"k1": 2000.0, "beta_deg": 45.0}   (see materials.py for each model)
E, nu         : (neo-Hookean only, when ``material_params`` is absent) Young's modulus
                and Poisson's ratio, converted to the Lame parameters
fixed_side    : "left" | "right" | "top" | "bottom"   (clamped side)
fixed_dofs    : "xy" (clamp) | "x" | "y"  (roller: only that component is held)
load_side     : side where the load acts
load_mode     : "force" (total force, consistent nodal loads) or
                "displacement" (prescribed displacement)
load_value    : (a, b)  force components or displacement components
                (for "displacement" a component may be None = free)
dtype         : torch dtype (float64 recommended)
device        : "cpu" (default) or "cuda"/"mps": where the solver tensors live
"""

import torch
from typing import Tuple

from difffea.elements import get_element, element_quadrature, edge_shape_fn
from difffea.mesh import create_rectangle_mesh
from difffea.gmsh_mesh import GEOMETRIES, GMSH_AVAILABLE, generate_gmsh_mesh
from difffea.materials import material_vector
from difffea.operators import precompute_geometry
from difffea.boundary_conditions import (select_boundary_nodes, dofs_from_node_mask,
                                     boundary_edges, edge_traction_load)


def lame_parameters(E: float, nu: float) -> Tuple[float, float]:
    """
    Lame parameters (mu, lambda) from Young's modulus and Poisson's ratio
    (3D / plane-strain relations):

            mu = E / (2 (1 + nu)),      lambda = E nu / ((1 + nu)(1 - 2 nu))
    """
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lmbda


# Sensible supports and loads for each geometry (used as GUI defaults).  The
# numbers are chosen so that the response is clearly nonlinear (strains of a few
# per cent up to ~20 %) but still converges in a handful of Newton iterations.
GEOMETRY_PRESETS = {
    "rectangle":     dict(fixed_side="left", fixed_dofs="xy", load_side="right",
                          load_mode="force", load_value=(0.0, -40.0)),       # cantilever bending
    "l_bracket":     dict(fixed_side="top", fixed_dofs="xy", load_side="right",
                          load_mode="force", load_value=(0.0, -2.0)),        # shear at the tip
    "plate_hole":    dict(fixed_side="left", fixed_dofs="xy", load_side="right",
                          load_mode="force", load_value=(400.0, 0.0)),       # uniaxial tension
    "notched_plate": dict(fixed_side="left", fixed_dofs="xy", load_side="right",
                          load_mode="force", load_value=(300.0, 0.0)),       # uniaxial tension
}


def default_spec(geometry: str = "plate_hole", element_type: str = "quad4", h: float = 0.2) -> dict:
    """
    A ready-to-run specification for a geometry: its default dimensions,
    supports and load (``GEOMETRY_PRESETS``), and a representative material.
    """
    return dict(geometry=geometry,
                geom_params=dict(GEOMETRIES[geometry]["params"]),
                element_type=element_type, h=h, mesher="gmsh",
                E=2600.0, nu=0.3,
                dtype=torch.float64,
                **GEOMETRY_PRESETS[geometry])


def to_device(params: dict, device) -> dict:
    """
    Move every tensor of a ``params`` dictionary to ``device`` (functional: returns a
    NEW dictionary).  Meshing, boundary conditions and the geometry cache are built on
    the CPU; this is the single place where the problem is transferred to a GPU.
    """
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in params.items()}


def build_mesh(spec: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mesh the geometry of ``spec`` -> (nodes (N, 2), conn (n_el, nen)).

    The "structured" mesher only knows rectangles; every geometry works with Gmsh.
    """
    dtype = spec.get("dtype", torch.float64)
    geometry, gp = spec["geometry"], spec["geom_params"]
    if spec.get("mesher", "gmsh") == "structured":
        if geometry != "rectangle":
            raise ValueError("The structured mesher only supports the 'rectangle' geometry.")
        # number of cells so that cell size ~ h in each direction (at least 1)
        nx = max(1, round(gp["width"] / spec["h"]))
        ny = max(1, round(gp["height"] / spec["h"]))
        return create_rectangle_mesh(spec["element_type"], gp["width"], gp["height"], nx, ny, dtype=dtype)
    if not GMSH_AVAILABLE:
        raise RuntimeError("Gmsh is not installed: use mesher='structured' or `pip install gmsh`.")
    return generate_gmsh_mesh(geometry, gp, spec["element_type"], spec["h"], dtype=dtype)


def build_problem(spec: dict) -> dict:
    """
    Assemble the ``params`` dictionary (see operators.py) from a specification.

    Steps
    -----
    1. mesh the geometry
    2. element-type data: shape functions and Gauss rule
    3. Dirichlet DOFs: clamped side (+ prescribed-displacement side, if any)
    4. external forces: consistent edge tractions on the loaded side
    5. material parameters from (E, nu)

    Returns
    -------
    params : dict with the solver keys  nodes, conn, shape_fn, GP, GW, material, mat_p, mat_scalable,
             f, m, fixed, u_D   and extra keys for post-processing  (element_type, ...)
    """
    dtype = spec.get("dtype", torch.float64)
    element_type = spec["element_type"]
    el = get_element(element_type)

    # --- 1. mesh --------------------------------------------------------------
    nodes, conn = build_mesh(spec)
    N = nodes.shape[0]

    # --- 2. element data ------------------------------------------------------
    GP, GW = element_quadrature(element_type, dtype=dtype)

    # --- 3. Dirichlet conditions ---------------------------------------------
    fixed_nodes = select_boundary_nodes(nodes, spec["fixed_side"])
    comps = {"xy": (0, 1), "x": (0,), "y": (1,)}[spec.get("fixed_dofs", "xy")]
    fixed = dofs_from_node_mask(fixed_nodes, comps)                       # (2N,) bool
    u_D = torch.zeros(2 * N, dtype=dtype)                                 # prescribed values (0 here)

    load_nodes = select_boundary_nodes(nodes, spec["load_side"])
    load_mode = spec.get("load_mode", "force")
    value = spec.get("load_value", (0.0, 0.0))

    f = torch.zeros(N, 2, dtype=dtype)                                    # external forces

    if load_mode == "displacement":
        # impose the displacement components on the load side (None = left free)
        for c, val in enumerate(value):
            if val is None:
                continue
            dof_mask = dofs_from_node_mask(load_nodes, (c,))
            fixed = fixed | dof_mask
            u_D = torch.where(dof_mask, torch.as_tensor(float(val), dtype=dtype), u_D)
    else:
        # force: integrate the traction along the boundary edges of the loaded side
        edges = boundary_edges(conn, el["edges"])
        f = edge_traction_load(nodes, edges, edge_shape_fn(element_type), load_nodes,
                               (float(value[0]), float(value[1])))

    # Dirichlet mask: 1.0 on free DOFs, 0.0 on constrained DOFs (see operators.py)
    m = torch.where(fixed, 0.0, 1.0).to(dtype)

    # --- 5. material ----------------------------------------------------------
    material = spec.get("material", "neo_hookean")
    values = spec.get("material_params")
    if values is None and material == "neo_hookean" and "E" in spec:
        mu0, lmbda0 = lame_parameters(spec["E"], spec["nu"])
        values = {"mu": mu0, "lmbda": lmbda0}
    mat_p, mat_scalable = material_vector(material, values, dtype=dtype)

    params = {
        "nodes": nodes, "conn": conn,
        "shape_fn": el["shape_fn"], "GP": GP, "GW": GW,
        "material": material, "mat_p": mat_p, "mat_scalable": mat_scalable,
        "f": f.reshape(-1), "m": m, "fixed": fixed, "u_D": u_D,
        # extras (not used by the solver, used by post-processing and the GUI)
        "element_type": element_type,
    }

    # --- 6. hoist the loop-invariant geometry out of the Newton loop ----------
    params = precompute_geometry(params)

    # --- 7. optionally move everything to the GPU ----------------------------
    device = spec.get("device", "cpu")
    return params if str(device) == "cpu" else to_device(params, device)
