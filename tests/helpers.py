"""Shared helpers for the test-suite: tiny problems that solve in a fraction of a second."""
import torch
from difffea.problem import build_problem, default_spec

DTYPE = torch.float64


def small_problem(element_type="quad4", load=(60.0, 0.0), h=0.5, geometry="rectangle",
                  mesher="structured", **overrides):
    """A small cantilever-like problem for ``element_type`` (structured mesh, no Gmsh needed)."""
    spec = default_spec(geometry, element_type, h)
    spec.update(mesher=mesher, load_value=load, E=2600.0, nu=0.3)
    spec.update(overrides)
    params = build_problem(spec)
    theta = torch.ones(params["conn"].shape[0], dtype=DTYPE)
    return params, theta


def dense(op, n):
    """Assemble the dense matrix of a linear operator by applying it to unit vectors (tests only!)."""
    eye = torch.eye(n, dtype=DTYPE)
    return torch.stack([op(eye[i]) for i in range(n)], dim=1)


def cauchy_at_node(u, params, theta, node):
    """
    Cauchy stress (sxx, syy, sxy) AT a mesh node, evaluated at the node's reference coordinates inside
    every element that contains it and then averaged over those elements.  Unlike ``compute_fields`` (one
    averaged value per element), this keeps the peak values that stress-concentration checks need.

    u : (2N,) displacement, params : problem dictionary, theta : (n_el,), node : int.  Returns (3,).
    """
    from difffea.elements import reference_node_coordinates
    from difffea.kinematics import Coordinate_Interpolation, Displacement_Interpolation
    from difffea.materials import get_material, material_of, element_material_parameters
    from difffea.postprocess import gauss_point_state

    conn, un = params["conn"], u.reshape(-1, 2)
    psi3 = get_material(material_of(params)[0])["psi3"]
    element_p = element_material_parameters(theta, params)                  # (n_el, n_p)
    ref = reference_node_coordinates(params["element_type"])                # (nen, 2) local node coords
    states = []
    for e, a in (conn == node).nonzero().tolist():                          # (element, local node) pairs
        X_fn = Coordinate_Interpolation(params["shape_fn"], params["nodes"][conn[e]])
        u_fn = Displacement_Interpolation(params["shape_fn"], un[conn[e]])
        s = gauss_point_state(u_fn, X_fn, element_p[e], psi3, ref[a, 0], ref[a, 1])
        states.append(s[2:5])                                               # sigma_xx, sigma_yy, sigma_xy
    return torch.stack(states).mean(dim=0)
