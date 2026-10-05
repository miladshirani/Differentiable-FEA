"""Meshes (structured and Gmsh), boundary detection and consistent edge loads."""
import pytest
import torch
from torch.func import vmap

from difffea.elements import element_names, get_element, element_quadrature, edge_shape_fn
from difffea.mesh import create_rectangle_mesh
from difffea.gmsh_mesh import GEOMETRIES, GMSH_AVAILABLE, generate_gmsh_mesh
from difffea.kinematics import Coordinate_Interpolation, Mesh_Jacobian
from difffea.boundary_conditions import (select_boundary_nodes, boundary_edges, edge_traction_load)

DTYPE = torch.float64


def mesh_area_and_min_detJ(nodes, conn, name):
    GP, GW = element_quadrature(name)
    shape_fn = get_element(name)["shape_fn"]

    def detJ_of_element(Xe):
        X_fn = Coordinate_Interpolation(shape_fn, Xe)
        return vmap(lambda a, b: Mesh_Jacobian(X_fn, a, b)[1])(GP[:, 0], GP[:, 1])

    d = vmap(detJ_of_element)(nodes[conn])                 # (n_el, n_q)
    return torch.sum(d * GW), d.min()


@pytest.mark.parametrize("name", element_names())
def test_structured_mesh_area_and_orientation(name):
    nodes, conn = create_rectangle_mesh(name, 3.0, 2.0, 4, 3)
    assert conn.shape[1] == get_element(name)["n_nodes"]
    area, min_det = mesh_area_and_min_detJ(nodes, conn, name)
    assert abs(area - 6.0) < 1e-12
    assert min_det > 0
    assert conn.max() + 1 == nodes.shape[0]                # no unused nodes


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh not installed")
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("name", element_names())
def test_gmsh_mesh_valid(geometry, name):
    p = GEOMETRIES[geometry]["params"]
    nodes, conn = generate_gmsh_mesh(geometry, p, name, 0.4)
    area, min_det = mesh_area_and_min_detJ(nodes, conn, name)
    exact = GEOMETRIES[geometry]["area"](p)
    # straight-edge elements approximate a circle from inside: allow 1 % (linear) / 0.01 % (quadratic)
    tol = 1e-2 if get_element(name)["order"] == 1 else 1e-4
    assert abs(area - exact) / exact < tol
    assert min_det > 0


def test_boundary_edges_of_structured_quad_mesh():
    nodes, conn = create_rectangle_mesh("quad4", 1.0, 1.0, 3, 2)
    edges = boundary_edges(conn, get_element("quad4")["edges"])
    assert edges.shape[0] == 2 * (3 + 2)                    # perimeter = 10 edges
    xy = nodes[edges]                                        # (E, 2, 2)
    on_boundary = ((xy[..., 0] < 1e-12) | (xy[..., 0] > 1 - 1e-12) |
                   (xy[..., 1] < 1e-12) | (xy[..., 1] > 1 - 1e-12))
    assert on_boundary.all()


@pytest.mark.parametrize("name", element_names())
def test_consistent_load_total_force_and_distribution(name):
    el = get_element(name)
    nodes, conn = create_rectangle_mesh(name, 2.0, 1.0, 3, 2)
    edges = boundary_edges(conn, el["edges"])
    mask = select_boundary_nodes(nodes, "right")
    f = edge_traction_load(nodes, edges, edge_shape_fn(name), mask, (5.0, -3.0))
    assert torch.allclose(f.sum(0), torch.tensor([5.0, -3.0], dtype=DTYPE), atol=1e-12)
    assert (f[~mask].abs().max() == 0)                       # nothing off the loaded side


def test_quadratic_edge_distributes_one_sixth_four_sixths_one_sixth():
    """A single Q8 edge: the corner nodes carry 1/6, the mid-side node 4/6 of the force."""
    nodes, conn = create_rectangle_mesh("quad8", 1.0, 1.0, 1, 1)
    edges = boundary_edges(conn, get_element("quad8")["edges"])
    mask = select_boundary_nodes(nodes, "right")
    f = edge_traction_load(nodes, edges, edge_shape_fn("quad8"), mask, (6.0, 0.0))
    right = torch.where(mask)[0]
    vals = sorted(f[right, 0].tolist())
    assert vals == pytest.approx([1.0, 1.0, 4.0])
