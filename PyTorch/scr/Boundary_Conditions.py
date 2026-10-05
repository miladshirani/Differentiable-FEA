import torch
from torch.func import vmap
from typing import Callable, Tuple

from scr.Gauss_Quadratures import Gauss_Quadratures_1D
from scr.Grad_shape_functions_2D import Grad_Shape_Functions_1D


def apply_boundary_conditions(
    nodes: torch.Tensor, 
    bc_list: list, 
    num_fields: int
):
    """
    TUTORIAL: Pure Functional Multiphysics Boundary Condition Compiler
    ==================================================================
    
    PURPOSE:
    --------
    Compiles human-readable boundary condition specifications into static,
    pre-allocated tensors. 
    - ZERO CLASSES!
    - 100% compatible with JIT, torch.compile, vmap, and JAX!
    - Supports arbitrary multiphysics:
        num_fields = 2 -> Solid mechanics [u_x, u_y]
        num_fields = 3 -> Phase-field fracture [u_x, u_y, damage_d]
        num_fields = 3 -> Thermoelasticity [u_x, u_y, Temp_T]
        num_fields = 4 -> Nematic elastomers [u_x, u_y, n_x, n_y]
        
    PARAMETERS:
    -----------
    nodes : torch.Tensor
        Undeformed nodal coordinates of shape (num_nodes, 2).
    bc_list : list of dicts
        List of boundary condition dictionaries defining subdomains, types, and values.
    num_fields : int
        Number of physical degree-of-freedom fields per node.
        
    RETURNS:
    --------
    is_fixed : torch.Tensor (bool)
        Shape (num_nodes, num_fields). True if a field is Dirichlet-constrained.
    prescribed_U : torch.Tensor (float)
        Shape (num_nodes, num_fields). Target values for the fixed fields.
    F_ext : torch.Tensor (float)
        Shape (num_nodes, num_fields). External Neumann forces/tractions applied to nodes.
    """
    num_nodes = nodes.shape[0]
    device = nodes.device
    dtype = nodes.dtype

    # Extract 1D spatial coordinate vectors
    # x: shape (num_nodes,), y: shape (num_nodes,)
    x = nodes[:, 0]
    y = nodes[:, 1]

    # -------------------------------------------------------------------------
    # STEP 1: Pre-allocate Static Output Tensors
    # -------------------------------------------------------------------------
    # By creating static tensors here, the downstream solver loop has NO dynamic
    # allocations, which is required for JIT compilation (torch.compile / jax.jit).
    is_fixed = torch.zeros((num_nodes, num_fields), dtype=torch.bool, device=device)
    prescribed_U = torch.zeros((num_nodes, num_fields), dtype=dtype, device=device)
    F_ext = torch.zeros((num_nodes, num_fields), dtype=dtype, device=device)

    # -------------------------------------------------------------------------
    # STEP 2: Process Each Boundary Condition
    # -------------------------------------------------------------------------
    for bc in bc_list:
        # A. Evaluate the spatial predicate (lambda x, y: condition)
        # mask is a 1D boolean tensor of length num_nodes: True for nodes on boundary.
        mask = bc["subdomain"](x, y)
        
        # Find integer node IDs where mask == True
        node_ids = torch.where(mask)[0]

        # If no nodes match this condition, safely skip
        if len(node_ids) == 0:
            continue

        # ---------------------------------------------------------------------
        # Case 1: Dirichlet Boundary Condition (Prescribed values)
        # ---------------------------------------------------------------------
        if bc["type"] == "dirichlet": 
            val = bc.get("value", 0.0)

            # If value is a function (e.g. lambda x, y: 0.1 * y for linear shear),
            # evaluate it at the specific boundary node coordinates:
            if callable(val):
                val = val(x[node_ids], y[node_ids])

            comp = bc.get("component", None)

            if comp is not None:
                # Constrain ONLY this specific field component (e.g. only u_x, or only Temp)
                is_fixed[node_ids, comp] = True
                prescribed_U[node_ids, comp] = val
            else:
                # Constrain ALL field components at these nodes
                is_fixed[node_ids, :] = True
                prescribed_U[node_ids, :] = val

        # ---------------------------------------------------------------------
        # Case 2: Neumann Boundary Condition (Surface tractions, loads, fluxes)
        # ---------------------------------------------------------------------
        elif bc["type"] == "neumann":
            # Convert user traction to a tensor of shape (num_fields,)
            trac = torch.as_tensor(bc["traction"], dtype=dtype, device=device)
            
            # Distribute total load evenly among all boundary nodes on that edge:
            # (In FEA, total edge force = sum of nodal equivalent forces)
            F_ext[node_ids, :] += trac / len(node_ids)

    return is_fixed, prescribed_U, F_ext


def enforce_dirichlet_functional(
    field: torch.Tensor, 
    prescribed: torch.Tensor, 
    is_fixed: torch.Tensor
) -> torch.Tensor:
    """
    TUTORIAL: Pure Functional Dirichlet Enforcer
    ============================================
    
    Instead of in-place mutation (e.g., U[is_fixed] = val), which breaks
    autograd graph tracing, torch.compile, and JAX purity:
    
    `torch.where(condition, if_true, if_false)` returns a new tensor where:
      - Wherever is_fixed is True  -> value is taken from `prescribed`
      - Wherever is_fixed is False -> value is preserved from `field`
      
    This is 100% differentiable and fuses into a single GPU kernel!
    """
    return torch.where(is_fixed, prescribed, field)


# =============================================================================
# Geometric selection of boundary nodes
# =============================================================================
def select_boundary_nodes(nodes: torch.Tensor, side: str, tol: float = 1e-6) -> torch.Tensor:
    """
    Boolean mask of the nodes lying on one side of the bounding box of the mesh.

    Parameters
    ----------
    nodes : (N, 2)
    side  : "left" (x = x_min), "right" (x = x_max), "bottom" (y = y_min), "top" (y = y_max)

    Returns
    -------
    mask : (N,) bool

    The bounding box is taken from the mesh itself, so the same call works for
    every geometry (rectangle, L-bracket, plate with hole, ...).
    """
    x, y = nodes[:, 0], nodes[:, 1]
    # absolute tolerance scaled by the size of the body
    eps = tol * torch.max(torch.max(x) - torch.min(x), torch.max(y) - torch.min(y))
    if side == "left":
        return x <= torch.min(x) + eps
    if side == "right":
        return x >= torch.max(x) - eps
    if side == "bottom":
        return y <= torch.min(y) + eps
    if side == "top":
        return y >= torch.max(y) - eps
    raise ValueError("side must be 'left', 'right', 'bottom' or 'top'")


def dofs_from_node_mask(node_mask: torch.Tensor, components=(0, 1)) -> torch.Tensor:
    """
    Expand a node mask (N,) to a flat DOF mask (2N,) for the given components
    (0 = x, 1 = y).  DOF numbering is node-major:  dof(node a, component i) = 2 a + i.
    """
    per_node = torch.zeros((node_mask.shape[0], 2), dtype=torch.bool, device=node_mask.device)
    for c in components:
        per_node[:, c] = node_mask
    return per_node.reshape(-1)


# =============================================================================
# Boundary edges and consistent (work-equivalent) nodal loads
# =============================================================================
def boundary_edges(conn: torch.Tensor, edge_local_nodes) -> torch.Tensor:
    """
    Find the edges of the mesh that belong to exactly ONE element.

    Interior edges are shared by two elements (counted twice); boundary edges
    appear once.  Edges are identified by the sorted pair of their END nodes.

    Parameters
    ----------
    conn             : (n_el, nen) connectivity
    edge_local_nodes : list of tuples of local node ids, one per element edge,
                       e.g. [(0, 1, 4), (1, 2, 5), ...] (see Elements.py)

    Returns
    -------
    edges : (n_boundary_edges, k)  global node ids [start, end(, mid)] of every
            boundary edge, k = 2 (linear) or 3 (quadratic)
    """
    n_el = conn.shape[0]
    # gather: for every local edge, the global ids of its nodes -> (n_el, k)
    per_edge = [conn[:, list(e)] for e in edge_local_nodes]
    # stack all elements' edges into one list: (n_el * n_edges_per_element, k)
    all_edges = torch.stack(per_edge, dim=1).reshape(-1, per_edge[0].shape[1])

    # identify an edge by its two END nodes, order-independent: key = min * N + max
    big = int(conn.max()) + 1
    a, b = all_edges[:, 0], all_edges[:, 1]
    key = torch.minimum(a, b) * big + torch.maximum(a, b)

    # how many times does each key occur?  boundary edges: exactly once
    _, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
    return all_edges[counts[inverse] == 1]


def edge_traction_load(nodes: torch.Tensor,
                       edges: torch.Tensor,
                       edge_shape_fn: Callable[[torch.Tensor], torch.Tensor],
                       node_mask: torch.Tensor,
                       total_force: Tuple[float, float],
                       n_gauss: int = 3) -> torch.Tensor:
    """
    CONSISTENT nodal forces of a uniform traction applied on a boundary segment.

    The traction t (force per unit reference length) is integrated against the
    edge shape functions:

            f_a = integral_edge  N_a(s) t  |dX/ds| ds  ~=  sum_q  w_q N_a(s_q) t |dX/ds|_q

    This is what makes the load "work-equivalent": for quadratic edges the end
    nodes get 1/6 and the mid node 4/6 of the force -- NOT one third each, as a
    naive equal split would give.

    Parameters
    ----------
    nodes         : (N, 2)
    edges         : (E, k) boundary edges from ``boundary_edges``
    edge_shape_fn : s -> (k,)  1D shape functions (linear or quadratic)
    node_mask     : (N,) bool, the segment where the load acts
    total_force   : (Fx, Fy) total force on the segment; spread uniformly over its length
    n_gauss       : Gauss points per edge

    Returns
    -------
    f : (N, 2) nodal force array (zeros off the loaded segment)
    """
    dtype, device = nodes.dtype, nodes.device
    f = torch.zeros_like(nodes)

    # keep only edges whose nodes ALL lie on the loaded segment
    on_segment = node_mask[edges].all(dim=1)
    edges = edges[on_segment]
    if edges.shape[0] == 0:
        return f

    s_pts, s_wts = Gauss_Quadratures_1D(n_gauss, dtype=dtype, device=device)   # (g,), (g,)
    N_s = vmap(edge_shape_fn)(s_pts)                                           # (g, k)
    dN_s = vmap(lambda s: Grad_Shape_Functions_1D(edge_shape_fn, s))(s_pts)    # (g, k)

    Xe = nodes[edges]                                          # (E, k, 2) edge coordinates
    # tangent dX/ds at every Gauss point of every edge: (E, g, 2)
    tangent = torch.einsum("gk,ekd->egd", dN_s, Xe)
    speed = torch.linalg.norm(tangent, dim=-1)                 # |dX/ds|, (E, g)

    # total length of the loaded segment -> traction magnitude per unit length
    length = torch.einsum("eg,g->", speed, s_wts)
    t = torch.as_tensor(total_force, dtype=dtype, device=device) / length      # (2,)

    # element force contribution  f[e, a, i] = sum_g w_g N_a(s_g) |dX/ds|_g t_i
    f_edge = torch.einsum("g,ga,eg,i->eai", s_wts, N_s, speed, t)             # (E, k, 2)

    # scatter-add the edge forces into the global nodal array
    f.index_add_(0, edges.reshape(-1), f_edge.reshape(-1, 2))
    return f
