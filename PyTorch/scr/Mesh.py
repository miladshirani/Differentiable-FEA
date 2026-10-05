"""
Mesh.py
=======

STRUCTURED mesh generators for a rectangle [0, Lx] x [0, Ly] -- no mesher
needed, only index arithmetic on a grid of node ids.  They are the simplest
place to learn what a mesh is:

    nodes : (N, 2)       coordinates of every node
    conn  : (n_el, nen)  for every element, the ids of its nodes in the local
                         order expected by the shape functions

For unstructured meshes of curved geometries see Gmsh_Mesh.py.
"""

import torch


def create_2d_quad_mesh(
    Lx: float, 
    Ly: float, 
    nx: int, 
    ny: int, 
    dtype=torch.float32, 
    device="cpu"
):
    """
    TUTORIAL: Structured 2D 4-Node Quadrilateral (Q4) Mesh Generator
    ================================================================
    
    CONVENTION: Pure [X, Y] Order (Dimension 0 = X, Dimension 1 = Y)
    ----------------------------------------------------------------
    - Everything is indexed as [x, y]:
        node_grid[i, j] means: index i along X, index j along Y.
    - No transposing (.T) is needed anywhere!

    HOW NODES ARE NUMBERED (Column-by-column along Y, then right along X):
    ----------------------------------------------------------------------
    For a 2x2 element mesh (nx=2, ny=2 -> 3x3 nodes):

        y = Ly:   2 --------- 5 --------- 8
                  |  Elem 1   |  Elem 3   |
        y = mid:  1 --------- 4 --------- 7
                  |  Elem 0   |  Elem 2   |
        y = 0:    0 --------- 3 --------- 6
                 x=0        x=mid        x=Lx

    LOCAL ELEMENT CORNERS (Strictly Counter-Clockwise):
    ---------------------------------------------------
        Node 3 (top-left)  * -------- * Node 2 (top-right)
                           |          |
                           | Element  |
                           |          |
        Node 0 (bot-left)  * -------- * Node 1 (bot-right)
    """

    # 1. Total counts
    num_nodes = (nx + 1) * (ny + 1)
    num_elements = nx * ny

    # 2. 1D coordinate ranges
    x = torch.linspace(0.0, Lx, nx + 1, dtype=dtype, device=device)
    y = torch.linspace(0.0, Ly, ny + 1, dtype=dtype, device=device)

    # 3. 2D Coordinate Grids:
    # Dimension 0 is X (size: nx + 1)
    # Dimension 1 is Y (size: ny + 1)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    # Combine coordinates directly without any transposing!
    # Shape: (num_nodes, 2)
    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    # 4. Node ID Grid:
    # Dimension 0 is X, Dimension 1 is Y.
    # Shape: (nx + 1, ny + 1)
    #
    #   node_grid = [
    #       [0, 1, 2],   <- Column 0 (x = 0):  nodes 0, 1, 2
    #       [3, 4, 5],   <- Column 1 (x = mid): nodes 3, 4, 5
    #       [6, 7, 8]    <- Column 2 (x = Lx):  nodes 6, 7, 8
    #   ]
    #
    # node_grid[i, j] directly gives the Node ID at x-index i and y-index j!
    node_grid = torch.arange(num_nodes, device=device).reshape(nx + 1, ny + 1)

    # 5. Extract the 4 Corners using your [x, y] slicing:
    # For any element at (i, j):
    #   - Moving right (x -> x+1) shifts the FIRST dimension:  1 : nx+1
    #   - Moving up    (y -> y+1) shifts the SECOND dimension: 1 : ny+1
    
    # Node 0 (Bottom-Left):  x = i,     y = j
    n0 = node_grid[0:nx,   0:ny  ].flatten()

    # Node 1 (Bottom-Right): x = i + 1, y = j      (shift X right by 1)
    n1 = node_grid[1:nx+1, 0:ny  ].flatten()

    # Node 2 (Top-Right):    x = i + 1, y = j + 1  (shift X right & Y up)
    n2 = node_grid[1:nx+1, 1:ny+1].flatten()

    # Node 3 (Top-Left):     x = i,     y = j + 1  (shift Y up by 1)
    n3 = node_grid[0:nx,   1:ny+1].flatten()

    # 6. Final Connectivity Matrix: shape (num_elements, 4)
    # Every row is [bot-left, bot-right, top-right, top-left] (CCW)
    elements = torch.stack([n0, n1, n2, n3], dim=-1)

    return nodes, elements


def create_2d_quad9_mesh(
    Lx: float, 
    Ly: float, 
    nx: int, 
    ny: int, 
    dtype=torch.float32, 
    device="cpu"
):
    """
    TUTORIAL: Structured 2D 9-Node Biquadratic Quadrilateral (Quad9) Mesh Generator
    ==============================================================================
    
    PURPOSE:
    --------
    Generates nodes and connectivity for 9-node Lagrangian quad elements.
    
    NODE NUMBERING PER ELEMENT (Matching shape_fn_quad9 in shape_functions_2D.py):
    ------------------------------------------------------------------------------
        4 (top-left)  * -------- 7 (top-mid) -------- * 3 (top-right)
                      |                               |
        8 (mid-left)  *           9 (center)          * 6 (mid-right)
                      |                               |
        1 (bot-left)  * -------- 5 (bot-mid) -------- * 2 (bot-right)
        
    Ordering: [N1, N2, N3, N4, N5, N6, N7, N8, N9]
      - Nodes 1 to 4: 4 corner nodes (Counter-Clockwise)
      - Nodes 5 to 8: 4 mid-edge nodes
      - Node 9:       1 interior center node
      
    SHAPES:
    -------
      - `nodes`: ((2*nx + 1) * (2*ny + 1), 2)
      - `elements`: (nx * ny, 9)
    """

    # 1. Total grid counts (each Quad9 element spans 2 grid steps)
    grid_nx = 2 * nx
    grid_ny = 2 * ny
    num_nodes = (grid_nx + 1) * (grid_ny + 1)

    # 2. Get the grid nodes by calling create_2d_quad_mesh with 2*nx, 2*ny!
    nodes, _ = create_2d_quad_mesh(Lx, Ly, grid_nx, grid_ny, dtype=dtype, device=device)

    # 3. Build the node ID grid matching [X, Y] indexing: shape (grid_nx + 1, grid_ny + 1)
    node_grid = torch.arange(num_nodes, device=device).reshape(grid_nx + 1, grid_ny + 1)

    # -------------------------------------------------------------------------
    # 4. Extract All 9 Nodes using Stride-2 Slices
    # -------------------------------------------------------------------------
    # Elements step by 2 in X and 2 in Y:
    #   i ranges over: 0, 2, 4, ..., 2*nx - 2  -> slice: [0 : grid_nx : 2]
    #   i + 1:         1, 3, 5, ..., 2*nx - 1  -> slice: [1 : grid_nx : 2]
    #   i + 2:         2, 4, 6, ..., 2*nx      -> slice: [2 : grid_nx + 1 : 2]
    
    # 4 Corner Nodes (CCW)
    n1 = node_grid[0 : grid_nx : 2,     0 : grid_ny : 2    ].flatten()  # (i,   j)   bot-left
    n2 = node_grid[2 : grid_nx + 1 : 2, 0 : grid_ny : 2    ].flatten()  # (i+2, j)   bot-right
    n3 = node_grid[2 : grid_nx + 1 : 2, 2 : grid_ny + 1 : 2].flatten()  # (i+2, j+2) top-right
    n4 = node_grid[0 : grid_nx : 2,     2 : grid_ny + 1 : 2].flatten()  # (i,   j+2) top-left

    # 4 Mid-Edge Nodes
    n5 = node_grid[1 : grid_nx : 2,     0 : grid_ny : 2    ].flatten()  # (i+1, j)   bot-mid
    n6 = node_grid[2 : grid_nx + 1 : 2, 1 : grid_ny : 2    ].flatten()  # (i+2, j+1) right-mid
    n7 = node_grid[1 : grid_nx : 2,     2 : grid_ny + 1 : 2].flatten()  # (i+1, j+2) top-mid
    n8 = node_grid[0 : grid_nx : 2,     1 : grid_ny : 2    ].flatten()  # (i,   j+1) left-mid

    # 1 Center Node
    n9 = node_grid[1 : grid_nx : 2,     1 : grid_ny : 2    ].flatten()  # (i+1, j+1) center

    # -------------------------------------------------------------------------
    # 5. Assemble Connectivity Matrix: shape (nx * ny, 9)
    # -------------------------------------------------------------------------
    elements = torch.stack([n1, n2, n3, n4, n5, n6, n7, n8, n9], dim=-1)

    return nodes, elements



def create_2d_quad8_mesh(
    Lx: float, 
    Ly: float, 
    nx: int, 
    ny: int, 
    dtype=torch.float32, 
    device="cpu"
):
    """
    TUTORIAL: Structured 2D 8-Node Serendipity Quadrilateral (Quad8) Mesh Generator
    ==============================================================================
    
    PURPOSE:
    --------
    Generates nodes and connectivity for 8-node Serendipity quad elements.
    
    NODE NUMBERING PER ELEMENT (Matching shape_fn_quad8 in shape_functions_2D.py):
    ------------------------------------------------------------------------------
        4 (top-left)  * -------- 7 (top-mid) -------- * 3 (top-right)
                      |                               |
        8 (mid-left)  *          (NO CENTER)          * 6 (mid-right)
                      |                               |
        1 (bot-left)  * -------- 5 (bot-mid) -------- * 2 (bot-right)
        
    Ordering: [N1, N2, N3, N4, N5, N6, N7, N8]
      - Nodes 1 to 4: 4 corner nodes (Counter-Clockwise)
      - Nodes 5 to 8: 4 mid-edge nodes
      
    SHAPES:
    -------
      - `nodes`: ((2*nx + 1)*(2*ny + 1) - nx*ny, 2)  <- all grid points minus the centers!
      - `elements`: (nx * ny, 8)
    """

    # 1. Total grid counts (each element spans 2 grid steps)
    grid_nx = 2 * nx
    grid_ny = 2 * ny
    total_grid_nodes = (grid_nx + 1) * (grid_ny + 1)

    # 2. Generate the full underlying grid of coordinates
    all_nodes, _ = create_2d_quad_mesh(Lx, Ly, grid_nx, grid_ny, dtype=dtype, device=device)

    # 3. Create a 2D boolean mask of active nodes: shape (grid_nx + 1, grid_ny + 1)
    # Start by assuming all nodes are used
    is_active_node = torch.ones((grid_nx + 1, grid_ny + 1), dtype=torch.bool, device=device)

    # The center nodes of all Quad8 elements are at (odd X, odd Y):
    # X indices: 1, 3, 5, ... -> slice [1 : grid_nx : 2]
    # Y indices: 1, 3, 5, ... -> slice [1 : grid_ny : 2]
    # Turn OFF the center nodes because Quad8 doesn't use them!
    is_active_node[1 : grid_nx : 2, 1 : grid_ny : 2] = False

    # Flatten the mask
    active_mask = is_active_node.flatten()

    # 4. Filter the node coordinates: keep ONLY active nodes (no floating points!)
    nodes = all_nodes[active_mask]

    # 5. Create a Re-indexing Map:
    # Maps old grid IDs (which had gaps) to new consecutive IDs (0, 1, 2, ... N_active-1)
    old_to_new_id = torch.full((total_grid_nodes,), -1, dtype=torch.int64, device=device)
    old_to_new_id[active_mask] = torch.arange(nodes.shape[0], device=device)

    # 6. Extract the 8 nodes using the old grid IDs (same stride-2 slicing as Quad9!)
    node_grid = torch.arange(total_grid_nodes, device=device).reshape(grid_nx + 1, grid_ny + 1)

    # Corners (CCW)
    n1 = node_grid[0 : grid_nx : 2,     0 : grid_ny : 2    ].flatten()  # (i,   j)   bot-left
    n2 = node_grid[2 : grid_nx + 1 : 2, 0 : grid_ny : 2    ].flatten()  # (i+2, j)   bot-right
    n3 = node_grid[2 : grid_nx + 1 : 2, 2 : grid_ny + 1 : 2].flatten()  # (i+2, j+2) top-right
    n4 = node_grid[0 : grid_nx : 2,     2 : grid_ny + 1 : 2].flatten()  # (i,   j+2) top-left

    # Mid-edges (NO center n9!)
    n5 = node_grid[1 : grid_nx : 2,     0 : grid_ny : 2    ].flatten()  # (i+1, j)   bot-mid
    n6 = node_grid[2 : grid_nx + 1 : 2, 1 : grid_ny : 2    ].flatten()  # (i+2, j+1) right-mid
    n7 = node_grid[1 : grid_nx : 2,     2 : grid_ny + 1 : 2].flatten()  # (i+1, j+2) top-mid
    n8 = node_grid[0 : grid_nx : 2,     1 : grid_ny : 2    ].flatten()  # (i,   j+1) left-mid

    # Stack the 8 old node IDs: shape (nx * ny, 8)
    old_elements = torch.stack([n1, n2, n3, n4, n5, n6, n7, n8], dim=-1)

    # 7. Remap to the new clean node IDs:
    elements = old_to_new_id[old_elements]

    return nodes, elements



def create_2d_tri3_mesh(
    Lx: float, 
    Ly: float, 
    nx: int, 
    ny: int, 
    dtype=torch.float32, 
    device="cpu"
):
    """
    TUTORIAL: Structured 2D 3-Node Triangular (Tri3) Mesh Generator
    ==============================================================
    
    DIAGONAL SPLIT FROM NODE 3 TO NODE 1 (BACKSLASH \\):
    ---------------------------------------------------
        Node 3 (top-left)  * -------- * Node 2 (top-right)
                           | \\        |
                           |   \\ Tri 2|
                           |     \\    |
                           | Tri 1 \\  |
                           |         \\|
        Node 0 (bot-left)  * -------- * Node 1 (bot-right)

    THE TWO TRIANGLES (BOTH STRICTLY COUNTER-CLOCKWISE):
    ----------------------------------------------------
      1. Tri 1 (Lower-Left):
         - Nodes: [n0, n1, n3]
         - Traversal: Node 0 (bot-left) -> Node 1 (bot-right) -> Node 3 (top-left)
         - Orientation: Counter-Clockwise (Area > 0, det(J) > 0)

      2. Tri 2 (Upper-Right):
         - Nodes: [n1, n2, n3]
         - Traversal: Node 1 (bot-right) -> Node 2 (top-right) -> Node 3 (top-left)
         - Orientation: Counter-Clockwise (Area > 0, det(J) > 0)

    SHAPES:
    -------
      - `nodes`: Same grid vertices -> shape: ((nx+1)*(ny+1), 2)
      - `tri_elements`: 2 triangles per quad -> shape: (2 * nx * ny, 3)
    """

    # -------------------------------------------------------------------------
    # STEP 1: Generate Base Quad Mesh
    # -------------------------------------------------------------------------
    # `nodes` has shape: ((nx+1)*(ny+1), 2)
    # `quad_elements` has shape: (nx*ny, 4), where columns are [n0, n1, n2, n3]
    nodes, quad_elements = create_2d_quad_mesh(
        Lx, Ly, nx, ny, dtype=dtype, device=device
    )

    # -------------------------------------------------------------------------
    # STEP 2: Extract the 4 Quad Corners for ALL Elements
    # -------------------------------------------------------------------------
    n0 = quad_elements[:, 0]  # Bottom-Left
    n1 = quad_elements[:, 1]  # Bottom-Right
    n2 = quad_elements[:, 2]  # Top-Right
    n3 = quad_elements[:, 3]  # Top-Left

    # -------------------------------------------------------------------------
    # STEP 3: Form the 2 Triangles per the Backslash (\\) Cut
    # -------------------------------------------------------------------------
    # Tri 1 (Lower-Left half): connects [n0, n1, n3]
    tri1 = torch.stack([n0, n1, n3], dim=-1)  # shape: (nx*ny, 3)

    # Tri 2 (Upper-Right half): connects [n1, n2, n3]
    tri2 = torch.stack([n1, n2, n3], dim=-1)  # shape: (nx*ny, 3)

    # -------------------------------------------------------------------------
    # STEP 4: Combine into the Final Triangular Connectivity Matrix
    # -------------------------------------------------------------------------
    # Stacking tri1 and tri2 along dimension 0 gives shape: (2 * nx * ny, 3)
    tri_elements = torch.cat([tri1, tri2], dim=0)

    return nodes, tri_elements





def create_2d_tri6_mesh(
    Lx: float, 
    Ly: float, 
    nx: int, 
    ny: int, 
    dtype=torch.float32, 
    device="cpu"
):
    """
    TUTORIAL: Structured 2D 6-Node Quadratic Triangular (Tri6) Mesh Generator
    ========================================================================
    
    PURPOSE:
    --------
    Generates nodes and connectivity for 6-node quadratic triangles by splitting
    each Quad9 element along the diagonal from top-left (N4) to bottom-right (N2).
    
    The center node (N9) serves as the midpoint along that internal diagonal!

    NODE ORDERING PER TRIANGLE (Matching shape_fn_tri6 in shape_functions_2D.py):
    ----------------------------------------------------------------------------
           3 (top)
           | \\
           6   5   (mid-sides)
           |     \\
           1 -- 4 -- 2 (right)
           
    Ordering: [Corner1, Corner2, Corner3, Mid4, Mid5, Mid6]
    
    SHAPES:
    -------
      - `nodes`: ((2*nx + 1) * (2*ny + 1), 2)
      - `elements`: (2 * nx * ny, 6)
    """

    # 1. Get the Quad9 mesh:
    # `quad9_elements` has shape (nx * ny, 9) with columns [n1, n2, n3, n4, n5, n6, n7, n8, n9]
    nodes, quad9_elements = create_2d_quad9_mesh(
        Lx, Ly, nx, ny, dtype=dtype, device=device
    )

    # 2. Unpack all 9 nodes of the Quad9 elements:
    n1 = quad9_elements[:, 0]  # bot-left
    n2 = quad9_elements[:, 1]  # bot-right
    n3 = quad9_elements[:, 2]  # top-right
    n4 = quad9_elements[:, 3]  # top-left
    n5 = quad9_elements[:, 4]  # bot-mid
    n6 = quad9_elements[:, 5]  # right-mid
    n7 = quad9_elements[:, 6]  # top-mid
    n8 = quad9_elements[:, 7]  # left-mid
    n9 = quad9_elements[:, 8]  # center

    # 3. Form Lower-Left Triangle (Tri 1):
    # Corners: [n1, n2, n4], Mid-edges: [n5, n9, n8]
    tri1 = torch.stack([n1, n2, n4, n5, n9, n8], dim=-1)  # shape (nx * ny, 6)

    # 4. Form Upper-Right Triangle (Tri 2):
    # Corners: [n2, n3, n4], Mid-edges: [n6, n7, n9]
    tri2 = torch.stack([n2, n3, n4, n6, n7, n9], dim=-1)  # shape (nx * ny, 6)

    # 5. Concatenate all triangles: shape (2 * nx * ny, 6)
    tri6_elements = torch.cat([tri1, tri2], dim=0)

    return nodes, tri6_elements


# =============================================================================
# Dispatcher: one entry point for all element types
# =============================================================================
def create_rectangle_mesh(element_type: str, Lx: float, Ly: float, nx: int, ny: int,
                          dtype=torch.float64, device="cpu"):
    """
    Structured mesh of the rectangle [0, Lx] x [0, Ly] with nx x ny cells of
    the requested element type ("quad4", "quad8", "quad9", "tri3", "tri6").
    Triangle meshes split every cell in two (so they have 2*nx*ny elements).

    Returns (nodes (N, 2), conn (n_el, nen)).
    """
    builders = {"quad4": create_2d_quad_mesh, "quad8": create_2d_quad8_mesh,
                "quad9": create_2d_quad9_mesh, "tri3": create_2d_tri3_mesh,
                "tri6": create_2d_tri6_mesh}
    if element_type not in builders:
        raise ValueError(f"Unknown element type '{element_type}'. Choose one of {list(builders)}.")
    return builders[element_type](Lx, Ly, nx, ny, dtype=dtype, device=device)
