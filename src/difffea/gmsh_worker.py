"""
gmsh_worker.py
==============

Runs Gmsh in a SEPARATE PROCESS.  This file imports only NumPy and Gmsh -- never PyTorch.

Why a separate process?  The Gmsh wheel ships its own OpenMP runtime, and PyTorch has
another one.  When both are loaded into the same Python process, PyTorch's parallel
code (used once a mesh is big enough) can jump into Gmsh's runtime and crash the whole
interpreter with a segmentation fault (in a GUI: "the page freezes").  Keeping Gmsh in
a child process makes the clash impossible.

Command-line protocol (used by ``Gmsh_Mesh.generate_gmsh_mesh``):
    stdin : JSON  {geometry, params, element_family, order, gmsh_type, n_nodes, incomplete, h}
    stdout: path of a ``.npz`` file with arrays  ``xy`` (N, 2)  and  ``conn`` (n_el, nen)
"""

import json
import sys
import tempfile
from typing import List

import numpy as np
import gmsh


# =============================================================================
# Geometry builders.  Each takes (h) = target element size, adds entities to the
# Gmsh "geo" kernel and returns the list of curve loops [outer, hole1, ...].
# Arcs are always added counter-clockwise; a minus sign reverses a curve in a loop.
# =============================================================================
def _geometry_rectangle(p: dict, h: float) -> List[int]:
    """Rectangle  [0, width] x [0, height]."""
    g = gmsh.model.geo
    w, hh = p["width"], p["height"]
    pts = [g.addPoint(0, 0, 0, h), g.addPoint(w, 0, 0, h),
           g.addPoint(w, hh, 0, h), g.addPoint(0, hh, 0, h)]
    lines = [g.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return [g.addCurveLoop(lines)]


def _geometry_l_bracket(p: dict, h: float) -> List[int]:
    """
    L-shaped bracket (arm thickness t):

        (0,H) -- (t,H)
          |        |
          |      (t,t) ------- (L,t)
          |                       |
        (0,0) ------------------ (L,0)
    """
    g = gmsh.model.geo
    L, H, t = p["L"], p["H"], p["t"]
    corners = [(0, 0), (L, 0), (L, t), (t, t), (t, H), (0, H)]
    pts = [g.addPoint(x, y, 0, h) for x, y in corners]
    lines = [g.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    return [g.addCurveLoop(lines)]


def _circle_loop(cx: float, cy: float, r: float, h: float) -> int:
    """Closed circle made of four counter-clockwise quarter arcs; refined (0.6 h)."""
    g = gmsh.model.geo
    centre = g.addPoint(cx, cy, 0, h)
    q = [g.addPoint(cx + r, cy, 0, 0.6 * h), g.addPoint(cx, cy + r, 0, 0.6 * h),
         g.addPoint(cx - r, cy, 0, 0.6 * h), g.addPoint(cx, cy - r, 0, 0.6 * h)]
    arcs = [g.addCircleArc(q[i], centre, q[(i + 1) % 4]) for i in range(4)]
    return g.addCurveLoop(arcs)


def _geometry_plate_hole(p: dict, h: float) -> List[int]:
    """Rectangle with a circular hole at its centre."""
    outer = _geometry_rectangle(p, h)[0]
    hole = _circle_loop(p["width"] / 2.0, p["height"] / 2.0, p["radius"], h)
    return [outer, hole]


def _geometry_notched_plate(p: dict, h: float) -> List[int]:
    """
    Rectangle with a semicircular notch of radius r in the middle of the bottom
    and of the top edge (a classical stress-concentration specimen).
    """
    g = gmsh.model.geo
    w, hh, r = p["width"], p["height"], p["radius"]
    cx = w / 2.0
    fine = 0.6 * h
    # bottom edge, left to right, with the notch bulging INTO the plate
    p0 = g.addPoint(0, 0, 0, h)
    A = g.addPoint(cx - r, 0, 0, fine)
    T = g.addPoint(cx, r, 0, fine)
    B = g.addPoint(cx + r, 0, 0, fine)
    p1 = g.addPoint(w, 0, 0, h)
    # top edge, right to left
    p2 = g.addPoint(w, hh, 0, h)
    D = g.addPoint(cx + r, hh, 0, fine)
    S = g.addPoint(cx, hh - r, 0, fine)
    E = g.addPoint(cx - r, hh, 0, fine)
    p3 = g.addPoint(0, hh, 0, h)
    Cb = g.addPoint(cx, 0, 0, h)         # arc centres
    Ct = g.addPoint(cx, hh, 0, h)

    l_bot0 = g.addLine(p0, A)
    arc_TA = g.addCircleArc(T, Cb, A)    # CCW  90 deg -> 180 deg   (we traverse it reversed)
    arc_BT = g.addCircleArc(B, Cb, T)    # CCW   0 deg ->  90 deg   (reversed)
    l_bot1 = g.addLine(B, p1)
    l_right = g.addLine(p1, p2)
    l_top0 = g.addLine(p2, D)
    arc_SD = g.addCircleArc(S, Ct, D)    # CCW 270 deg -> 360 deg   (reversed)
    arc_ES = g.addCircleArc(E, Ct, S)    # CCW 180 deg -> 270 deg   (reversed)
    l_top1 = g.addLine(E, p3)
    l_left = g.addLine(p3, p0)
    return [g.addCurveLoop([l_bot0, -arc_TA, -arc_BT, l_bot1, l_right,
                            l_top0, -arc_SD, -arc_ES, l_top1, l_left])]



BUILDERS = {"rectangle": _geometry_rectangle, "l_bracket": _geometry_l_bracket,
            "plate_hole": _geometry_plate_hole, "notched_plate": _geometry_notched_plate}


def mesh_geometry(request: dict):
    """
    Mesh a geometry and return (xy (N, 2) float64, conn (n_el, nen) int64).
    ``request`` is the dictionary described in the module docstring.
    """
    is_quad = request["element_family"] == "quad"

    # Pure-quad recipe: (1) mesh with triangles of size 2h, (2) merge pairs of triangles
    # into quads (blossom recombination), (3) split EVERY element into four (all-quad
    # subdivision).  Step 3 guarantees no triangle survives and halves the edge length,
    # so the final quads have edges ~ h.
    h_size = 2.0 * request["h"] if is_quad else request["h"]

    # interruptible=False: do not install signal handlers (they only work in the main thread)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)            # silence the mesher
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add(request["geometry"])

        # 1. geometry: curve loops -> one plane surface (first loop = outer boundary)
        loops = BUILDERS[request["geometry"]](request["params"], h_size)
        gmsh.model.geo.addPlaneSurface(loops)
        gmsh.model.geo.synchronize()

        # 2. mesh controls
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_size)
        if is_quad:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)             # triangles -> quads
            gmsh.option.setNumber("Mesh.Algorithm", 8)                # frontal-Delaunay for quads
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)   # blossom matching
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)     # 1 = split into all-quads
        if request["incomplete"]:
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)    # serendipity: no centre node

        # 3. generate the first-order mesh, then raise the order if needed
        gmsh.model.mesh.generate(2)
        if request["order"] == 2:
            gmsh.model.mesh.setOrder(2)

        # 4. nodes: tag -> contiguous index via a lookup table
        tags, coords, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, dtype=np.int64)
        xy = np.asarray(coords).reshape(-1, 3)[:, :2].astype(np.float64)
        lookup = np.full(tags.max() + 1, -1, dtype=np.int64)
        lookup[tags] = np.arange(len(tags))

        # 5. elements: we need exactly one type, the requested one
        types, _, node_lists = gmsh.model.mesh.getElements(dim=2)
        types = [int(t) for t in types]
        if types != [request["gmsh_type"]]:
            raise RuntimeError(f"Gmsh could not build a pure mesh of type {request['gmsh_type']} (got {types}).")
        conn = np.asarray(node_lists[0], dtype=np.int64).reshape(-1, request["n_nodes"])
        return xy, lookup[conn]
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    req = json.loads(sys.stdin.read())
    xy_, conn_ = mesh_geometry(req)
    out = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    out.close()
    np.savez(out.name, xy=xy_, conn=conn_)
    print(out.name)
