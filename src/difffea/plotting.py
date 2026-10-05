"""
plotting.py
===========

Matplotlib visualisation of meshes and fields on deformed configurations.
Pure functions: each takes data and an Axes and draws; nothing is stored.

Geometry used for drawing
-------------------------
Contours are drawn on the CORNER nodes of the elements (every element is split
into triangles for ``tripcolor``); element edges are drawn as polylines through
ALL the edge nodes (so quadratic elements show their mid-side nodes and curved
boundaries).
"""

import io

import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

from difffea.elements import get_element
from difffea.postprocess import compute_fields


def _triangles_of(conn_np: np.ndarray, element_type: str) -> np.ndarray:
    """Split every element into triangles built from its CORNER nodes -> (n_tri, 3)."""
    c = get_element(element_type)["n_corners"]
    corners = conn_np[:, :c]
    if c == 3:
        return corners
    # quad: two triangles (0,1,2) and (0,2,3)
    return np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]], axis=0)


def _edge_segments(xy: np.ndarray, conn_np: np.ndarray, element_type: str) -> np.ndarray:
    """
    Line segments of all element edges, as an array (n_seg, 2, 2).  Quadratic
    edges are drawn as two segments through the mid-side node.
    """
    segs = []
    for e in get_element(element_type)["edges"]:
        if len(e) == 2:
            order = [e[0], e[1]]
        else:
            order = [e[0], e[2], e[1]]               # start -> mid -> end
        pts = xy[conn_np[:, order]]                  # (n_el, 2 or 3, 2)
        for a in range(len(order) - 1):
            segs.append(np.stack([pts[:, a], pts[:, a + 1]], axis=1))
    return np.concatenate(segs, axis=0)


def plot_mesh(ax, nodes, conn, element_type, color="0.35", linewidth=0.5, show_nodes=False):
    """Draw the undeformed mesh (edges only) on ``ax``."""
    xy, cn = np.asarray(nodes), np.asarray(conn)
    ax.add_collection(LineCollection(_edge_segments(xy, cn, element_type),
                                     colors=color, linewidths=linewidth))
    if show_nodes:
        ax.plot(xy[:, 0], xy[:, 1], ".", color="crimson", markersize=3)
    ax.set_aspect("equal")
    ax.autoscale()


def plot_field(ax, nodes, conn, element_type, nodal_values, u=None, scale=1.0,
               cmap="viridis", show_edges=True, ghost=True, label="", vmin=None, vmax=None):
    """
    Filled contour of a nodal field on the (scaled) deformed configuration.

    Parameters
    ----------
    nodes, conn   : mesh
    nodal_values  : (N,) values to colour
    u             : (2N,) or (N, 2) displacement; None draws the undeformed shape
    scale         : displacement magnification factor
    ghost         : also draw the undeformed outline in light grey
    Returns the ScalarMappable (for a colour bar).
    """
    xy0 = np.asarray(nodes)
    cn = np.asarray(conn)
    xy = xy0 if u is None else xy0 + scale * np.asarray(u).reshape(-1, 2)

    if ghost and u is not None:
        ax.add_collection(LineCollection(_edge_segments(xy0, cn, element_type),
                                         colors="0.8", linewidths=0.4, linestyles="--"))

    tri = Triangulation(xy[:, 0], xy[:, 1], _triangles_of(cn, element_type))
    # quadratic meshes have mid-side nodes that are not in any triangle: matplotlib
    # ignores unused points, so colouring by the full nodal vector is fine.
    pc = ax.tripcolor(tri, np.asarray(nodal_values), shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    if show_edges:
        ax.add_collection(LineCollection(_edge_segments(xy, cn, element_type),
                                         colors="k", linewidths=0.25, alpha=0.6))
    ax.set_aspect("equal")
    ax.autoscale()
    return pc


def make_deformation_gif(params, snapshots, field_key, field_label="", cmap="viridis",
                         scale=1.0, n_frames=24, frame_ms=90, theta=None) -> bytes:
    """
    Animate the loading history: the body deforms from the undeformed shape (load
    factor 0) to the final state while a field is shown as a contour.

    Frames are placed uniformly in load factor.  Between two converged states the
    displacement is interpolated linearly, so the animation is smooth even when
    the solver used only a few load steps; every frame's fields are then computed
    from its displacement with ``compute_fields`` (exact kinematics and stress).

    Parameters
    ----------
    params     : problem dictionary
    snapshots  : list of (load_factor, u) from ``newton_krylov_solve``
    field_key  : key of ``compute_fields`` to colour, e.g. "mises_n"
    n_frames   : number of animation frames;  frame_ms : milliseconds per frame

    Returns the GIF file as bytes.
    """
    from PIL import Image
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if theta is None:
        theta = torch.ones(params["conn"].shape[0], dtype=params["nodes"].dtype, device=params["nodes"].device)
    lfs = torch.tensor([s[0] for s in snapshots], dtype=params["nodes"].dtype,
                       device=params["nodes"].device)                              # (S,)
    U = torch.stack([s[1] for s in snapshots])                                   # (S, 2N)

    # --- 1. displacement of every frame by linear interpolation in load factor
    targets = torch.linspace(0.0, float(lfs[-1]), n_frames, dtype=lfs.dtype, device=lfs.device)
    idx = torch.clamp(torch.searchsorted(lfs, targets, right=True) - 1, 0, len(lfs) - 2)
    t = (targets - lfs[idx]) / (lfs[idx + 1] - lfs[idx])
    frames_u = U[idx] + t[:, None] * (U[idx + 1] - U[idx])                       # (F, 2N)

    # --- 2. fields of every frame, and common colour limits
    values = [compute_fields(u, theta, params)[field_key].cpu().numpy() for u in frames_u]
    vmin, vmax = min(v.min() for v in values), max(v.max() for v in values)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12

    # --- 3. fixed axis limits that contain every deformed configuration
    nodes_np, conn_np = params["nodes"].cpu().numpy(), params["conn"].cpu().numpy()
    xy_all = np.concatenate([nodes_np + scale * u.cpu().numpy().reshape(-1, 2) for u in frames_u])
    pad = 0.05 * max(np.ptp(xy_all[:, 0]), np.ptp(xy_all[:, 1]))
    xlim = (xy_all[:, 0].min() - pad, xy_all[:, 0].max() + pad)
    ylim = (xy_all[:, 1].min() - pad, xy_all[:, 1].max() + pad)

    # --- 4. draw the frames
    images = []
    for k, (u, val) in enumerate(zip(frames_u, values)):
        fig = Figure(figsize=(7, 4.6), dpi=80)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        pc = plot_field(ax, nodes_np, conn_np, params["element_type"], val, u.cpu().numpy(), scale,
                        cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=ax, label=field_label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{field_label}   load factor {targets[k]:.2f}")
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))

    # --- 5. encode (the last frame is held longer), looping forever
    durations = [frame_ms] * (len(images) - 1) + [900]
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False)
    return buf.getvalue()
