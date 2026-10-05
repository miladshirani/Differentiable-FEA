"""
Benchmark: Plate with a circular hole under tension
===================================================

Demonstrates, with the purely functional code base:
  * Gmsh meshing of a curved / re-entrant geometry (any element type)
  * matrix-free inexact Newton-Krylov (PCG) forward solve
  * exact adjoint sensitivity of the compliance with respect to element stiffness
    (one extra linear solve, no assembled matrix)
  * stress-concentration post-processing

    python examples/benchmark_plate_with_hole.py [element_type]     # default quad4
"""
import os
import sys

import numpy as np
import torch
from matplotlib.figure import Figure

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scr.Problem import default_spec, build_problem      # noqa: E402
from scr.Newton_Krylov import newton_krylov_solve        # noqa: E402
from scr.Sensitivity import compute_sensitivity          # noqa: E402
from scr.Postprocess import compute_fields               # noqa: E402
from scr.Plotting import plot_field                      # noqa: E402


def run(element_type: str = "quad4", h: float = 0.12):
    print("=" * 70)
    print("BENCHMARK: Plate with a circular hole under tension   (" + element_type + ")")
    print("=" * 70)

    # 1. problem: geometry, mesh, supports, load -> params
    params = build_problem(default_spec("plate_hole", element_type, h))
    n_el = params["conn"].shape[0]
    theta = torch.ones(n_el, dtype=torch.float64)
    load = float(torch.norm(params["f"]))
    print(f"mesh: {params['nodes'].shape[0]} nodes, {n_el} {element_type} elements")

    # 2. forward solve (matrix-free Newton-Krylov)
    u, info = newton_krylov_solve(params, theta, n_load_steps=3, tol=1e-8 * load, verbose=True)
    fields = compute_fields(u, theta, params)
    print(f"\nconverged={info['converged']}  max|u|={fields['u_mag'].max():.4f}  "
          f"max von Mises={fields['mises'].max():.1f}")

    # 3. adjoint sensitivity of the compliance  J = f . u  with respect to theta
    loss_fn = lambda w: torch.dot(params["f"], w)
    _, dJ_dtheta = compute_sensitivity(params, theta, loss_fn, n_load_steps=3, tol=1e-8 * load)
    print(f"compliance J = {loss_fn(u):.5e};  dJ/dtheta in [{dJ_dtheta.min():.3e}, {dJ_dtheta.max():.3e}]")

    # 4. figure: displacement | von Mises | stiffness importance -dJ/dtheta
    fig = Figure(figsize=(17, 5.2))
    axes = fig.subplots(1, 3)
    nodes, conn = params["nodes"].numpy(), params["conn"].numpy()
    pc = plot_field(axes[0], nodes, conn, element_type, fields["u_mag_n"].numpy(), u.numpy(), 1.0)
    fig.colorbar(pc, ax=axes[0], label="|u|")
    axes[0].set_title("Displacement magnitude (deformed)")
    pc = plot_field(axes[1], nodes, conn, element_type, fields["mises_n"].numpy(), u.numpy(), 1.0, cmap="inferno")
    fig.colorbar(pc, ax=axes[1], label="von Mises")
    axes[1].set_title("von Mises stress (near the hole)")
    # element-wise quantity: colour each element by its sensitivity via nodal averaging
    count = np.bincount(conn.reshape(-1), minlength=nodes.shape[0]).clip(min=1)
    imp = np.bincount(conn.reshape(-1), weights=np.repeat(-dJ_dtheta.numpy(), conn.shape[1]),
                      minlength=nodes.shape[0]) / count
    pc = plot_field(axes[2], nodes, conn, element_type, imp, None, cmap="magma")
    fig.colorbar(pc, ax=axes[2], label="-dJ/dtheta")
    axes[2].set_title("Adjoint sensitivity (stiffness importance)")
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "plate_with_hole_deformation.png")
    fig.savefig(path, dpi=200)
    print(f"saved {os.path.abspath(path)}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "quad4")
