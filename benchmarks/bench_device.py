"""
bench_device.py
===============

CPU-vs-GPU measurements for the matrix-free tangent product and a full Newton-Krylov solve
(roadmap step 2).  The same script runs on a laptop CPU and on a CUDA GPU (e.g. a free Colab T4):

    python benchmarks/bench_device.py --device cpu
    python benchmarks/bench_device.py --device cuda

For each mesh size it reports (median of repeated runs, GPU work synchronised before the clock stops):
  * time of R(u) and of one tangent product A(v) for the three evaluations "jvp", "linearize", "qp"
    (see operators.py), the one-off setup cost of "qp", and peak GPU memory;
and for one moderate size a complete nonlinear solve (uniaxial tension, neo-Hookean, Jacobi-preconditioned
CG) with the reference ("jvp") and the fast ("qp") tangent: time, Newton and CG iterations, final residual,
and -- on a GPU -- the maximum difference to the CPU solution of the same problem (correctness check).

Everything is float64 (float32 is blocked by a PyTorch forward-AD issue, see docs/KNOWN_ISSUES.md).  The
free Colab T4 has weak float64 units (1/32 of float32 rate); these kernels are memory-bound, but only the
measurement tells how much that matters.

Output: printed table + benchmarks/results/bench_<device>.json (device name, versions; no paths, no user data).
"""
import argparse
import json
import os
import platform
import time

import torch

import difffea
from difffea.problem import default_spec, build_problem
from difffea.operators import global_residual, make_tangent_operator
from difffea.newton_krylov import newton_krylov_solve
from profile_matvec import timeit, sync, make_problem            # shared helpers (same folder)

HERE = os.path.dirname(os.path.abspath(__file__))


def tension_problem(n_elements, device):
    """Structured Q4 rectangle (2:1) in uniaxial tension (well conditioned for Jacobi-CG), on ``device``."""
    nx = int(round((n_elements * 2.0) ** 0.5))
    ny = max(1, n_elements // nx)
    spec = default_spec("rectangle", "quad4", 1.0)
    height = float(ny)
    spec.update(mesher="structured", geom_params=dict(width=float(nx), height=height), h=1.0,
                E=2600.0, nu=0.3, load_value=(0.1 * 2600.0 * height, 0.0), device=device)
    params = build_problem(spec)
    theta = torch.ones(params["conn"].shape[0], dtype=torch.float64, device=device)
    return params, theta


def matvec_sweep(sizes, device, reps):
    """Time R(u) and A(v) (three evaluations) for each requested mesh size."""
    rows = []
    for n in sizes:
        params, theta, u, v = make_problem(n, "quad4", device)
        n_el = params["conn"].shape[0]
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        row = dict(n_elements=n_el, n_dofs=int(u.numel()))
        row["residual_s"] = timeit(lambda: global_residual(u, theta, params), device, reps)
        for kind in ("jvp", "linearize", "qp"):
            if kind == "jvp" and n_el > 400_000 and device == "cpu":
                continue                                           # too slow to be worth waiting for
            A = make_tangent_operator(kind, u, theta, params)
            row[f"A_{kind}_s"] = timeit(lambda: A(v), device, reps)
        row["setup_qp_s"] = timeit(lambda: make_tangent_operator("qp", u, theta, params), device, 3)
        if str(device).startswith("cuda"):
            row["peak_gpu_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rows.append(row)
        print("  {n_elements:8d} el  R {r:8.2f} ms | A(v): ".format(n_elements=n_el, r=row["residual_s"] * 1e3)
              + "  ".join(f"{k} {row[f'A_{k}_s'] * 1e3:8.2f} ms" for k in ("jvp", "linearize", "qp")
                          if f"A_{k}_s" in row), flush=True)
    return rows


def solve_case(n_elements, device, tangent):
    """One complete nonlinear solve; returns (u on CPU, stats dict)."""
    params, theta = tension_problem(n_elements, device)
    tol = 1e-8 * float(torch.norm(params["f"]))
    sync(device)
    t0 = time.perf_counter()
    u, info = newton_krylov_solve(params, theta, 2, tol, verbose=False, preconditioner="jacobi", tangent=tangent)
    sync(device)
    secs = time.perf_counter() - t0
    R = global_residual(u, theta, params, 1.0)
    stats = dict(tangent=tangent, n_elements=int(params["conn"].shape[0]), seconds=secs,
                 converged=bool(info["converged"]), newton=int(info["total_newton"]), cg=int(info["total_cg"]),
                 final_residual=float(torch.norm(R)))
    return u.cpu(), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2_000, 10_000, 40_000, 160_000])
    ap.add_argument("--solve-elements", type=int, default=10_000)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()
    dev = args.device
    is_cuda = dev.startswith("cuda")
    if is_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available (Colab: Runtime > Change runtime type > T4 GPU).")

    out = dict(difffea=difffea.__version__, torch=torch.__version__, python=platform.python_version(),
               platform=platform.platform(), device=dev, threads=torch.get_num_threads(), dtype="float64")
    if is_cuda:
        out["gpu"] = torch.cuda.get_device_name(0)
        out["cuda"] = torch.version.cuda
    print(f"device={dev}  torch={torch.__version__}" + (f"  GPU={out['gpu']}" if is_cuda else
                                                       f"  threads={out['threads']}"))

    print("\n--- A(v) sweep (median times) ---")
    out["matvec_sweep"] = matvec_sweep(args.sizes, dev, args.reps)

    print(f"\n--- full solve, {args.solve_elements} Q4 elements, uniaxial tension, Jacobi-CG ---")
    out["solves"], sols = [], {}
    for tangent in ("jvp", "qp"):
        u, stats = solve_case(args.solve_elements, dev, tangent)
        sols[tangent] = u
        out["solves"].append(stats)
        print(f"  {tangent:10s} {stats['seconds']:8.2f} s  converged={stats['converged']}  Newton={stats['newton']}  "
              f"CG={stats['cg']}  |R|={stats['final_residual']:.2e}")
    out["max_diff_qp_vs_jvp"] = float((sols["qp"] - sols["jvp"]).abs().max() / sols["jvp"].abs().max())
    print(f"  max relative difference of the two solutions: {out['max_diff_qp_vs_jvp']:.1e}")

    if is_cuda:
        u_cpu, stats = solve_case(args.solve_elements, "cpu", "qp")
        out["solves"].append(dict(stats, note="CPU reference run on the same machine"))
        out["max_diff_gpu_vs_cpu"] = float((sols["qp"] - u_cpu).abs().max() / u_cpu.abs().max())
        print(f"  CPU qp reference: {stats['seconds']:.2f} s; GPU-vs-CPU solution difference "
              f"{out['max_diff_gpu_vs_cpu']:.1e}")
        out["peak_gpu_MB_total"] = torch.cuda.max_memory_allocated() / 2 ** 20

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"bench_{dev.split(':')[0]}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote", os.path.relpath(path, os.path.dirname(HERE)))
    print("\nJSON (copy this block when reporting results):")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
