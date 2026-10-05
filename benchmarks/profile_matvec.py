"""
profile_matvec.py
=================

WHERE DOES THE TIME GO?  (roadmap step 2: profile before optimising)

Measures, for one mesh, on one device (CPU or CUDA):

  * wall time of the building blocks of a Krylov iteration: residual R(u), tangent product A(v) (jvp),
    transpose product A^T(v) (vjp), Jacobi diagonal;
  * the same tangent product through ``torch.func.linearize`` (primal evaluated once, then only the
    tangent is applied) -- the obvious candidate optimisation;
  * an ASSEMBLED-matrix baseline: time to assemble the sparse tangent, time of one SciPy CSR
    matrix-vector product, and the memory of the matrix (only meaningful on the CPU);
  * a machine-bandwidth reference (large vector triad) and the MINIMUM memory traffic of one A(v), so the
    achieved fraction of the memory bandwidth can be stated instead of guessed;
  * a ``torch.profiler`` table of the operators that dominate A(v).

Run (click Run in VS Code or `python benchmarks/profile_matvec.py [--elements 40000] [--device cpu|cuda]`).
Results are printed and written to benchmarks/results/profile_<device>.json (no machine names or paths).
"""
import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import torch
from torch.func import linearize

import difffea
from difffea.problem import default_spec, build_problem
from difffea.operators import (global_residual, make_A_operator, make_A_operator_qp, make_AT_operator,
                               compute_jacobi_diagonal, assemble_tangent_sparse)

HERE = os.path.dirname(os.path.abspath(__file__))


def sync(device):
    """Wait for queued GPU work (no-op on the CPU) so that timings are real."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timeit(fn, device, reps=10, warmup=2):
    """Median wall time of ``fn()`` in seconds over ``reps`` runs after ``warmup`` runs."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def make_problem(n_elements, element_type, device):
    """Structured rectangle with about ``n_elements`` elements (two triangles per cell for tri elements)."""
    cells = n_elements // (2 if element_type.startswith("tri") else 1)
    nx = int(round((cells * 2.0) ** 0.5))                  # 2:1 aspect rectangle
    ny = max(1, cells // nx)
    spec = default_spec("rectangle", element_type, 1.0)
    spec.update(mesher="structured", geom_params=dict(width=float(nx), height=float(ny)), h=1.0, device=device)
    params = build_problem(spec)
    n_el = params["conn"].shape[0]
    theta = torch.ones(n_el, dtype=torch.float64, device=device)
    gen = torch.Generator().manual_seed(0)
    u = (1e-3 * torch.randn(2 * params["nodes"].shape[0], dtype=torch.float64, generator=gen)).to(device)
    v = torch.randn(2 * params["nodes"].shape[0], dtype=torch.float64, generator=gen).to(device)
    return params, theta, u, v


def stream_triad_bandwidth(device, n=2 ** 25, reps=10):
    """Bytes/s of a = b + s*c on float64 vectors of n entries (3 x 8 bytes of traffic per entry)."""
    b = torch.rand(n, dtype=torch.float64, device=device)
    c = torch.rand(n, dtype=torch.float64, device=device)
    a = torch.empty_like(b)
    t = timeit(lambda: torch.add(b, c, alpha=1.5, out=a), device, reps=reps)
    return 3 * 8 * n / t


def min_traffic_bytes(params):
    """
    Lower bound on the memory traffic of ONE tangent product (float64 / int64 indices), counting each array
    that MUST be read or written once: cached geometry, connectivity, material table, v and u gathered per
    element, the element forces scattered, plus the global vectors.
    """
    n_el, nen = params["conn"].shape
    n_dof = 2 * params["nodes"].shape[0]
    geom = params["dN_dX"].numel() * 8 + params["w_detJ"].numel() * 8
    conn = params["conn"].numel() * 8
    gathered = 2 * n_el * nen * 2 * 8                       # u_e and v_e
    scattered = n_el * nen * 2 * 8 + n_dof * 8              # element forces + global result
    globals_ = 4 * n_dof * 8                                # u, v, mask, result
    return geom + conn + gathered + scattered + globals_


def profile_table(fn, device, top=10):
    """Run ``fn`` under torch.profiler and return the ``top`` operators by self time as dict rows."""
    from torch.profiler import profile, ProfilerActivity
    on_gpu = str(device).startswith("cuda")
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_gpu else [])
    fn(); sync(device)
    with profile(activities=acts) as prof:
        for _ in range(3):
            fn()
        sync(device)
    events = list(prof.key_averages())
    # PyTorch renamed the per-event device time: self_cuda_time_total (older) -> self_device_time_total (newer)
    candidates = ["self_device_time_total", "self_cuda_time_total"] if on_gpu else ["self_cpu_time_total"]
    key = next((k for k in candidates if events and hasattr(events[0], k)), None)
    if key is None:
        raise RuntimeError(f"profiler events have none of the attributes {candidates}")
    events = [e for e in events if getattr(e, key) > 0]
    total = sum(getattr(e, key) for e in events)
    events.sort(key=lambda e: getattr(e, key), reverse=True)
    return [dict(op=e.key, calls=e.count // 3, percent=100.0 * getattr(e, key) / total,
                 ms_per_call_total=getattr(e, key) / 3e3) for e in events[:top]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", type=int, default=40000)
    ap.add_argument("--element-type", default="quad4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    dev = args.device

    params, theta, u, v = make_problem(args.elements, args.element_type, dev)
    n_el, n_dof = params["conn"].shape[0], 2 * params["nodes"].shape[0]
    print(f"{args.element_type}: {n_el} elements, {n_dof} DOFs, device={dev}, torch={torch.__version__}, "
          f"threads={torch.get_num_threads()}")

    A = make_A_operator(u, theta, params)
    AT = make_AT_operator(u, theta, params)
    _, jvp_fn = linearize(lambda w: global_residual(w, theta, params), u)       # primal evaluated once
    A_qp = make_A_operator_qp(u, theta, params)                                 # tangent stored at Gauss points
    m = params["m"]

    t = {}
    t["residual R(u)"] = timeit(lambda: global_residual(u, theta, params), dev, args.reps)
    t["A(v) = jvp (current)"] = timeit(lambda: A(v), dev, args.reps)
    t["A(v) via linearize (primal cached)"] = timeit(lambda: m * jvp_fn(m * v) + (1.0 - m) * v, dev, args.reps)
    t["A(v) stored Gauss-point tangent (qp)"] = timeit(lambda: A_qp(v), dev, args.reps)
    # one-off cost per Newton iteration (paid once, then amortised over all CG iterations)
    t["setup: linearize"] = timeit(lambda: linearize(lambda w: global_residual(w, theta, params), u), dev, 3)
    t["setup: qp (moduli at Gauss points)"] = timeit(lambda: make_A_operator_qp(u, theta, params), dev, 3)
    t["A^T(v) = vjp (pullback only)"] = timeit(lambda: AT(v), dev, args.reps)
    t["Jacobi diagonal (once per Newton step)"] = timeit(lambda: compute_jacobi_diagonal(u, theta, params), dev,
                                                         max(3, args.reps // 3))
    err = float((A(v) - (m * jvp_fn(m * v) + (1.0 - m) * v)).abs().max() / A(v).abs().max())
    err_qp = float((A(v) - A_qp(v)).abs().max() / A(v).abs().max())

    out = dict(difffea=difffea.__version__, torch=torch.__version__, platform=platform.platform(),
               python=platform.python_version(), device=dev, threads=torch.get_num_threads(),
               element_type=args.element_type, n_elements=n_el, n_dofs=n_dof,
               seconds={k: float(x) for k, x in t.items()}, linearize_vs_jvp_rel_diff=err, qp_vs_jvp_rel_diff=err_qp)
    if str(dev).startswith("cuda"):
        out["gpu"] = torch.cuda.get_device_name(0)
        out["peak_memory_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20

    # ---- assembled baseline (CPU only: SciPy) --------------------------------------------------
    if dev == "cpu":
        import scipy.sparse as sp
        t0 = time.perf_counter()
        K = assemble_tangent_sparse(u, theta, params).tocsr()
        t_asm = time.perf_counter() - t0
        vn = v.numpy()
        t_spmv = timeit(lambda: K @ vn, dev, args.reps)
        out["assembled"] = dict(assembly_s=t_asm, spmv_s=float(t_spmv), nnz=int(K.nnz),
                                matrix_MB=(K.data.nbytes + K.indices.nbytes + K.indptr.nbytes) / 2 ** 20,
                                breakeven_matvecs=float(t_asm / max(t["A(v) = jvp (current)"] - t_spmv, 1e-12)))

    # ---- bandwidth reference ----------------------------------------------------------------------
    bw = stream_triad_bandwidth(dev)
    traffic = min_traffic_bytes(params)
    t_ideal = traffic / bw
    out["bandwidth_GBps_triad"] = bw / 1e9
    out["min_traffic_MB_per_matvec"] = traffic / 2 ** 20
    out["ideal_matvec_s_at_triad_bandwidth"] = t_ideal
    out["fraction_of_bandwidth_current"] = t_ideal / t["A(v) = jvp (current)"]
    out["fraction_of_bandwidth_linearize"] = t_ideal / t["A(v) via linearize (primal cached)"]
    out["fraction_of_bandwidth_qp"] = t_ideal / t["A(v) stored Gauss-point tangent (qp)"]
    out["qp_storage_MB"] = n_el * params["dN_dX"].shape[1] * 16 * 8 / 2 ** 20

    # ---- print ----------------------------------------------------------------------------------
    print("\n--- wall time (median of runs) ---")
    for k, x in t.items():
        print(f"{k:42s} {x * 1e3:10.2f} ms")
    print(f"max rel. difference of the result vs jvp: linearize {err:.1e}, qp {err_qp:.1e}")
    if "assembled" in out:
        a = out["assembled"]
        print(f"assembled: assembly {a['assembly_s'] * 1e3:.0f} ms, CSR matvec {a['spmv_s'] * 1e3:.2f} ms, "
              f"{a['matrix_MB']:.0f} MB, break-even after {a['breakeven_matvecs']:.0f} matvecs")
    print(f"\nbandwidth (triad) {out['bandwidth_GBps_triad']:.1f} GB/s; minimum traffic of one A(v) "
          f"{out['min_traffic_MB_per_matvec']:.1f} MB -> ideal {t_ideal * 1e3:.3f} ms; "
          f"fraction of that reached: jvp {100 * out['fraction_of_bandwidth_current']:.2f} %, "
          f"linearize {100 * out['fraction_of_bandwidth_linearize']:.2f} %, "
          f"qp {100 * out['fraction_of_bandwidth_qp']:.2f} % (stores {out['qp_storage_MB']:.0f} MB)")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"profile_{dev.split(':')[0]}_{args.element_type}.json")

    def save():
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)

    save()                                  # the timings are safe on disk before the profiler (which can fail) runs

    # ---- operator-level profile (last: it depends on the PyTorch version) -------------------------
    print("\n--- torch.profiler, top operators of A(v) (self time) ---")
    try:
        out["profile_A_v_top_ops"] = profile_table(lambda: A(v), dev)
        for r in out["profile_A_v_top_ops"]:
            print(f"{r['percent']:6.1f} %  {r['ms_per_call_total']:9.2f} ms  x{r['calls']:<4d} {r['op']}")
    except Exception as exc:                # noqa: BLE001 -- report and keep the measurements
        out["profile_error"] = f"{type(exc).__name__}: {exc}"
        print("profiler failed (timings above are still valid):", out["profile_error"])
    save()
    print("\nwrote", os.path.relpath(path, os.path.dirname(HERE)))
    print("\nJSON (copy this block when reporting results):")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
