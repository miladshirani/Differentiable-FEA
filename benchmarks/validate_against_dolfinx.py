"""
validate_against_dolfinx.py
===========================

VALIDATION (roadmap step 1): solve the SAME discrete problem with difffea and with FEniCSx/dolfinx and
compare.  "Same" means: identical mesh (nodes and connectivity are exported from difffea), identical
neo-Hookean energy, identical Gauss rule, identical dead load and clamp, identical load steps.  The
two codes then solve the same nonlinear algebraic system, so the difference measures errors in the
implementation (element, kinematics, residual, tangent, loads), not discretisation error.

How to run (click Run on this file in VS Code, or `python benchmarks/validate_against_dolfinx.py`):
the dolfinx part needs a Python that has dolfinx installed.  Tell this script where it is with the
environment variable  DIFFFEA_DOLFINX_PYTHON  (path to that python executable).  Example:

    DIFFFEA_DOLFINX_PYTHON=/path/to/env/bin/python  python benchmarks/validate_against_dolfinx.py

Output: a table on screen and benchmarks/results/validate_dolfinx.json (versions, per-case errors).
The script never modifies the dolfinx environment.
"""
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

import difffea
from difffea.problem import default_spec, build_problem, lame_parameters
from difffea.newton_krylov import newton_krylov_solve
from difffea.postprocess import compute_fields, support_reactions
from difffea.elements import get_element

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "dolfinx_reference.py")
RESULTS = os.path.join(HERE, "results", "validate_dolfinx.json")

N_STEPS = 3
# (geometry, element type, element size).  dolfinx has no 8-node serendipity geometry, so Q8 is not compared.
CASES = [("plate_hole", "tri3", 0.25), ("plate_hole", "tri6", 0.4), ("plate_hole", "quad4", 0.25),
         ("plate_hole", "quad9", 0.4), ("rectangle", "quad4", 0.25), ("rectangle", "tri6", 0.5)]


def solve_difffea(geometry, element_type, h):
    """Solve one case with difffea; returns (spec, params, u (N,2), fields, reaction (2,), seconds)."""
    spec = default_spec(geometry, element_type, h)
    if geometry == "rectangle":
        spec["mesher"] = "structured"                         # the structured mesher needs no Gmsh
    params = build_problem(spec)
    theta = torch.ones(params["conn"].shape[0], dtype=torch.float64)
    tol = 1e-11 * float(torch.norm(params["f"]))
    t0 = time.time()
    u, info = newton_krylov_solve(params, theta, N_STEPS, tol, verbose=False, preconditioner="ilu")
    secs = time.time() - t0
    assert info["converged"], f"difffea did not converge for {geometry}/{element_type}"
    fields = compute_fields(u, theta, params)
    return spec, params, u.reshape(-1, 2), fields, support_reactions(u, theta, params), secs


def run_dolfinx(python, spec, params, workdir):
    """Export the case, run the dolfinx reference in a child process, return its results."""
    el = get_element(spec["element_type"])
    mu, lmbda = lame_parameters(spec["E"], spec["nu"])
    fin, fout = os.path.join(workdir, "case_in.npz"), os.path.join(workdir, "case_out.npz")
    np.savez(fin, nodes=params["nodes"].numpy(), conn=params["conn"].numpy(), family=el["family"],
             order=el["order"], mu=mu, lmbda=lmbda, force=np.array(spec["load_value"], dtype=float),
             GP=params["GP"].numpy(), GW=params["GW"].numpy(), n_steps=N_STEPS)
    proc = subprocess.run([python, REFERENCE, fin, fout], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("dolfinx reference failed:\n" + proc.stderr[-3000:])
    return dict(np.load(fout))


def main():
    python = os.environ.get("DIFFFEA_DOLFINX_PYTHON")
    if not python or not os.path.exists(python):
        sys.exit("Set DIFFFEA_DOLFINX_PYTHON to the python executable of an environment that has dolfinx.")

    rows, version = [], None
    print(f"{'case':22s} {'nodes':>6s} {'max|u|':>9s} {'max|du|/max|u|':>15s} {'energy rel.diff':>16s} "
          f"{'reaction rel.diff':>18s}")
    with tempfile.TemporaryDirectory() as workdir:
        for geometry, element_type, h in CASES:
            spec, params, u, fields, reaction, secs = solve_difffea(geometry, element_type, h)
            ref = run_dolfinx(python, spec, params, workdir)
            version = str(ref["version"])
            u_ref = ref["u"]
            err_u = float(np.abs(u.numpy() - u_ref).max() / np.abs(u_ref).max())
            err_e = abs(float(fields["strain_energy"]) - float(ref["energy"])) / abs(float(ref["energy"]))
            r_ref = ref["reaction"]
            err_r = float(np.linalg.norm(reaction.numpy() - r_ref) / np.linalg.norm(r_ref))
            name = f"{geometry}/{element_type}"
            print(f"{name:22s} {u.shape[0]:6d} {np.abs(u_ref).max():9.4f} {err_u:15.2e} {err_e:16.2e} {err_r:18.2e}")
            rows.append(dict(case=name, geometry=geometry, element_type=element_type, h=h,
                             n_nodes=int(u.shape[0]), n_elements=int(params["conn"].shape[0]),
                             max_u=float(np.abs(u_ref).max()), rel_err_u=err_u, rel_err_energy=err_e,
                             rel_err_reaction=err_r, dolfinx_newton_iterations=int(ref["newton"])))

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as fh:
        json.dump(dict(difffea=difffea.__version__, torch=torch.__version__, dolfinx=version,
                       platform=platform.platform(), python=platform.python_version(), n_load_steps=N_STEPS,
                       cases=rows), fh, indent=2)
    print("\nwrote", os.path.relpath(RESULTS))


if __name__ == "__main__":
    main()
