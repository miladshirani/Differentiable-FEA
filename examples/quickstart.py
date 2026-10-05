"""
Quick start: solve a nonlinear elasticity problem in ~10 lines.

    python examples/quickstart.py
"""
import torch

from difffea.problem import default_spec, build_problem
from difffea.newton_krylov import newton_krylov_solve
from difffea.postprocess import compute_fields

spec = default_spec("plate_hole", element_type="quad9", h=0.25)   # plate with a hole, Q9 elements
params = build_problem(spec)                                      # mesh + BCs + cached geometry
theta = torch.ones(params["conn"].shape[0], dtype=torch.float64)  # nominal material everywhere

u, info = newton_krylov_solve(params, theta, n_load_steps=3, tol=1e-8 * 400, verbose=True)
fields = compute_fields(u, theta, params)

print(f"\nconverged: {info['converged']}   Newton its: {info['total_newton']}   CG its: {info['total_cg']}")
print(f"max |u| = {fields['u_mag'].max():.4f}    max von Mises = {fields['mises'].max():.1f}")
