"""
Comparison with ANALYTIC / classical reference solutions (small load, so the finite-strain solver must
reproduce linear elasticity).  Plane strain throughout: E' = E / (1 - nu^2) in the beam formulas.

1. Cantilever beam, tip shear force P.  Timoshenko beam theory:
        w_tip = P L^3 / (3 E' I)  +  (6/5) P L / (G A)            I = H^3/12, A = H (unit thickness)
   The clamp prevents warping and the load is uniform (not parabolic), so beam theory is only
   approximate: the quadratic elements converge to ~0.4 % below it for L/H = 10 (measured); the
   plane-STRESS formula would be 9 % off, so a 1 % tolerance still catches an E vs E' mix-up.
   Q4 with full integration suffers from shear locking (known, and visible in the second test).

2. Plate with a circular hole under uniaxial tension (Kirsch problem, finite width).  Peak stress at the
   hole edge: Heywood's finite-width formula  sigma_max = sigma_nom (2 + (1 - d/W)^3) / (1 - d/W),
   sigma_nom = applied stress on the gross section, d = hole diameter, W = plate width.  Tri6 approaches
   the peak from below and Q9 from above as the mesh is refined (measured), so refinement must tighten
   both towards the formula.  The formula itself is an engineering fit (about 1-2 %).
"""
import pytest
import torch

from difffea.problem import default_spec, build_problem
from difffea.newton_krylov import newton_krylov_solve
from difffea.gmsh_mesh import GMSH_AVAILABLE
from helpers import cauchy_at_node

E, NU = 1000.0, 0.3


def _tip_deflection(element_type, h, L=10.0, H=1.0, P=-0.002):
    """Average tip deflection of a cantilever (structured mesh) under a small total shear force P."""
    spec = default_spec("rectangle", element_type, h)
    spec.update(mesher="structured", geom_params=dict(width=L, height=H), E=E, nu=NU, load_value=(0.0, P))
    params = build_problem(spec)
    theta = torch.ones(params["conn"].shape[0], dtype=torch.float64)
    u, info = newton_krylov_solve(params, theta, 2, 1e-9 * abs(P), verbose=False, preconditioner="ilu")
    assert info["converged"]
    tip = params["nodes"][:, 0] > L - 1e-9
    w = u.reshape(-1, 2)[tip, 1].mean().item()
    E_plane = E / (1.0 - NU ** 2)
    G = E / (2.0 * (1.0 + NU))
    w_beam = P * L ** 3 / (3.0 * E_plane * H ** 3 / 12.0) + 1.2 * P * L / (G * H)
    return w / w_beam                                              # ratio FE / Timoshenko


@pytest.mark.parametrize("element_type", ["quad8", "quad9", "tri6"])
def test_cantilever_matches_timoshenko_beam(element_type):
    assert _tip_deflection(element_type, 0.25) == pytest.approx(1.0, abs=0.01)


def test_cantilever_q4_converges_from_below_by_refinement():
    coarse, fine = _tip_deflection("quad4", 0.25), _tip_deflection("quad4", 0.125)
    assert coarse < fine < 1.0                                     # locking: too stiff, improves with h
    assert fine > 0.97


def _hole_peak_stress(element_type, h, W=20.0, Hh=10.0, r=0.5):
    """(sigma_xx at the hole top node) / (Heywood prediction) for a plate with a hole, sigma_nom = 1."""
    spec = default_spec("plate_hole", element_type, h)
    spec.update(geom_params=dict(width=W, height=Hh, radius=r), E=E, nu=NU, load_value=(1.0 * Hh, 0.0))
    params = build_problem(spec)
    theta = torch.ones(params["conn"].shape[0], dtype=torch.float64)
    u, info = newton_krylov_solve(params, theta, 1, 1e-9, verbose=False, preconditioner="ilu")
    assert info["converged"]
    X = params["nodes"]
    top = torch.tensor([W / 2, Hh / 2 + r], dtype=torch.float64)            # top of the hole
    node = int(((X - top) ** 2).sum(1).argmin())
    assert float((X[node] - top).norm()) < 1e-9                             # the mesh has a node exactly there
    sxx = cauchy_at_node(u, params, theta, node)[0].item()
    d_over_W = 2 * r / Hh
    heywood = (2.0 + (1.0 - d_over_W) ** 3) / (1.0 - d_over_W)
    # far field: sigma_xx at the middle of the bottom edge should be close to the nominal stress 1
    bottom = torch.tensor([W / 2, 0.0], dtype=torch.float64)
    far = int(((X - bottom) ** 2).sum(1).argmin())
    assert cauchy_at_node(u, params, theta, far)[0].item() == pytest.approx(1.0, abs=0.04)
    return sxx / heywood


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="needs Gmsh")
def test_plate_with_hole_peak_stress_matches_heywood():
    tri_coarse, tri_fine = _hole_peak_stress("tri6", 0.5), _hole_peak_stress("tri6", 0.25)
    quad_coarse, quad_fine = _hole_peak_stress("quad9", 0.5), _hole_peak_stress("quad9", 0.25)
    # the two element families bracket the formula and tighten with refinement
    assert tri_coarse < tri_fine < 1.0 < quad_fine < quad_coarse
    assert tri_fine == pytest.approx(1.0, abs=0.03)
    assert quad_fine == pytest.approx(1.0, abs=0.03)
