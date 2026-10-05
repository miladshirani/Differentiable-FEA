"""
dolfinx_reference.py
====================

REFERENCE SOLVE with FEniCSx/dolfinx for the validation of difffea (step 1 of the roadmap).

This file is run by a Python that has dolfinx installed (NOT by the difffea environment, and it never
imports torch or difffea):

    <dolfinx-python> benchmarks/dolfinx_reference.py  case_in.npz  case_out.npz

It reads a problem exported by ``validate_against_dolfinx.py`` -- the SAME mesh (nodes, connectivity),
the SAME neo-Hookean energy, the SAME Gauss rule and the SAME loads -- and solves it with dolfinx's own
automatic differentiation (UFL) and an assembled sparse tangent.  Because every ingredient is identical,
the two codes must agree up to the Newton tolerance; any larger difference is a bug in one of them.

Energy (plane strain, per unit reference area):
        psi(F) = mu/2 (tr(F^T F) - 2 - 2 ln J) + lambda/2 (ln J)^2
Loads: dead traction t = F_total / L on the right edge (x = x_max), clamp on the left edge (x = x_min),
applied in ``n_steps`` equal increments (each Newton solve starts from the previous converged state).

Output npz: ``u`` (N, 2) displacement at the exported node coordinates, ``energy`` (strain energy),
``reaction`` (2,) = sum of the INTERNAL force over the clamped nodes (same definition as difffea's
``support_reactions``), ``newton`` (total Newton iterations).
"""
import sys

import numpy as np
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from mpi4py import MPI

import basix.ufl
import dolfinx
import ufl
from dolfinx import fem, mesh as dmesh

CELL = {"tri": ("triangle", dmesh.CellType.triangle),
        "quad": ("quadrilateral", dmesh.CellType.quadrilateral)}


def make_mesh(nodes, conn, family, order):
    """
    dolfinx mesh with the SAME nodes as difffea.  Our connectivity follows the Gmsh node numbering, so
    dolfinx's own helper ``perm_gmsh`` converts it to dolfinx's ordering.  The geometry degree equals the
    element order, so the curved boundary of the hole is represented exactly as in difffea.
    """
    name, ctype = CELL[family]
    perm = dolfinx.cpp.io.perm_gmsh(ctype, conn.shape[1])
    cells = np.ascontiguousarray(conn[:, perm]).astype(np.int64)
    coord_el = basix.ufl.element("Lagrange", name, order, shape=(2,))
    return dmesh.create_mesh(MPI.COMM_SELF, cells, coord_el, np.ascontiguousarray(nodes, dtype=np.float64))


def solve(case):
    """
    Newton solve of the clamped / dead-load problem described by the dict ``case``.
    Returns (u Function, V, energy, internal-force vector, flat clamped dof indices, Newton iterations).
    """
    nodes, conn = case["nodes"], case["conn"]
    family, order = case["family"], case["order"]
    mu, lmbda, force, n_steps = case["mu"], case["lmbda"], case["force"], case["n_steps"]

    msh = make_mesh(nodes, conn, family, order)
    name, _ = CELL[family]
    V = fem.functionspace(msh, basix.ufl.element("Lagrange", name, order, shape=(2,)))
    bs = V.dofmap.index_map_bs                                             # = 2 components per node

    xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
    length = nodes[:, 1].max() - nodes[:, 1].min()                         # length of the loaded edge
    atol = 1e-9 * max(xmax - xmin, length)

    # --- Dirichlet: clamp the left edge (both components) ----------------------------------------
    node_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], xmin, atol=atol))
    bc = fem.dirichletbc(fem.Function(V), node_dofs)                       # value 0
    clamped = np.array([d * bs + c for d in node_dofs for c in range(bs)])  # flat dof ids of the clamp

    # --- Neumann: dead traction on the right edge ------------------------------------------------
    fdim = msh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], xmax, atol=atol))
    tags = dmesh.meshtags(msh, fdim, np.sort(facets), np.full(len(facets), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags, subdomain_id=1)

    # --- the SAME Gauss rule as difffea (reference coordinates of the dolfinx cell) -------------
    pts, wts = np.asarray(case["GP"], dtype=np.float64), np.asarray(case["GW"], dtype=np.float64)
    if family == "quad":                   # difffea uses [-1,1]^2, dolfinx [0,1]^2
        pts, wts = 0.5 * (pts + 1.0), wts / 4.0
    dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_rule": "custom",
                                                 "quadrature_points": pts,
                                                 "quadrature_weights": wts})

    # --- energy, residual, tangent (UFL automatic differentiation) -----------------------------
    u = fem.Function(V)
    v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    lam = fem.Constant(msh, 0.0)                                           # load factor in (0, 1]
    traction = ufl.as_vector((float(force[0]) / length, float(force[1]) / length))
    F = ufl.Identity(2) + ufl.grad(u)
    J = ufl.det(F)
    psi = 0.5 * mu * (ufl.inner(F, F) - 2.0 - 2.0 * ufl.ln(J)) + 0.5 * lmbda * ufl.ln(J) ** 2
    Pi = psi * dx - lam * ufl.dot(traction, u) * ds
    R_form = fem.form(ufl.derivative(Pi, u, v))
    K_form = fem.form(ufl.derivative(ufl.derivative(Pi, u, v), u, du))
    Fint_form = fem.form(ufl.derivative(psi * dx, u, v))
    E_form = fem.form(psi * dx)

    def assemble():
        """Residual (clamped rows zeroed) and tangent (clamped rows/cols replaced by the identity)."""
        r = fem.assemble_vector(R_form).array.copy()
        r[clamped] = 0.0
        return r, fem.assemble_matrix(K_form, bcs=[bc]).to_scipy().tocsc()

    ftol = 1e-11 * max(1.0, float(np.linalg.norm(force)))
    newton_total = 0
    for step in range(1, n_steps + 1):
        lam.value = step / n_steps
        r, K = assemble()
        for _ in range(40):
            if np.linalg.norm(r) <= ftol:
                break
            delta = spla.spsolve(K, -r)
            alpha, rn0, u0 = 1.0, np.linalg.norm(r), u.x.array.copy()
            while True:                                         # backtracking on the residual norm
                u.x.array[:] = u0 + alpha * delta
                r_new, K_new = assemble()
                rn = np.linalg.norm(r_new)
                if (np.isfinite(rn) and rn < (1.0 - 1e-4 * alpha) * rn0) or alpha < 1e-10:
                    break
                alpha *= 0.5
            r, K = r_new, K_new
            newton_total += 1
        else:
            raise RuntimeError(f"Newton did not converge in load step {step}")

    energy = fem.assemble_scalar(E_form)
    f_int = fem.assemble_vector(Fint_form).array.copy()
    return u, V, float(energy), f_int, clamped, newton_total


def main(path_in, path_out):
    d = np.load(path_in, allow_pickle=False)
    case = dict(nodes=d["nodes"], conn=d["conn"], family=str(d["family"]), order=int(d["order"]),
                mu=float(d["mu"]), lmbda=float(d["lmbda"]), force=d["force"], GP=d["GP"], GW=d["GW"],
                n_steps=int(d["n_steps"]))
    u, V, energy, f_int, clamped, newton = solve(case)

    # displacement at the exported node coordinates: match dolfinx dof coordinates to our node list
    bs = V.dofmap.index_map_bs
    dist, idx = cKDTree(V.tabulate_dof_coordinates()[:, :2]).query(d["nodes"])
    assert dist.max() < 1e-9 * max(1.0, np.abs(d["nodes"]).max()), "node matching failed"
    uu = u.x.array.reshape(-1, bs)[idx]
    reaction = f_int[clamped].reshape(-1, bs).sum(axis=0)
    np.savez(path_out, u=uu, energy=energy, reaction=reaction, newton=newton, version=dolfinx.__version__)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
