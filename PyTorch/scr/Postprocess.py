"""
Postprocess.py
==============

From the displacement vector u to PHYSICAL FIELDS: deformation gradient,
Cauchy stress, von Mises stress, strain energy, nodal contours and support
reactions.

The pattern is the same as in the residual: write the computation for ONE
Gauss point, ``vmap`` it over the Gauss points of an element, then over all
elements, then average to nodes with one scatter-add.
"""

import torch
from torch.func import vmap, grad

from scr.Kinematics import (Coordinate_Interpolation, Displacement_Interpolation,
                            Global_Deformation_Gradient)
from scr.Materials import (embed_plane_strain, get_material, material_of,
                           element_material_parameters)
from scr.Operators import global_residual


def gauss_point_state(u_fn, X_fn, p, psi3, xi, eta):
    """
    Kinematic and stress state at ONE Gauss point, for ANY library material.

    Returns a vector (8,):  [det J_mesh, J, sigma_xx, sigma_yy, sigma_xy, sigma_zz, psi, 0]
    (the last slot is padding so that the vector length is a fixed 8).

    The 3D (plane-strain) energy gives the full first Piola-Kirchhoff stress by one
    derivative, and the Cauchy (true) stress follows:

            P3 = d psi3 / d F3,        sigma = P3 F3^T / J     (3 x 3)

    so the out-of-plane stress sigma_zz, which von Mises needs, is a by-product.
    """
    F2, det_mesh, _ = Global_Deformation_Gradient(u_fn, X_fn, xi, eta)    # (2,2), ()
    F3 = embed_plane_strain(F2)                                           # (3,3)
    P3 = grad(psi3, argnums=0)(F3, p)                                     # (3,3)
    J = F2[0, 0] * F2[1, 1] - F2[0, 1] * F2[1, 0]                         # area change

    sigma = torch.einsum("ij,kj->ik", P3, F3) / J                         # P F^T / J
    psi = psi3(F3, p)                                                     # energy density

    zero = torch.zeros_like(J)
    return torch.stack([det_mesh, J, sigma[0, 0], sigma[1, 1], sigma[0, 1], sigma[2, 2], psi, zero])


def element_state(ue, Xe, p, shape_fn, GP, GW, psi3):
    """
    Quadrature-weighted AVERAGE of ``gauss_point_state`` over one element.

    Returns (8,) with the same layout (the first entry is the element area).
    """
    X_fn = Coordinate_Interpolation(shape_fn, Xe)
    u_fn = Displacement_Interpolation(shape_fn, ue)

    # vmap over Gauss points -> (n_q, 8)
    states = vmap(gauss_point_state, in_dims=(None, None, None, None, 0, 0))(
        u_fn, X_fn, p, psi3, GP[:, 0], GP[:, 1])

    weights = GW * states[:, 0]                      # w_q * det J_q : physical weights
    area = weights.sum()
    mean = torch.einsum("q,qk->k", weights, states) / area
    # keep the TOTAL area in slot 0 (instead of its average)
    return torch.cat([area[None], mean[1:]])


def compute_fields(u, theta, params) -> dict:
    """
    All post-processing fields of a converged solution.

    Returns a dict of tensors
        per element (n_el,):  "J", "sxx", "syy", "sxy", "mises", "energy_density"
        per node    (N,):     "ux", "uy", "u_mag" (also available as "ux_n", ...) and the
                              nodal averages of the element fields, named with a "_n"
                              suffix (e.g. "mises_n")
        scalars:              "strain_energy", "min_J"
    """
    conn, nodes = params["conn"], params["nodes"]
    N = nodes.shape[0]
    u_nodal = u.reshape(N, 2)

    # gather element data (same pattern as global_residual)
    ue, Xe = u_nodal[conn], nodes[conn]                    # (n_el, nen, 2)
    element_p = element_material_parameters(theta, params)          # (n_el, n_p)
    psi3 = get_material(material_of(params)[0])["psi3"]

    # vmap over elements -> (n_el, 8)
    S = vmap(element_state, in_dims=(0, 0, 0, None, None, None, None))(
        ue, Xe, element_p, params["shape_fn"], params["GP"], params["GW"], psi3)

    area, J, sxx, syy, sxy, szz, psi = (S[:, k] for k in range(7))

    # von Mises equivalent stress in 3D with the plane-strain out-of-plane stress szz
    mises = torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3.0 * sxy ** 2)

    out = {"J": J, "sxx": sxx, "syy": syy, "sxy": sxy, "mises": mises, "energy_density": psi,
           "ux": u_nodal[:, 0], "uy": u_nodal[:, 1], "u_mag": torch.linalg.norm(u_nodal, dim=1),
           "strain_energy": torch.sum(psi * area), "min_J": J.min()}

    # displacement components are already nodal; give them the same "_n" names
    # as the averaged element fields so that a GUI can treat all fields alike
    for name in ("ux", "uy", "u_mag"):
        out[name + "_n"] = out[name]

    # nodal averages: scatter-add every element value to its nodes, divide by the
    # number of elements that touch the node
    count = torch.zeros(N, dtype=u.dtype, device=u.device).index_add_(
        0, conn.reshape(-1), torch.ones(conn.numel(), dtype=u.dtype, device=u.device))
    for name in ("J", "sxx", "syy", "sxy", "mises", "energy_density"):
        per_node = torch.zeros(N, dtype=u.dtype, device=u.device)
        per_node.index_add_(0, conn.reshape(-1),
                            out[name][:, None].expand(-1, conn.shape[1]).reshape(-1))
        out[name + "_n"] = per_node / count.clamp(min=1.0)
    return out


def support_reactions(u, theta, params) -> torch.Tensor:
    """
    Reaction forces at the constrained DOFs, shape (2,) = total (Rx, Ry).

    ``global_residual`` zeroes the constrained rows (the mask), so we evaluate
    the INTERNAL force with an all-ones mask and no external load.  The internal
    force R_int is the external force each node needs to be in equilibrium; at a
    constrained DOF that external force IS the support reaction.  Global
    equilibrium therefore reads   sum(reactions) + sum(applied loads) = 0.
    """
    p = dict(params)
    p["m"] = torch.ones_like(params["m"])
    p["f"] = torch.zeros_like(params["f"])
    R_int = global_residual(u, theta, p, 1.0).reshape(-1, 2)         # (N, 2)
    fixed = params["fixed"].reshape(-1, 2)                           # (N, 2) bool
    # sum the internal force over the constrained DOFs only
    return torch.sum(torch.where(fixed, R_int, torch.zeros_like(R_int)), dim=0)
