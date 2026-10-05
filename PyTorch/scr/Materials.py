"""
Materials.py
============

The MATERIAL LIBRARY: hyperelastic strain-energy densities, each a pure function

        psi3(F3, p) -> scalar

of a 3x3 deformation gradient F3 and a vector p of material parameters.

Why 3x3?  The code solves plane-strain problems: the in-plane 2x2 deformation
gradient F is embedded as  F3 = [[F, 0], [0, 1]]  (no out-of-plane stretch).
Writing the energies in 3D has two advantages: the formulas are the textbook
ones (invariants I1, I2, I4 of the 3D right Cauchy-Green tensor), and the
out-of-plane stress sigma_zz needed for von Mises comes out of the SAME
derivative as everything else.

As everywhere in this code base, the energy is the ONLY thing a material
provides: stress  P = d psi / dF,  force and tangent follow by autodiff.

Adding a material = writing one function and adding one row to ``MATERIAL_LIBRARY``.

Parameter vector p
------------------
Each model lists the names of its parameters (``param_names``) and which of them
are STIFFNESS-LIKE (``scalable``).  The design field theta of the adjoint
(one number per element) multiplies exactly the scalable parameters.
"""

import torch
from typing import Callable, Dict


# =============================================================================
# Plane-strain embedding and 3D helpers
# =============================================================================
def embed_plane_strain(F: torch.Tensor) -> torch.Tensor:
    """
    In-plane gradient F (2,2) -> F3 (3,3) = [[F, 0], [0, 1]]   (plane strain: F33 = 1).

    Built with ``stack`` (not in-place assignment) so that every autodiff transform
    can differentiate through it.
    """
    zero = torch.zeros_like(F[0, 0])
    one = torch.ones_like(F[0, 0])
    return torch.stack([torch.stack([F[0, 0], F[0, 1], zero]),
                        torch.stack([F[1, 0], F[1, 1], zero]),
                        torch.stack([zero, zero, one])])


def _det3(F: torch.Tensor) -> torch.Tensor:
    """det F3 for the block-diagonal embedding: det(F2) * F33 (explicit, so second
    derivatives stay finite -- same reason as in Constitutive_Relations.py)."""
    return (F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]) * F[2, 2]


def _right_cauchy_green(F: torch.Tensor) -> torch.Tensor:
    """C = F^T F  (3,3)."""
    return torch.einsum("ki,kj->ij", F, F)


# =============================================================================
# The energies.  p is a 1D tensor ordered as listed in param_names.
# =============================================================================
def psi_neo_hookean(F, p):
    """
    Compressible neo-Hookean:   psi = mu/2 (I1 - 3 - 2 ln J) + lambda/2 (ln J)^2
    p = [mu, lmbda].   Reproduces ``Constitutive_Relations.neo_Hookean_energy``.
    """
    mu, lmbda = p[0], p[1]
    I1 = torch.sum(F * F)                  # tr(F^T F)
    logJ = torch.log(_det3(F))
    return 0.5 * mu * (I1 - 3.0 - 2.0 * logJ) + 0.5 * lmbda * logJ ** 2


def psi_mooney_rivlin(F, p):
    """
    Compressible Mooney-Rivlin with isochoric invariants and a logarithmic volumetric term:

        psi = c10 (I1b - 3) + c01 (I2b - 3) + kappa/2 (ln J)^2,
        I1b = J^(-2/3) I1,   I2b = J^(-4/3) I2,   I2 = (I1^2 - tr C^2)/2

    p = [c10, c01, kappa].   Initial shear modulus  mu0 = 2 (c10 + c01).
    """
    c10, c01, kappa = p[0], p[1], p[2]
    C = _right_cauchy_green(F)
    I1 = torch.trace(C)
    I2 = 0.5 * (I1 ** 2 - torch.trace(C @ C))
    J = _det3(F)
    I1b = J ** (-2.0 / 3.0) * I1
    I2b = J ** (-4.0 / 3.0) * I2
    return c10 * (I1b - 3.0) + c01 * (I2b - 3.0) + 0.5 * kappa * torch.log(J) ** 2


def psi_saint_venant(F, p):
    """
    Saint Venant-Kirchhoff:  psi = lambda/2 (tr E)^2 + mu tr(E^2),   E = (C - I)/2.
    p = [mu, lmbda].  Linear in the Green-Lagrange strain: simple, but NOT suited
    to large compression (it does not resist element inversion).
    """
    mu, lmbda = p[0], p[1]
    E = 0.5 * (_right_cauchy_green(F) - torch.eye(3, dtype=F.dtype, device=F.device))
    return 0.5 * lmbda * torch.trace(E) ** 2 + mu * torch.sum(E * E)


def psi_demiray(F, p):
    """
    Demiray / exponential (strain-stiffening) model for soft tissue:

        psi = mu/(2 gamma) (exp(gamma (I1 - 3)) - 1) - mu ln J + lambda/2 (ln J)^2

    p = [mu, gamma, lmbda].  For gamma -> 0 it reduces to neo-Hookean: the
    exponential makes the material stiffen sharply as it is stretched.
    """
    mu, gamma, lmbda = p[0], p[1], p[2]
    I1 = torch.sum(F * F)
    logJ = torch.log(_det3(F))
    return (mu / (2.0 * gamma)) * (torch.exp(gamma * (I1 - 3.0)) - 1.0) - mu * logJ + 0.5 * lmbda * logJ ** 2


def psi_fiber_reinforced(F, p):
    """
    FIBROUS material: neo-Hookean ground matrix reinforced by two symmetric families of
    collagen-like fibres (Holzapfel-Gasser-Ogden model, in-plane fibres):

        psi = psi_NH(F) + sum_{families i}  k1/(2 k2) ( exp( k2 <E_i>^2 ) - 1 ),
        E_i = d I1 + (1 - 3 d) I4_i - 1,       I4_i = a_i . C a_i

    * fibre directions  a_{1,2} = (cos beta, +-sin beta)  in the reference configuration
    * I4 is the squared fibre stretch;  <x> = max(x, 0) (Macaulay bracket): fibres carry
      load ONLY IN TENSION, they buckle and go slack in compression
    * d in [0, 1/3] is the fibre DISPERSION (0 = perfectly aligned, 1/3 = isotropic fibre
      distribution); with this definition E_i = 0 in the undeformed state

    p = [mu, lmbda, k1, k2, beta_deg, d, second_family]
        mu, lmbda : matrix (neo-Hookean) parameters
        k1        : fibre stiffness (stress-like);  k2 : fibre stiffening (dimensionless)
        beta_deg  : angle of the fibres with the x axis, in degrees
        d         : dispersion;  second_family : 1 = two families (+-beta), 0 = one family (+beta)
    """
    mu, lmbda, k1, k2, beta_deg, d, second = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

    psi_matrix = psi_neo_hookean(F, p[:2])

    C = _right_cauchy_green(F)
    I1 = torch.trace(C)
    beta = torch.deg2rad(beta_deg)
    zero = torch.zeros_like(beta)

    def fibre_energy(a):
        I4 = a @ C @ a                                   # squared stretch along the fibre
        E = d * I1 + (1.0 - 3.0 * d) * I4 - 1.0         # generalised structure invariant (0 at F = I)
        E_tension = torch.relu(E)                        # fibres only work in tension
        return (k1 / (2.0 * k2)) * (torch.exp(k2 * E_tension ** 2) - 1.0)

    a1 = torch.stack([torch.cos(beta), torch.sin(beta), zero])
    a2 = torch.stack([torch.cos(beta), -torch.sin(beta), zero])
    return psi_matrix + fibre_energy(a1) + second * fibre_energy(a2)


# =============================================================================
# The library
# =============================================================================
MATERIAL_LIBRARY: Dict[str, dict] = {
    "neo_hookean": dict(
        label="Neo-Hookean (isotropic rubber)", psi3=psi_neo_hookean,
        param_names=["mu", "lmbda"], defaults=[1000.0, 1500.0],
        scalable=["mu", "lmbda"], fibrous=False),
    "mooney_rivlin": dict(
        label="Mooney-Rivlin (isotropic rubber, two invariants)", psi3=psi_mooney_rivlin,
        param_names=["c10", "c01", "kappa"], defaults=[400.0, 100.0, 1500.0],
        scalable=["c10", "c01", "kappa"], fibrous=False),
    "saint_venant": dict(
        label="Saint Venant-Kirchhoff (linear in Green strain)", psi3=psi_saint_venant,
        param_names=["mu", "lmbda"], defaults=[1000.0, 1500.0],
        scalable=["mu", "lmbda"], fibrous=False),
    "demiray": dict(
        label="Demiray (exponential, strain-stiffening tissue)", psi3=psi_demiray,
        param_names=["mu", "gamma", "lmbda"], defaults=[500.0, 3.0, 1500.0],
        scalable=["mu", "lmbda"], fibrous=False),
    "fiber_reinforced": dict(
        label="Fibre-reinforced (HGO: matrix + two fibre families)", psi3=psi_fiber_reinforced,
        param_names=["mu", "lmbda", "k1", "k2", "beta_deg", "d", "second_family"],
        defaults=[400.0, 1200.0, 1500.0, 4.0, 30.0, 0.0, 1.0],
        scalable=["mu", "lmbda", "k1"], fibrous=True),
}


def material_names():
    return list(MATERIAL_LIBRARY.keys())


def get_material(name: str) -> dict:
    if name not in MATERIAL_LIBRARY:
        raise ValueError(f"Unknown material '{name}'. Choose one of {material_names()}.")
    return MATERIAL_LIBRARY[name]


def psi_2d(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Plane-strain energy  psi(F2, p)  of a library material: embeds the 2x2 gradient
    into 3x3 and calls the 3D energy.  This is the function the solver differentiates.
    """
    psi3 = get_material(name)["psi3"]
    return lambda F2, p: psi3(embed_plane_strain(F2), p)


def material_vector(name: str, values: dict = None, dtype=torch.float64):
    """
    Parameter vector p (n_p,) and the mask of scalable entries (n_p,) bool for a material.
    ``values`` overrides the defaults by name, e.g. {"k1": 2000.0}.
    """
    mat = get_material(name)
    vals = dict(zip(mat["param_names"], mat["defaults"]))
    if values:
        unknown = set(values) - set(vals)
        if unknown:
            raise ValueError(f"Unknown parameters {sorted(unknown)} for '{name}'; valid: {mat['param_names']}")
        vals.update({k: float(v) for k, v in values.items()})
    p = torch.tensor([vals[n] for n in mat["param_names"]], dtype=dtype)
    mask = torch.tensor([n in mat["scalable"] for n in mat["param_names"]], dtype=torch.bool)
    return p, mask


def material_of(params: dict):
    """
    Material data of a problem dictionary:  (name, p0 (n_p,), scalable mask (n_p,)).

    Problems built before the library existed carry only ``mu0``/``lmbda0``; they
    are interpreted as neo-Hookean.
    """
    if "material" in params:
        return params["material"], params["mat_p"], params["mat_scalable"]
    dev = params["nodes"].device
    p = torch.tensor([params["mu0"], params["lmbda0"]], dtype=params["nodes"].dtype, device=dev)
    return "neo_hookean", p, torch.ones(2, dtype=torch.bool, device=dev)


def element_material_parameters(theta: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Per-element parameter table (n_el, n_p): the nominal vector p0 with the
    SCALABLE entries multiplied by the element's design factor theta_e.
    """
    _, p0, scalable = material_of(params)
    factor = torch.where(scalable[None, :], theta[:, None], torch.ones_like(theta)[:, None])   # (n_el, n_p)
    return p0[None, :].to(theta.dtype) * factor
