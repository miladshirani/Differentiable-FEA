"""
constitutive_relations.py
=========================

The MATERIAL LAW.  A hyperelastic material is defined by a single scalar,
the strain-energy density psi(F).  Everything else follows by differentiation:

        P = d psi / d F          (first Piola-Kirchhoff stress, 2x2)

So we only ever type the energy.  To change the material (Mooney-Rivlin,
Ogden, ...) replace ``neo_Hookean_energy``; stress, force and tangent
update automatically.
"""

import torch
from torch.func import grad, vmap


def neo_Hookean_energy(F: torch.Tensor,
                       mu: float,
                       lmbda: float
                       ) -> torch.Tensor:
    """
    Compressible neo-Hookean strain-energy density (plane strain):

        psi(F) = mu/2 * (I1 - 2 - 2 ln J) + lambda/2 * (ln J)^2

    with  I1 = tr(F^T F)  and  J = det F.

    Parameters
    ----------
    F     : (2, 2) deformation gradient
    mu    : shear modulus (first Lame parameter)
    lmbda : second Lame parameter

    Returns
    -------
    psi : () scalar energy per unit reference area
    """
    # Closed-form 2x2 determinant.  We avoid torch.linalg.det here because its
    # second derivative is NaN when F has repeated singular values (e.g. F = I),
    # and the tangent needs exactly that second derivative.
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]

    I1 = torch.sum(F * F)      # tr(F^T F) = sum of squared entries
    logJ = torch.log(J)        # NaN if J <= 0: the signal of an inverted element

    return 0.5 * mu * (I1 - 2.0 - 2.0 * logJ) + 0.5 * lmbda * (logJ ** 2)


def Piola_Stress(F: torch.Tensor,
                 mu: float,
                 lmbda: float
                 ) -> torch.Tensor:
    """
    First Piola-Kirchhoff stress  P = d psi / d F   (2, 2).

    ``grad(..., argnums=0)`` differentiates the scalar energy with respect to
    its first argument F, giving a tensor with the same shape as F.
    """
    return grad(neo_Hookean_energy, argnums=0)(F, mu, lmbda)


# Batched versions over a leading axis of F (mu, lmbda shared):
#   F : (n, 2, 2)  ->  P : (n, 2, 2)   and   psi : (n,)
Piola_Batched = vmap(Piola_Stress, in_dims=(0, None, None))
neo_Hookean_Batched = vmap(neo_Hookean_energy, in_dims=(0, None, None))
