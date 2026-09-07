import torch
from torch.func import grad, vmap


def neo_Hookean_energy( F: torch.Tensor, 
                        mu: float, 
                        lmbda: float
                     ) -> torch.Tensor:
    
    """
    Computes the compressible Neo-Hookean strain energy density psi(F).

    Parameters
    ----------
    F : torch.Tensor
        Deformation gradient tensor of shape (2, 2).
    mu : float
        Lame parameter (shear modulus).
    lmbda : float
        Lame parameter.

    Returns
    -------
    torch.Tensor
        Scalar strain energy density value of shape ().
    """
    J  = torch.linalg.det(F)
    C  = F.T @ F
    I1 = torch.trace(C)
    logJ = torch.log(J)

    return 0.5 * mu * (I1 - 2.0 - 2.0 * logJ) + 0.5 * lmbda * (logJ**2)




def Piola_Stress(F: torch.Tensor, 
                 mu: float, 
                 lmbda: float
                 ) -> torch.Tensor:
    
    Piola = grad(neo_Hookean_energy, argnim = 0)(F, mu, lmbda)
    return Piola


Piola_Batched       = vmap(Piola_Stress,       in_dims = (0, None, None))
neo_Hookean_Batched = vmap(neo_Hookean_energy, in_dims = (0, None, None))


    

