import torch
from torch.func import grad, vmap
from typing import Tuple
from scipy.special import roots_legendre

def Gauss_Quadratures_1D(n_points_per_dim: int,
                         dtype: torch.dtype = torch.float64, 
                         device: str = "cpu"
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Computes 1D Gauss-Legendre quadrature points and weights on [-1, 1].

    Parameters
    ----------
    n_points_per_dim : int
        Number of Gauss integration points along 1D.
    dtype : torch.dtype
        Target PyTorch floating-point type (default: float64).
    device : str
        Target compute device.

    Returns
    -------
    Gauss_Points_1D : torch.Tensor
        Integration coordinates of shape (n_points_per_dim,).
    Gauss_Weights_1D : torch.Tensor
        Integration weights of shape (n_points_per_dim,).
    """
    pts_1d, wts_1d = roots_legendre(n_points_per_dim)
    Gauss_Points_Tensor  = torch.tensor(pts_1d, dtype = dtype, device = device)
    Gauss_Weights_Tensor = torch.tensor(wts_1d, dtype = dtype, device = device)

    return Gauss_Points_Tensor, Gauss_Weights_Tensor



def Gauss_Quadratures_2D(n_points_per_dim: int,
                         dtype: torch.dtype = torch.float64, 
                         device: str = "cpu"
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes 2D Gauss-Legendre quadrature points and weights on [-1, 1]^2
    via tensor product for quadrilateral elements.

    Parameters
    ----------
    n_points_per_dim : int
        Number of Gauss points per axis (e.g., 2 -> 2x2 = 4 total points).
    dtype : torch.dtype
        Target PyTorch floating-point type.
    device : str
        Target compute device.

    Returns
    -------
    Gauss_Points_2D : torch.Tensor
        Integration coordinates of shape (n_points_per_dim**2, 2) where columns are [xi, eta].
    Gauss_Weights_2D : torch.Tensor
        Tensor-product integration weights of shape (n_points_per_dim**2,).
    """
    Gauss_Points_1D, Gauss_Weights_1D = Gauss_Quadratures_1D(n_points_per_dim, dtype = dtype, device = device)

    # Nested vmap to generate all (xi, eta) coordinate pairs -> shape (N^2, 2)
    Gauss_Points_2D  = vmap(lambda y: vmap(lambda x: torch.stack([x , y]))(Gauss_Points_1D))(Gauss_Points_1D).reshape(-1, 2)

    # Outer product for tensor-product weights (w_i * w_j) -> shape (N^2,)
    Gauss_Weights_2D = vmap(lambda y: vmap(lambda x: x * y)(Gauss_Weights_1D))(Gauss_Weights_1D).reshape(-1)

    # Or the following
    # Gauss_Weights_2D = torch.outer(Gauss_Weights_1D, Gauss_Weights_1D).reshape(-1)

    return Gauss_Points_2D, Gauss_Weights_2D