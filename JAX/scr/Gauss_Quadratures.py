import jax
import jax.numpy as jnp 
from scipy.special import roots_legendre


def Gauss_Quadratures_1D(n_points_per_dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes 1D Gauss-Legendre quadrature points and weights on [-1, 1].
    
    Args:
        n_points_per_dim: Number of Gauss points.
        
    Returns:
        Gauss_Points_1D: Integration coordinates, shape (n_points_per_dim,).
        Gauss_Weights_1D: Quadrature weights, shape (n_points_per_dim,).
    """
    Gauss_Points_1D, Gauss_Weights_1D = roots_legendre(n_points_per_dim)
    return jnp.array(Gauss_Points_1D), jnp.array(Gauss_Weights_1D)


def Gauss_Quadratures_2D(n_points_per_dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes 2D Gauss-Legendre quadrature points and weights on [-1, 1]^2.
    
    Returns:
        Gauss_Points_2D: Coordinates of shape (n_points_per_dim**2, 2).
        Gauss_Weights_2D: Weights of shape (n_points_per_dim**2,).
    """
    Gauss_Points_1D, Gauss_Weights_1D = Gauss_Quadratures_1D(n_points_per_dim)

    Gauss_Points_2D   = jax.vmap(lambda y: jax.vmap(lambda x: jnp.array([x, y]))(Gauss_Points_1D))(Gauss_Points_1D).reshape(-1,2)
    Gauss_Weights_2D  = jnp.outer(Gauss_Weights_1D, Gauss_Weights_1D).ravel()

    return Gauss_Points_2D, Gauss_Weights_2D

