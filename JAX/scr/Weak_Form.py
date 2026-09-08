import jax
import jax.numpy as jnp

from scr.constitutive_relations import Piola_Stress
from scr.Kinematics import Local_Deformation_Gradient, Global_Deformation_Gradient
from scr.Gauss_Quadratures import Gauss_Quadratures_2D





Gauss_Points, Gauss_Weights = Gauss_Quadratures_2D(2)

def Local_Weak_Form(u_fn, mu: float, lmbda: float, xi: float, eta: float) -> float:
    
    Local_F, local_Grad_u = Local_Deformation_Gradient(u_fn, xi, eta)
    Local_P = Piola_Stress(Local_F, mu, lmbda)
    
    # Double contraction P : Grad_u = sum_ij (P_ij * Grad_u_ij)
    return jnp.tensordot(Local_P, local_Grad_u)


def Integrated_Local_Weak_Form(u_fn, mu: float, lmbda: float,
                                Gauss_Points: jnp.ndarray, # Shape: (n_gp, 2) -> columns are [xi, eta]
                                Gauss_Weights: jnp.ndarray # Shape: (n_gp,)
                                ) -> float:
    """
    Integrates the local weak form across all Gauss points on the reference element.

    Parameters
    ----------
    u_fn : Callable
        Interpolated displacement function u(xi, eta) -> (2,).
    mu : float
        Lame parameter (shear modulus).
    lmbda : float
        Lame parameter.
    Gauss_Points : jnp.ndarray
        Array of Gauss integration coordinates, shape (n_gp, 2).
    Gauss_Weights : jnp.ndarray
        Array of Gauss weights, shape (n_gp,).

    Returns
    -------
    float
        Integrated scalar reference virtual work.
    """
    
    Local_Weak = jax.vmap(Local_Weak_Form, in_axes=(None, None, None, 0, 0))
    Local_Weak_Batched = Local_Weak(u_fn, mu, lmbda, Gauss_Points[:,0], Gauss_Points[:,1])
    Total_Local_Weak_form = jnp.dot(Local_Weak_Batched, Gauss_Weights)

    return Total_Local_Weak_form

def Global_Weak_Form(u_fn, X_fn, mu: float, lmbda: float, xi: float, eta: float) -> float:

    Global_F, det_Global_X, Global_Grad_u  = Global_Deformation_Gradient(u_fn, X_fn, xi, eta)                       
    Global_P = Piola_Stress(Global_F, mu, lmbda)
    Global_Weak_Form = jnp.tensordot(Global_P, Global_Grad_u) * det_Global_X

    return Global_Weak_Form


def Integrated_Global_Weak_Form(u_fn, X_fn, mu: float, lmbda: float, 
                                Gauss_Points: jnp.ndarray, # Shape: (n_gp, 2) -> columns are [xi, eta]
                                Gauss_Weights: jnp.ndarray # Shape: (n_gp,)
                                ) -> float:
    

    G_weak = jax.vmap(Global_Weak_Form, in_axes=(None, None, None, None, 0, 0))
    G_weak_Batched = G_weak(u_fn, X_fn, mu, lmbda, Gauss_Points[:, 0], Gauss_Points[:, 1])
    G_total_weak_form = jnp.dot(G_weak_Batched, Gauss_Weights)

    return G_total_weak_form

    

    

