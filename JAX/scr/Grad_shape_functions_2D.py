import jax
import jax.numpy as jnp
from typing import Callable

def Grad_Shape_Functions(shape_fn: Callable,
                          xi: jnp.ndarray,
                          eta: jnp.ndarray) -> jnp.ndarray:
    
    """Gradient of shape functions w.r.t. (xi, eta). Returns shape (n_nodes, 2)."""

    dN_dxi, dN_deta = jax.jacfwd(shape_fn, argnums=(0, 1))(xi, eta)
    
    return jnp.stack([dN_dxi, dN_deta], axis=-1)