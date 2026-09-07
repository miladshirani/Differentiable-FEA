import jax
import jax.numpy as jnp


# In this code you can define different constitutive relations
# In this code F is deformation gradient

def neo_Hookean_energy(F: jnp.ndarray, mu: float, lmbda: float) -> float:
    dim =F.shape[0]
    C = jnp.dot(F.T, F)
    J = jnp.linalg.det(F)
    logJ = jnp.log(J)
    I1 = jnp.trace(C)

    return  0.5 * mu * (I1 - dim - 2 * logJ) + 0.5 * lmbda * (logJ**2)

def Piola_Stress(F: jnp.ndarray, mu: float, lmbda: float) -> jnp.ndarray:
    Piola = jax.grad(neo_Hookean_energy, argnums=0)(F, mu, lmbda)
    return Piola

Piola_Batched         = jax.vmap(Piola_Stress,       in_axes = (0, None, None))
neo_Hookean_Batched   = jax.vmap(neo_Hookean_energy, in_axes = (0, None, None))


