import jax
import jax.numpy as jnp
from typing import Callable, Tuple



def Interpolation_Fn(ShapeFunctions: jnp.ndarray,  # shape (n_nodes,) evaluated at (xi, eta)
                        nodal_values: jnp.ndarray   #  Shape: (n_nodes, 2) e.g., X_nodes or u_nodes
                         ) -> jnp.ndarray:

    """
    Interpolates any discrete nodal vector field to points within an element.

    Computes the isoparametric field interpolation via the linear combination:
        v(xi, eta) = sum_{i=1}^{n_nodes} N_i(xi, eta) * v_i = N @ nodal_values

    Parameters
    ----------
    ShapeFunctions : jnp.ndarray
        Array of shape function values evaluated at reference coordinates (xi, eta).
        Can be 1D with shape (n_nodes,) for a single point, or 2D with shape
        (n_points, n_nodes) for batched evaluation across multiple points.
    nodal_values : jnp.ndarray
        Discrete nodal values of the field (e.g., coordinates X or displacements u),
        with shape (n_nodes, 2).

    Returns
    -------
    jnp.ndarray
        Interpolated vector field value(s). Shape is (2,) for a single evaluation
        point, or (n_points, 2) when evaluated across multiple points.
    """
    # Dot/Matrix product: (n_nodes,) @ (n_nodes, 2) -> (2,)
    # Batch product:      (n_pts, n_nodes) @ (n_nodes, 2) -> (n_pts, 2)
    return ShapeFunctions @ nodal_values


def Coordinate_Interpolation(ShapeFunctions: jnp.ndarray, 
                             Nodal_Coordinates: jnp.ndarray,  # Shape: (n_nodes, 2)
                            ) -> jnp.ndarray:
    """
    Maps reference parent coordinates (xi, eta) to physical reference coordinates (X, Y).

    Computes the physical undeformed position vector inside the element:
        X(xi, eta) = sum_{i=1}^{n_nodes} N_i(xi, eta) * X_i

    Parameters
    ----------
    ShapeFunctions : jnp.ndarray
        Evaluated shape functions with shape (n_nodes,) or (n_points, n_nodes).
    Nodal_Coordinates : jnp.ndarray
        Reference configuration coordinates of the element nodes [X, Y],
        with shape (n_nodes, 2).

    Returns
    -------
    jnp.ndarray
        Interpolated undeformed coordinate vector(s) [X, Y].
        Shape: (2,) for a single point, or (n_points, 2) for a batch.
    """
    X_inside = Interpolation_Fn(ShapeFunctions, Nodal_Coordinates)
    return X_inside


def Displacement_Interpolation(ShapeFunctions: jnp.ndarray, 
                              Nodal_Displacements: jnp.ndarray,  # Shape: (n_nodes, 2)
                              ) -> jnp.ndarray:
    """
    Interpolates nodal displacements to continuous interior displacement fields.

    Computes the displacement vector field inside the element:
        u(xi, eta) = sum_{i=1}^{n_nodes} N_i(xi, eta) * u_i

    Parameters
    ----------
    ShapeFunctions : jnp.ndarray
        Evaluated shape functions with shape (n_nodes,) or (n_points, n_nodes).
    Nodal_Displacements : jnp.ndarray
        Displacement components of the element nodes [u_x, u_y],
        with shape (n_nodes, 2).

    Returns
    -------
    jnp.ndarray
        Interpolated displacement vector(s) [u_x, u_y].
        Shape: (2,) for a single point, or (n_points, 2) for a batch.
    """
    U_inside = Interpolation_Fn(ShapeFunctions, Nodal_Displacements)
    return U_inside


def Mesh_Jacobian(X_fn: Callable[[float, float], jnp.ndarray],
                  xi: float, 
                  eta: float) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
    """
    Computes the mesh Jacobian matrix, its determinant, and its inverse
    at a reference point (xi, eta).

    Parameters
    ----------
    X_fn : Callable
        Interpolated physical coordinate function X(xi, eta) -> (2,).
    xi : float
        Parent coordinate xi.
    eta : float
        Parent coordinate eta.

    Returns
    -------
    J_mesh : jnp.ndarray
        Mesh Jacobian dX/dxi of shape (2, 2).
    det_J : float
        Determinant of the Jacobian (area scaling factor).
    inv_J : jnp.ndarray
        Inverse of the Jacobian matrix of shape (2, 2).
    """
    dX_dxi, dX_deta = jax.jacfwd(X_fn, argnums=(0, 1))(xi, eta)  # each has shape (2,)
    Grad_X = jnp.stack([dX_dxi, dX_deta], axis=-1)
    det_Grad_x = jnp.linalg.det(Grad_X)
    inv_Grad_X = jnp.linalg.inv(Grad_X)

    return Grad_X, det_Grad_x, inv_Grad_X



def Local_Deformation_Gradient(u_fn, xi: float, eta: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes reference deformation gradient F and displacement gradient Grad_u
    by differentiating the interpolated displacement field u(xi, eta).
    """
    du_dxi, du_deta = jax.jacfwd(u_fn, argnums=(0, 1))(xi, eta)  # each has shape (2,)

    # Stack along columns to form grad_xi(u) = [du/dxi, du/deta] of shape (2, 2)
    Local_Grad_u = jnp.stack([du_dxi, du_deta], axis=-1)
    Local_F = jnp.eye(2) + Local_Grad_u

    return Local_F, Local_Grad_u


def Global_Deformation_Gradient(u_fn: Callable[[float, float], jnp.ndarray],
                                X_fn: Callable[[float, float], jnp.ndarray],
                                xi: float,
                                eta: float
                                ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Computes the physical deformation gradient F, displacement gradient grad_X(u),
    and area scaling factor det(J) using the chain rule.

    Parameters
    ----------
    u_fn : Callable
        Interpolated displacement function u(xi, eta) -> (2,).
    X_fn : Callable
        Interpolated coordinate function X(xi, eta) -> (2,).
    xi : float
        Parent coordinate xi.
    eta : float
        Parent coordinate eta.

    Returns
    -------
    F : jnp.ndarray
        Total physical deformation gradient tensor of shape (2, 2).
    Grad_u_physical : jnp.ndarray
        Physical displacement gradient du/dX of shape (2, 2).
    det_J : float
        Jacobian determinant for numerical integration weighting.
    """

    _ , Local_Grad_u = Local_Deformation_Gradient(u_fn, xi, eta)
    _ , det_Grad_X, inv_Grad_X = Mesh_Jacobian(X_fn, xi, eta)

    Global_Grad_u = Local_Grad_u @ inv_Grad_X
    Global_F = jnp.eye(2) + Global_Grad_u
    

    return Global_F, det_Grad_X, Global_Grad_u 
    


