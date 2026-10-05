"""
weak_form.py
============

The WEAK FORM (principle of virtual work) of finite-strain elasticity.

Strong form:    Div P = 0   in the body,      P N = t   on the loaded boundary.
Multiply by a virtual displacement field v and integrate by parts:

        delta W(u; v) = integral_Omega  P(F(u)) : Grad_X(v)  dOmega   -   (external work)

At equilibrium delta W = 0 for EVERY admissible v.  The pieces below compute
the integrand at ONE Gauss point and then sum it over the quadrature rule of
ONE element.

Design rule: every function works on one point / one element with fixed shapes;
``vmap`` supplies the loops (see operators.py).
"""

import torch
from torch.func import vmap

from difffea.constitutive_relations import Piola_Stress
from difffea.kinematics import Local_Deformation_Gradient, Global_Deformation_Gradient


# =============================================================================
# "Local" versions: integrand in PARENT coordinates only (no mesh geometry).
# Useful for hand examples and unit tests on the reference element.
# =============================================================================
def Local_Weak_Form(u_fn, mu, lmbda, xi, eta):
    """
    Integrand  P(F_local(u)) : Grad_xi(u)  at one point, geometry ignored
    (i.e. the element is the reference element itself).

    Returns a scalar ().
    """
    Local_F, Local_Grad_u = Local_Deformation_Gradient(u_fn, xi, eta)    # both (2, 2)
    Local_P = Piola_Stress(Local_F, mu, lmbda)                           # (2, 2)
    # double contraction A:B = sum_ij A_ij B_ij
    return torch.einsum("ij,ij->", Local_P, Local_Grad_u)


def Integrated_Local_Weak_Form(u_fn, mu, lmbda, Gauss_Points, Gauss_Weights):
    """
    Quadrature sum of ``Local_Weak_Form`` over all Gauss points of one element.

    Gauss_Points : (n_q, 2)    Gauss_Weights : (n_q,)    ->   scalar ()
    """
    # vmap over the Gauss points: u_fn, mu, lmbda are shared (None), xi, eta batched (0)
    Local_Weak = vmap(Local_Weak_Form, in_dims=(None, None, None, 0, 0))
    values = Local_Weak(u_fn, mu, lmbda, Gauss_Points[:, 0], Gauss_Points[:, 1])   # (n_q,)
    # weighted sum  sum_q w_q f_q
    return torch.einsum("i,i->", values, Gauss_Weights)


# =============================================================================
# "Global" versions: integrand on the PHYSICAL element (uses the mesh Jacobian)
# =============================================================================
def Global_Weak_Form(u_fn, X_fn, mu, lmbda, xi, eta):
    """
    Integrand  P(F(u)) : Grad_X(u) * det(dX/dxi)  at one Gauss point.

    (Self-paired virtual field v = u, i.e. a "stress power"-type density.)
    """
    Global_F, det_Grad_X, Global_Grad_u = Global_Deformation_Gradient(u_fn, X_fn, xi, eta)
    Global_P = Piola_Stress(Global_F, mu, lmbda)
    return torch.einsum("ij,ij->", Global_P, Global_Grad_u) * det_Grad_X


def Integrated_Global_Weak_Form(u_fn, X_fn, mu, lmbda, Gauss_Points, Gauss_Weights):
    """Quadrature sum of ``Global_Weak_Form`` over one element -> scalar ()."""
    Global_Weak = vmap(Global_Weak_Form, in_dims=(None, None, None, None, 0, 0))
    values = Global_Weak(u_fn, X_fn, mu, lmbda, Gauss_Points[:, 0], Gauss_Points[:, 1])  # (n_q,)
    return torch.einsum("i,i->", values, Gauss_Weights)


# =============================================================================
# The virtual work that actually drives the solver
# =============================================================================
def Global_Virtual_Work(u_fn, v_fn, X_fn, mu, lmbda, xi, eta):
    """
    Virtual-work density at ONE Gauss point (xi, eta):

            dW(xi, eta) = P(F(u)) : Grad_X(v) * det(dX/dxi)

    Note the TRIAL field u enters through the stress, the VIRTUAL field v only
    through its gradient.  dW is LINEAR in v: that single fact lets us recover
    the element force vector by differentiating with respect to v (see
    ``elem_residual`` in operators.py).

    Parameters
    ----------
    u_fn, v_fn : closures (xi, eta) -> (2,)   trial and virtual displacement
    X_fn       : closure (xi, eta) -> (2,)    geometry map
    mu, lmbda  : material parameters (scalars)
    xi, eta    : scalar tensors, the Gauss point

    Returns
    -------
    () scalar
    """
    # 1. kinematics of the trial field:  F (2,2)  and the volume factor det J
    F, det_Grad_X, _ = Global_Deformation_Gradient(u_fn, X_fn, xi, eta)

    # 2. kinematics of the virtual field: only its gradient Grad_X(v) (2,2) is needed
    _, _, Grad_v = Global_Deformation_Gradient(v_fn, X_fn, xi, eta)

    # 3. constitutive law: stress from the TRIAL deformation
    P = Piola_Stress(F, mu, lmbda)                                  # (2, 2)

    # 4. double contraction P : Grad_v, scaled by the area factor det J
    return torch.einsum("ij,ij->", P, Grad_v) * det_Grad_X


def Integrated_Global_Virtual_Work(u_fn, v_fn, X_fn, mu, lmbda, Gauss_Points, Gauss_Weights):
    """
    Virtual work of ONE element = quadrature sum of ``Global_Virtual_Work``:

            delta W^e = sum_q  w_q * dW(xi_q, eta_q)

    Parameters
    ----------
    Gauss_Points  : (n_q, 2)   [xi, eta] of every Gauss point
    Gauss_Weights : (n_q,)

    Returns
    -------
    () scalar
    """
    # vmap over Gauss points.  in_dims = (None x5, 0, 0): the five closures /
    # parameters are shared, only (xi, eta) change from point to point.
    batched_virtual_work = vmap(Global_Virtual_Work,
                                in_dims=(None, None, None, None, None, 0, 0))

    integrands = batched_virtual_work(u_fn, v_fn, X_fn, mu, lmbda,
                                      Gauss_Points[:, 0], Gauss_Points[:, 1])   # (n_q,)

    # quadrature: dot product of integrand values with the weights
    return torch.einsum("q,q->", integrands, Gauss_Weights)
