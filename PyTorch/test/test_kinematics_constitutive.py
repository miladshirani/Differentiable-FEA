"""Isoparametric kinematics and the neo-Hookean law."""
import math
import torch

from scr.shape_functions_2D import shape_fn_quad4
from scr.Kinematics import (Coordinate_Interpolation, Displacement_Interpolation,
                            Mesh_Jacobian, Global_Deformation_Gradient)
from scr.Constitutive_Relations import (neo_Hookean_energy, Piola_Stress, Piola_Batched,
                                        neo_Hookean_Batched)

DTYPE = torch.float64
MU, LMBDA = 1.0, 1.0


def test_interpolation_reproduces_nodal_values_at_nodes():
    Xe = torch.tensor([[0., 0.], [2., 0.], [2.5, 1.], [0., 1.]], dtype=DTYPE)
    X_fn = Coordinate_Interpolation(shape_fn_quad4, Xe)
    corners = torch.tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]], dtype=DTYPE)
    for a in range(4):
        assert torch.allclose(X_fn(corners[a, 0], corners[a, 1]), Xe[a])


def test_mesh_jacobian_of_rectangle_is_constant_half_lengths():
    Xe = torch.tensor([[0., 0.], [2., 0.], [2., 1.], [0., 1.]], dtype=DTYPE)       # 2 x 1 rectangle
    X_fn = Coordinate_Interpolation(shape_fn_quad4, Xe)
    J, detJ, invJ = Mesh_Jacobian(X_fn, torch.tensor(0.3, dtype=DTYPE), torch.tensor(-0.6, dtype=DTYPE))
    assert torch.allclose(J, torch.diag(torch.tensor([1.0, 0.5], dtype=DTYPE)))     # dX/dxi = Lx/2 ...
    assert torch.allclose(detJ, torch.tensor(0.5, dtype=DTYPE))                     # area / 4
    assert torch.allclose(J @ invJ, torch.eye(2, dtype=DTYPE))


def test_affine_displacement_gives_exact_deformation_gradient():
    """u = (F0 - I) X on a distorted element must give F = F0 at every point."""
    Xe = torch.tensor([[0., 0.], [2., 0.3], [2.2, 1.5], [-0.1, 1.]], dtype=DTYPE)
    F0 = torch.tensor([[1.2, 0.1], [-0.05, 0.9]], dtype=DTYPE)
    ue = Xe @ (F0 - torch.eye(2, dtype=DTYPE)).T
    X_fn = Coordinate_Interpolation(shape_fn_quad4, Xe)
    u_fn = Displacement_Interpolation(shape_fn_quad4, ue)
    F, detJ, _ = Global_Deformation_Gradient(u_fn, X_fn, torch.tensor(0.2, dtype=DTYPE),
                                             torch.tensor(-0.7, dtype=DTYPE))
    assert torch.allclose(F, F0, atol=1e-12)


def test_energy_and_stress_vanish_at_identity():
    F = torch.eye(2, dtype=DTYPE)
    assert abs(neo_Hookean_energy(F, MU, LMBDA)) < 1e-14
    assert torch.allclose(Piola_Stress(F, MU, LMBDA), torch.zeros(2, 2, dtype=DTYPE), atol=1e-14)


def test_piola_matches_closed_form():
    """P = mu (F - F^-T) + lambda ln(J) F^-T"""
    F = torch.tensor([[1.1, 0.2], [0.05, 0.95]], dtype=DTYPE)
    Finv_T = torch.linalg.inv(F).T
    J = torch.linalg.det(F)
    expected = MU * (F - Finv_T) + LMBDA * torch.log(J) * Finv_T
    assert torch.allclose(Piola_Stress(F, MU, LMBDA), expected, atol=1e-12)


def test_batched_equals_looped():
    Fb = torch.tensor([[[1., 0.], [0., 1.]], [[1.1, 0.], [0., 1.]], [[1., 0.2], [0., 1.]]], dtype=DTYPE)
    assert torch.allclose(neo_Hookean_Batched(Fb, MU, LMBDA),
                          torch.stack([neo_Hookean_energy(F, MU, LMBDA) for F in Fb]))
    assert torch.allclose(Piola_Batched(Fb, MU, LMBDA),
                          torch.stack([Piola_Stress(F, MU, LMBDA) for F in Fb]))


def test_frame_indifference():
    """psi(Q F) = psi(F) for a rotation Q (objectivity)."""
    F = torch.tensor([[1.1, 0.2], [0.05, 0.95]], dtype=DTYPE)
    a = 0.8
    Q = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=DTYPE)
    assert abs(neo_Hookean_energy(Q @ F, MU, LMBDA) - neo_Hookean_energy(F, MU, LMBDA)) < 1e-12
