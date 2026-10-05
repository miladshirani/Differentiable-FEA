"""Shape functions: Kronecker delta, partition of unity, vanishing gradient sum, AD vs finite difference."""
import pytest
import torch
from torch.func import vmap

from scr.Elements import ELEMENT_LIBRARY, element_names, reference_node_coordinates
from scr.Grad_shape_functions_2D import Grad_Shape_Functions_2D, Grad_Shape_Functions_2D_Batched

DTYPE = torch.float64


@pytest.mark.parametrize("name", element_names())
def test_kronecker_delta_at_nodes(name):
    """N_a(node_b) = delta_ab: each function is 1 at its own node and 0 at all others."""
    nodes = reference_node_coordinates(name)
    N = vmap(ELEMENT_LIBRARY[name]["shape_fn"])(nodes[:, 0], nodes[:, 1])
    assert torch.allclose(N, torch.eye(nodes.shape[0], dtype=DTYPE), atol=1e-12)


@pytest.mark.parametrize("name", element_names())
def test_partition_of_unity_and_zero_gradient_sum(name):
    """sum_a N_a = 1  =>  sum_a grad N_a = 0 (a constant field has no gradient)."""
    el = ELEMENT_LIBRARY[name]
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(20, 2, generator=g, dtype=DTYPE) * 0.5       # inside both reference domains
    N = vmap(el["shape_fn"])(pts[:, 0], pts[:, 1])
    assert torch.allclose(N.sum(1), torch.ones(20, dtype=DTYPE), atol=1e-12)
    dN = Grad_Shape_Functions_2D_Batched(el["shape_fn"], pts[:, 0], pts[:, 1])   # (20, nen, 2)
    assert dN.shape == (20, el["n_nodes"], 2)
    assert dN.sum(1).abs().max() < 1e-12


@pytest.mark.parametrize("name", element_names())
def test_autodiff_gradient_matches_finite_difference(name):
    fn = ELEMENT_LIBRARY[name]["shape_fn"]
    xi, eta = torch.tensor(0.21, dtype=DTYPE), torch.tensor(0.17, dtype=DTYPE)
    dN = Grad_Shape_Functions_2D(fn, xi, eta)
    h = 1e-6
    fd_xi = (fn(xi + h, eta) - fn(xi - h, eta)) / (2 * h)
    fd_eta = (fn(xi, eta + h) - fn(xi, eta - h)) / (2 * h)
    assert torch.allclose(dN[:, 0], fd_xi, atol=1e-8)
    assert torch.allclose(dN[:, 1], fd_eta, atol=1e-8)


def test_quad4_gradient_matches_hand_derivation():
    """dN/dxi = 1/4 [-(1-eta), (1-eta), (1+eta), -(1+eta)]"""
    from scr.shape_functions_2D import shape_fn_quad4
    xi, eta = torch.tensor(0.3, dtype=DTYPE), torch.tensor(-0.4, dtype=DTYPE)
    dN = Grad_Shape_Functions_2D(shape_fn_quad4, xi, eta)
    expected_dxi = 0.25 * torch.tensor([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=DTYPE)
    assert torch.allclose(dN[:, 0], expected_dxi)
