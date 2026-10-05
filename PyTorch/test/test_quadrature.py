"""Quadrature rules integrate polynomials exactly up to their degree."""
import math
import pytest
import torch

from scr.Gauss_Quadratures import (Gauss_Quadratures_1D, Gauss_Quadratures_2D,
                                   Gauss_Quadratures_Triangle)
from scr.Elements import element_names, element_quadrature, get_element


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_gauss_1d_exact_to_degree_2n_minus_1(n):
    x, w = Gauss_Quadratures_1D(n)
    for k in range(2 * n):                                   # degrees 0 .. 2n-1
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        assert abs(torch.sum(w * x ** k) - exact) < 1e-12


def test_gauss_2d_weights_sum_to_area_and_integrate_monomials():
    P, W = Gauss_Quadratures_2D(3)
    assert abs(W.sum() - 4.0) < 1e-12                        # area of [-1,1]^2
    # integral of xi^2 eta^2 over [-1,1]^2 is (2/3)^2
    assert abs(torch.sum(W * P[:, 0] ** 2 * P[:, 1] ** 2) - 4.0 / 9.0) < 1e-12


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6, 8])
def test_triangle_rule_exact_up_to_degree(degree):
    """integral of xi^a eta^b over the unit triangle = a! b! / (a+b+2)!"""
    P, W = Gauss_Quadratures_Triangle(degree)
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            exact = math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
            assert abs(torch.sum(W * P[:, 0] ** a * P[:, 1] ** b) - exact) < 1e-12


@pytest.mark.parametrize("name", element_names())
def test_default_rule_weights_equal_reference_area(name):
    GP, GW = element_quadrature(name)
    area = 4.0 if get_element(name)["family"] == "quad" else 0.5
    assert abs(GW.sum() - area) < 1e-12
    assert GP.shape == (GW.shape[0], 2)
