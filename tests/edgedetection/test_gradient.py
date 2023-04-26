import pytest

import numpy
import numpy as np

from py_src.swdatatoolkit.edgedetection import GradientCalculator, Gradient


####################################
# Test Gradient
####################################
def test_gradient_throws_when_no_argument_in_constructor():
    with pytest.raises(ValueError):
        Gradient()


def test_gradient_throws_when_only_ny_argument_in_constructor():
    with pytest.raises(ValueError):
        Gradient(ny=1)


def test_gradient_throws_when_only_nx_argument_in_constructor():
    with pytest.raises(ValueError):
        Gradient(nx=1)


def test_gradient_throws_when_gy_shape_different_in_constructor():
    gx = np.arange(9).reshape((3, 3))
    gy = np.arange(12).reshape((3, 4))
    with pytest.raises(ValueError):
        Gradient(gx=gx, gy=gy)


def test_gradient_throws_when_gd_shape_different_in_constructor():
    gx = np.arange(9).reshape((3, 3))
    gy = np.arange(9).reshape((3, 3))
    gd = np.arange(12).reshape((3, 4))
    with pytest.raises(ValueError):
        Gradient(gx=gx, gy=gy, gd=gd)


####################################
# Test GradientCalculator
####################################
def test_gradient_calculator_throws_when_wrong_argument_in_constructor():
    with pytest.raises(NotImplementedError):
        GradientCalculator('something')


def test_gradient_calculator_throws_when_wrong_argument_type_in_cart():
    calc = GradientCalculator()
    arg = Gradient(nx=2, ny=2)
    with pytest.raises(TypeError):
        calc.calculate_gradient_cart(arg)


def test_gradient_calculator_throws_when_wrong_argument_type_in_polar():
    calc = GradientCalculator()
    arg = Gradient(nx=2, ny=2)
    with pytest.raises(TypeError):
        calc.calculate_gradient_polar(arg)


def test_gradient_calculator_cart_horizontal_sobel():
    a = np.arange(9).reshape((3, 3))
    b = np.array([[0, 0, 0], [0, 8, 0], [0, 0, 0]])

    calc = GradientCalculator()
    grad = calc.calculate_gradient_cart(a)

    assert (grad.gx == b).all()


def test_gradient_calculator_cart_verticle_sobel():
    a = np.arange(9).reshape((3, 3))
    b = np.array([[0, 0, 0], [0, 24, 0], [0, 0, 0]])

    calc = GradientCalculator()
    grad = calc.calculate_gradient_cart(a)

    assert (grad.gy == b).all()


def test_gradient_calculator_cart_diag_sobel():
    a = np.arange(9).reshape((3, 3))
    b = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])

    calc = GradientCalculator()
    grad = calc.calculate_gradient_cart(a)

    assert (grad.gd == b).all()


def test_gradient_calculator_cart_horizontal_prewitt():
    a = np.arange(9).reshape((3, 3))
    b = np.array([[0, 0, 0], [0, 6, 0], [0, 0, 0]])

    calc = GradientCalculator('prewitt')
    grad = calc.calculate_gradient_cart(a)

    assert (grad.gx == b).all()


def test_gradient_calculator_cart_verticle_prewitt():
    a = np.arange(9).reshape((3, 3))
    b = np.array([[0, 0, 0], [0, -18, 0], [0, 0, 0]])

    calc = GradientCalculator('prewitt')
    grad = calc.calculate_gradient_cart(a)

    assert (grad.gy == b).all()


def test_gradient_calculator_cart_diag_prewitt():
    a = np.arange(9).reshape((3, 3))
    b = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])

    calc = GradientCalculator('prewitt')
    grad = calc.calculate_gradient_cart(a)

    assert (grad.gd == b).all()
