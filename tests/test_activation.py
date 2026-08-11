import numpy as np
import pytest

from neural_network.activation import (
    Activation,
    identity,
    relu,
    sigmoid,
    tanh,
)


class TestActivationFn:
    def test_sigmoid_at_zero_is_half(self):
        assert sigmoid.fn(0.0) == pytest.approx(0.5)

    def test_tanh_at_zero_is_zero(self):
        assert tanh.fn(0.0) == pytest.approx(0.0)

    def test_relu_clamps_negatives_to_zero(self):
        assert relu.fn(-1.0) == pytest.approx(0.0)

    def test_relu_passes_positives_through(self):
        assert relu.fn(2.0) == pytest.approx(2.0)

    def test_identity_returns_input(self):
        assert identity.fn(3.5) == pytest.approx(3.5)


class TestActivationDerivative:
    def test_sigmoid_derivative_at_zero_is_quarter(self):
        assert sigmoid.derivative(0.0) == pytest.approx(0.25)

    def test_relu_derivative_of_negative_is_zero(self):
        assert relu.derivative(-1.0) == pytest.approx(0.0)

    def test_relu_derivative_of_positive_is_one(self):
        assert relu.derivative(2.0) == pytest.approx(1.0)

    def test_relu_derivative_at_zero_is_zero(self):
        assert relu.derivative(0.0) == pytest.approx(0.0)

    def test_tanh_derivative_at_zero_is_one(self):
        assert tanh.derivative(0.0) == pytest.approx(1.0)

    def test_identity_derivative_is_one(self):
        assert identity.derivative(42.0) == pytest.approx(1.0)


class TestElementwiseOverArrays:
    def test_sigmoid_fn_is_elementwise(self):
        z = np.array([-1.0, 0.0, 1.0])
        expected = 1.0 / (1.0 + np.exp(-z))
        np.testing.assert_allclose(sigmoid.fn(z), expected)

    def test_relu_fn_is_elementwise(self):
        z = np.array([-2.0, -0.5, 0.0, 3.0])
        np.testing.assert_allclose(relu.fn(z), np.array([0.0, 0.0, 0.0, 3.0]))

    def test_tanh_fn_is_elementwise(self):
        z = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(tanh.fn(z), np.tanh(z))

    def test_identity_fn_is_elementwise(self):
        z = np.array([-1.0, 0.0, 2.5])
        np.testing.assert_allclose(identity.fn(z), z)

    def test_sigmoid_derivative_is_elementwise(self):
        z = np.array([-1.0, 0.0, 1.0])
        s = 1.0 / (1.0 + np.exp(-z))
        np.testing.assert_allclose(sigmoid.derivative(z), s * (1.0 - s))

    def test_relu_derivative_is_elementwise(self):
        z = np.array([-2.0, 0.0, 3.0])
        np.testing.assert_allclose(relu.derivative(z), np.array([0.0, 0.0, 1.0]))

    def test_tanh_derivative_is_elementwise(self):
        z = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(tanh.derivative(z), 1.0 - np.tanh(z) ** 2)

    def test_identity_derivative_is_ones_like_input(self):
        z = np.array([-1.0, 0.0, 2.5])
        np.testing.assert_allclose(identity.derivative(z), np.ones_like(z))


class TestActivationValueObject:
    def test_is_frozen_dataclass(self):
        with pytest.raises(Exception):
            sigmoid.name = "changed"

    def test_carries_its_name(self):
        assert sigmoid.name == "sigmoid"
        assert relu.name == "relu"
        assert tanh.name == "tanh"
        assert identity.name == "identity"

    def test_can_construct_custom_activation(self):
        custom = Activation(name="double", fn=lambda z: 2 * z, derivative=lambda z: 2)
        assert custom.fn(3) == 6
        assert custom.derivative(99) == 2
