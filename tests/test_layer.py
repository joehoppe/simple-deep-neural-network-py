import numpy as np
import pytest

from neural_network.activation import Activation, identity, relu, sigmoid
from neural_network.layer import Layer


def _linear(name="linear"):
    """A pass-through activation so tests can check the affine result directly."""
    return Activation(name=name, fn=lambda z: z, derivative=lambda z: np.ones_like(z))


class TestForwardSingleVector:
    def test_matches_hand_computed_affine_through_activation(self):
        W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (n_out=3, n_in=2)
        b = np.array([0.5, -1.0, 0.0])
        layer = Layer(W, b, _linear())
        x = np.array([1.0, 1.0])

        out = layer.predict(x)

        expected = W @ x + b  # linear activation => affine result
        np.testing.assert_allclose(out, expected)

    def test_returns_shape_n_out(self):
        layer = Layer(np.zeros((4, 2)), np.zeros(4), identity)
        out = layer.predict(np.array([1.0, 2.0]))
        assert out.shape == (4,)

    def test_activation_is_applied_elementwise(self):
        W = np.array([[0.0, 0.0]])
        b = np.array([0.0])
        layer = Layer(W, b, sigmoid)  # z = 0 -> sigmoid(0) = 0.5
        np.testing.assert_allclose(layer.predict(np.array([3.0, 4.0])), [0.5])


class TestForwardBatch:
    def test_batch_matches_hand_computed(self):
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 2.0])
        layer = Layer(W, b, _linear())
        X = np.array([[1.0, 1.0], [2.0, 3.0], [0.0, 0.0]])  # (m=3, n_in=2)

        out = layer.predict(X)

        expected = X @ W.T + b
        np.testing.assert_allclose(out, expected)

    def test_batch_returns_shape_m_by_n_out(self):
        layer = Layer(np.zeros((5, 3)), np.zeros(5), identity)
        out = layer.predict(np.zeros((7, 3)))
        assert out.shape == (7, 5)


class TestFromNeurons:
    def test_builds_expected_matrix_and_bias(self):
        rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        bias = [0.1, 0.2]
        layer = Layer.from_neurons(rows, bias, identity)

        np.testing.assert_allclose(layer.W, np.array(rows))
        np.testing.assert_allclose(layer.b, np.array(bias))

    def test_forward_after_from_neurons(self):
        layer = Layer.from_neurons([[2.0]], [1.0], _linear())
        np.testing.assert_allclose(layer.predict(np.array([3.0])), [7.0])

    def test_non_rectangular_rows_raise_value_error(self):
        with pytest.raises(ValueError):
            Layer.from_neurons([[1.0, 2.0], [3.0]], [0.0, 0.0], identity)


class TestValidation:
    def test_w_must_be_2d(self):
        with pytest.raises(ValueError):
            Layer(np.array([1.0, 2.0, 3.0]), np.array([0.0]), identity)

    def test_b_length_must_match_w_rows(self):
        with pytest.raises(ValueError):
            Layer(np.zeros((3, 2)), np.zeros(2), identity)

    def test_b_must_be_1d(self):
        with pytest.raises(ValueError):
            Layer(np.zeros((3, 2)), np.zeros((3, 1)), identity)

    def test_activation_must_be_activation_instance(self):
        with pytest.raises(TypeError):
            Layer(np.zeros((3, 2)), np.zeros(3), lambda z: z)

    def test_predict_input_width_must_match(self):
        layer = Layer(np.zeros((3, 2)), np.zeros(3), identity)
        with pytest.raises(ValueError):
            layer.predict(np.array([1.0, 2.0, 3.0]))  # 3 != n_in=2

    def test_coerces_integer_inputs_to_float(self):
        layer = Layer([[1, 2], [3, 4]], [0, 0], relu)
        assert layer.W.dtype == np.float64
        assert layer.b.dtype == np.float64
