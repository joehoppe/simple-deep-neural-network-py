import pytest

from neural_network import PerceptronService, WeightNode


class DataNode:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


def test_predicts_one_when_weighted_sum_meets_threshold():
    weights = WeightNode(0.5, WeightNode(1.5, WeightNode(-1.0)))
    perceptron = PerceptronService(weights, bias=0.25, threshold=1.0)

    assert perceptron.dot_product([2, 1, 0.5]) == 2.0
    assert perceptron.weighted_sum([2, 1, 0.5]) == 2.25
    assert perceptron.predict([2, 1, 0.5]) == 1


def test_predicts_zero_when_weighted_sum_is_below_threshold():
    perceptron = PerceptronService.from_weights([0.25, -0.5], threshold=1.0)

    assert perceptron.predict([2, 1]) == 0


def test_accepts_nodes_with_data_attribute():
    weights = DataNode(2, DataNode(3))
    perceptron = PerceptronService(weights)

    assert perceptron.weighted_sum([4, 5]) == 23.0


def test_predict_uses_injected_activation_function():
    weights = WeightNode(0.5, WeightNode(1.5))
    perceptron = PerceptronService(
        weights,
        bias=0.25,
        activation_function=lambda total: total * 2,
    )

    assert perceptron.predict([2, 1]) == 5.5


def test_from_weights_accepts_injected_activation_function():
    perceptron = PerceptronService.from_weights(
        [1, -1],
        activation_function=lambda total: max(0.0, total),
    )

    assert perceptron.predict([2, 3]) == 0.0


def test_rejects_input_count_mismatch():
    perceptron = PerceptronService.from_weights([1, 2])

    with pytest.raises(ValueError):
        perceptron.predict([1])

    with pytest.raises(ValueError):
        perceptron.predict([1, 2, 3])


def test_rejects_non_numeric_weights():
    perceptron = PerceptronService(WeightNode("bad"))

    with pytest.raises(TypeError):
        perceptron.predict([1])


def test_rejects_non_callable_activation_function():
    with pytest.raises(TypeError):
        PerceptronService.from_weights([1], activation_function=1)


def test_rejects_non_numeric_activation_output():
    perceptron = PerceptronService.from_weights(
        [1],
        activation_function=lambda total: "bad",
    )

    with pytest.raises(TypeError):
        perceptron.predict([1])
