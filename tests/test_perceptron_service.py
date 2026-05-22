import unittest

from neural_network import PerceptronService, WeightNode


class DataNode:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


class PerceptronServiceTests(unittest.TestCase):
    def test_predicts_one_when_weighted_sum_meets_threshold(self):
        weights = WeightNode(0.5, WeightNode(1.5, WeightNode(-1.0)))
        perceptron = PerceptronService(weights, bias=0.25, threshold=1.0)

        self.assertEqual(perceptron.weighted_sum([2, 1, 0.5]), 2.25)
        self.assertEqual(perceptron.predict([2, 1, 0.5]), 1)

    def test_predicts_zero_when_weighted_sum_is_below_threshold(self):
        perceptron = PerceptronService.from_weights([0.25, -0.5], threshold=1.0)

        self.assertEqual(perceptron.predict([2, 1]), 0)

    def test_accepts_nodes_with_data_attribute(self):
        weights = DataNode(2, DataNode(3))
        perceptron = PerceptronService(weights)

        self.assertEqual(perceptron.weighted_sum([4, 5]), 23.0)

    def test_predict_uses_injected_activation_function(self):
        weights = WeightNode(0.5, WeightNode(1.5))
        perceptron = PerceptronService(
            weights,
            bias=0.25,
            activation_function=lambda total: total * 2,
        )

        self.assertEqual(perceptron.predict([2, 1]), 5.5)

    def test_from_weights_accepts_injected_activation_function(self):
        perceptron = PerceptronService.from_weights(
            [1, -1],
            activation_function=lambda total: max(0.0, total),
        )

        self.assertEqual(perceptron.predict([2, 3]), 0.0)

    def test_rejects_input_count_mismatch(self):
        perceptron = PerceptronService.from_weights([1, 2])

        with self.assertRaises(ValueError):
            perceptron.predict([1])

        with self.assertRaises(ValueError):
            perceptron.predict([1, 2, 3])

    def test_rejects_non_numeric_weights(self):
        perceptron = PerceptronService(WeightNode("bad"))

        with self.assertRaises(TypeError):
            perceptron.predict([1])

    def test_rejects_non_callable_activation_function(self):
        with self.assertRaises(TypeError):
            PerceptronService.from_weights([1], activation_function=1)

    def test_rejects_non_numeric_activation_output(self):
        perceptron = PerceptronService.from_weights([1], activation_function=lambda total: "bad")

        with self.assertRaises(TypeError):
            perceptron.predict([1])


if __name__ == "__main__":
    unittest.main()
