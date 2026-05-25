from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable, Iterable, Optional, Protocol, runtime_checkable


ActivationFunction = Callable[[float], float]


@runtime_checkable
class LinkedListNode(Protocol):
    """Protocol for linked-list nodes accepted by Perceptron."""

    next: Optional["LinkedListNode"]


@dataclass
class WeightNode:
    """A simple linked-list node for perceptron weights."""

    value: float
    next: Optional["WeightNode"] = None


class Perceptron:
    """Compute a single perceptron neuron's activated output."""

    def __init__(
        self,
        weights: Optional[LinkedListNode],
        bias: float = 0.0,
        threshold: float = 0.0,
        activation_function: Optional[ActivationFunction] = None,
    ):
        self.weight_head = weights
        self.bias = self._validate_number(bias, "bias")
        self.threshold = self._validate_number(threshold, "threshold")
        self.activation_function = self._validate_activation_function(activation_function)

    @property
    def weights(self) -> Optional[LinkedListNode]:
        return self.weight_head

    @weights.setter
    def weights(self, weights: Optional[LinkedListNode]) -> None:
        self.weight_head = weights

    def predict(self, inputs: Iterable[float]) -> float:
        weighted_sum = self.weighted_sum(inputs)
        return self._validate_number(self.activation_function(weighted_sum), "activation output")

    def dot_product(self, inputs: Iterable[float]) -> float:
        total = 0.0
        input_iterator = iter(inputs)
        current_node = self.weight_head

        while current_node is not None:
            try:
                input_value = next(input_iterator)
            except StopIteration as exc:
                raise ValueError("Not enough input values for the linked list of weights.") from exc

            weight_value = self._node_value(current_node)
            total += weight_value * self._validate_number(input_value, "input")
            current_node = current_node.next

        try:
            next(input_iterator)
        except StopIteration:
            return total

        raise ValueError("Too many input values for the linked list of weights.")

    def weighted_sum(self, inputs: Iterable[float]) -> float:
        return self.dot_product(inputs) + self.bias

    @staticmethod
    def from_weights(
        weights: Iterable[float],
        bias: float = 0.0,
        threshold: float = 0.0,
        activation_function: Optional[ActivationFunction] = None,
    ) -> "Perceptron":
        head: Optional[WeightNode] = None

        for weight in reversed(list(weights)):
            head = WeightNode(Perceptron._validate_number(weight, "weight"), head)

        return Perceptron(head, bias=bias, threshold=threshold, activation_function=activation_function)

    @staticmethod
    def _node_value(node: LinkedListNode) -> float:
        if hasattr(node, "value"):
            return Perceptron._validate_number(getattr(node, "value"), "weight")

        if hasattr(node, "data"):
            return Perceptron._validate_number(getattr(node, "data"), "weight")

        raise TypeError("Weight nodes must expose a numeric 'value' or 'data' attribute.")

    @staticmethod
    def _validate_number(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")

        return float(value)

    def _default_activation_function(self, weighted_sum: float) -> float:
        return 1.0 if weighted_sum >= self.threshold else 0.0

    def _validate_activation_function(
        self, activation_function: Optional[ActivationFunction]
    ) -> ActivationFunction:
        if activation_function is None:
            return self._default_activation_function

        if not callable(activation_function):
            raise TypeError("activation_function must be callable.")

        return activation_function


class PerceptronLayer:
    """Compute the outputs of a layer of perceptrons."""

    def __init__(self, perceptrons: Iterable[Perceptron]):
        self.perceptrons = self._validate_perceptrons(perceptrons)

    def predict(self, inputs: Iterable[float]) -> list[float]:
        input_values = list(inputs)
        return [perceptron.predict(input_values) for perceptron in self.perceptrons]

    @staticmethod
    def _validate_perceptrons(perceptrons: Iterable[Perceptron]) -> list[Perceptron]:
        if perceptrons is None:
            raise TypeError("perceptrons must be an iterable of Perceptron instances.")

        perceptron_list = list(perceptrons)

        for perceptron in perceptron_list:
            if not isinstance(perceptron, Perceptron):
                raise TypeError("perceptrons must contain only Perceptron instances.")

        return perceptron_list
