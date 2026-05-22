from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class LinkedListNode(Protocol):
    """Protocol for linked-list nodes accepted by PerceptronService."""

    next: Optional["LinkedListNode"]


@dataclass
class WeightNode:
    """A simple linked-list node for perceptron weights."""

    value: float
    next: Optional["WeightNode"] = None


class PerceptronService:
    """Service for computing a single perceptron neuron's binary output."""

    def __init__(self, weights: Optional[LinkedListNode], bias: float = 0.0, threshold: float = 0.0):
        self.weights = weights
        self.bias = self._validate_number(bias, "bias")
        self.threshold = self._validate_number(threshold, "threshold")

    def predict(self, inputs: Iterable[float]) -> int:
        weighted_sum = self.weighted_sum(inputs)
        return 1 if weighted_sum >= self.threshold else 0

    def weighted_sum(self, inputs: Iterable[float]) -> float:
        total = self.bias
        input_iterator = iter(inputs)
        current_weight = self.weights

        while current_weight is not None:
            try:
                input_value = next(input_iterator)
            except StopIteration as exc:
                raise ValueError("Not enough input values for the linked list of weights.") from exc

            weight_value = self._node_value(current_weight)
            total += weight_value * self._validate_number(input_value, "input")
            current_weight = current_weight.next

        try:
            next(input_iterator)
        except StopIteration:
            return total

        raise ValueError("Too many input values for the linked list of weights.")

    @staticmethod
    def from_weights(weights: Iterable[float], bias: float = 0.0, threshold: float = 0.0) -> "PerceptronService":
        head: Optional[WeightNode] = None

        for weight in reversed(list(weights)):
            head = WeightNode(PerceptronService._validate_number(weight, "weight"), head)

        return PerceptronService(head, bias=bias, threshold=threshold)

    @staticmethod
    def _node_value(node: LinkedListNode) -> float:
        if hasattr(node, "value"):
            return PerceptronService._validate_number(getattr(node, "value"), "weight")

        if hasattr(node, "data"):
            return PerceptronService._validate_number(getattr(node, "data"), "weight")

        raise TypeError("Weight nodes must expose a numeric 'value' or 'data' attribute.")

    @staticmethod
    def _validate_number(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")

        return float(value)
