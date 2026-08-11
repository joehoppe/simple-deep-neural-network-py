"""A fully-connected layer: a weight matrix, a bias vector, and an activation.

A "neuron" is a row - a row in ``W``.
The layer stores its weights as one matrix instead of a list of Perceptron objects.
W[i] is a 1-D vector of length n_in: the complete weight list for neuron i, one weight per input.

W = [[w00, w01, w02],   <- neuron 0's weights on inputs 0,1,2
     [w10, w11, w12]]   <- neuron 1's weights on inputs 0,1,2
b = [b0, b1]            <- one bias per row/neuron

If you demand one row at a time, the library must hand you row 0 before it touches row 1
Computing row 0 sweeps the entire W through the CPU cache; 
computing row 1 sweeps all of W through again. For a large W, 
that's re-reading it from main memory for every single sample.

A "neuron" is a row of ``W``; the layer owns the matrix and is the sole unit
of computation. The forward pass is a single vectorized affine transform
followed by the elementwise activation, and supports both a single input
vector and a batch of vectors.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from neural_network.activation import Activation


class Layer:
    """Weight matrix ``W`` (n_out, n_in) + bias ``b`` (n_out,) + activation."""

    def __init__(self, W, b, activation: Activation):
        if not isinstance(activation, Activation):
            raise TypeError("activation must be an Activation instance.")

        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if W.ndim != 2:
            raise ValueError(f"W must be 2-D (n_out, n_in); got shape {W.shape}.")
        if b.ndim != 1:
            raise ValueError(f"b must be 1-D (n_out,); got shape {b.shape}.")
        if b.shape[0] != W.shape[0]:
            raise ValueError(
                f"b length {b.shape[0]} must equal number of neurons W.shape[0]={W.shape[0]}."
            )

        self.W = W
        self.b = b
        self.activation = activation

    @classmethod
    def from_neurons(
        cls, rows: Iterable[Iterable[float]], bias: Iterable[float], activation: Activation
    ) -> "Layer":
        """Build a layer from per-neuron weight rows and a bias vector."""
        row_list = [list(row) for row in rows]

        if row_list:
            width = len(row_list[0])
            if any(len(row) != width for row in row_list):
                raise ValueError("from_neurons rows must all be the same length.")

        try:
            W = np.asarray(row_list, dtype=np.float64)
        except ValueError as exc:  # ragged input numpy cannot rectangularize
            raise ValueError("from_neurons rows must be rectangular.") from exc

        if W.ndim != 2:
            raise ValueError("from_neurons rows must form a 2-D matrix.")

        return cls(W, np.asarray(list(bias), dtype=np.float64), activation)

    def predict(self, x):
        """Forward pass. ``x`` is ``(n_in,)`` or a batch ``(m, n_in)``."""
        x = np.asarray(x, dtype=np.float64)

        if x.shape[-1] != self.W.shape[1]:
            raise ValueError(
                f"Input trailing dimension {x.shape[-1]} must equal "
                f"number of weights per neuron W.shape[1]={self.W.shape[1]}."
            )

        z = x @ self.W.T + self.b
        return self.activation.fn(z)
