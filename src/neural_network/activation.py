"""Activation functions bundled with their derivatives.

An :class:`Activation` pairs an elementwise function ``fn`` with its
derivative ``derivative`` (``d fn / d z``). The layer's forward pass uses
``fn``; ``derivative`` exists for the training/backprop work to come.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayLike = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Activation:
    """An elementwise activation and its derivative.

    Both ``fn`` and ``derivative`` operate elementwise and are array-aware,
    so they work on a single vector ``(n_out,)`` or a batch ``(m, n_out)``.
    """

    name: str
    fn: ArrayLike
    derivative: ArrayLike


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# Squashes any number into the range 0..1, so the output reads like a
# probability or a soft "off/on" switch.
sigmoid = Activation(
    name="sigmoid",
    fn=_sigmoid,
    derivative=lambda z: _sigmoid(z) * (1.0 - _sigmoid(z)),
)

# Passes positive numbers through unchanged and flattens negatives to zero,
# which is cheap to compute and the usual default for hidden layers.
relu = Activation(
    name="relu",
    fn=lambda z: np.maximum(0.0, z),
    derivative=lambda z: np.where(np.asarray(z) > 0, 1.0, 0.0),
)

# Like sigmoid but squashes into -1..1, so outputs are centred on zero and
# can express "negative" as well as "positive" signals.
tanh = Activation(
    name="tanh",
    fn=lambda z: np.tanh(z),
    derivative=lambda z: 1.0 - np.tanh(z) ** 2,
)

# Leaves the number exactly as it is — the "no activation" option, used when a
# layer should output any value at all (e.g. predicting a price).
identity = Activation(
    name="identity",
    fn=lambda z: z,
    derivative=lambda z: np.ones_like(z, dtype=float),
)
