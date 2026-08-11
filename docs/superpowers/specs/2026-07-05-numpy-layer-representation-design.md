# Design: NumPy-backed `Layer`

**Date:** 2026-07-05
**Status:** Approved (pending spec review)
**Scope:** Replace the linked-list weight representation with a NumPy weight
matrix and a vectorized forward pass. Introduce a derivative-aware `Activation`
seam. Training and layer-chaining are explicitly out of scope (separate spec).

## Motivation

The current `Perceptron` stores weights as a singly-linked list of `WeightNode`
objects, and `PerceptronLayer` is a `list[Perceptron]`. Training via
backpropagation is planned in the near future, and the linked-list
representation fights it: backprop's core operation is the transpose product
`Wᵀ · delta` across a whole layer's weights, which is one cache-friendly NumPy
line over a matrix but a hand-rolled multi-cursor traversal over a list of
linked lists. It also blocks vectorization and per-weight optimizer state.

Moving weights to a NumPy matrix makes the forward pass (and later, backprop)
readable and fast, and collapses the layer to its true shape: **a weight matrix
+ a bias vector + an activation**.

## Decisions

These were settled during brainstorming and are fixed for this spec:

1. **`Layer` is the atom.** `Perceptron` and `WeightNode` are deleted. There is
   no single-neuron object; a "neuron" is a row of the weight matrix. The layer
   owns the matrix and is the sole unit of computation.
2. **Familiar construction is preserved** via `Layer.from_neurons([...])`, which
   builds the matrix from a list of per-neuron weight rows — so nothing is lost
   ergonomically by dropping `Perceptron`.
3. **Activations become derivative-aware now.** A small `Activation` value object
   bundles `fn` and `derivative` (both array-aware). The `derivative` is unused
   this spec but exists so the training spec does not have to reopen the API. The
   un-trainable step-function default is retired.
4. **Scope is representation + forward pass only.** `Network` (layer chaining),
   `fit`/backprop, optimizers, and loss functions are deferred to a follow-up
   spec.
5. **`numpy` is added** as a project dependency (there is none today). Default
   dtype is `float64`.
6. **Forward pass supports batches** (single vector and a stack of vectors).
7. **`activation` is a required argument** — no default activation.
8. **Breaking API change accepted.** No deprecation shim; the package is at
   0.1.0 and the break is the point.

## Module & file structure

```
src/neural_network/
  layer.py        # Layer (weight matrix + bias vector + activation)
  activation.py   # Activation value object + sigmoid / relu / tanh / identity
  __init__.py     # exports Layer, Activation, sigmoid, relu, tanh, identity
```

- `perceptron.py` is deleted.
- `activation.py` is a separate module because activations are a self-contained
  concern the layer merely consumes.

## Components

### `Activation` (activation.py)

A frozen dataclass bundling an elementwise function and its derivative:

```python
@dataclass(frozen=True)
class Activation:
    name: str
    fn:         Callable[[np.ndarray], np.ndarray]   # elementwise
    derivative: Callable[[np.ndarray], np.ndarray]   # d fn / d z (dormant this spec)
```

Module-level instances shipped:

| name      | `fn(z)`                 | `derivative(z)`                     |
|-----------|-------------------------|-------------------------------------|
| `sigmoid` | `1 / (1 + exp(-z))`     | `s * (1 - s)` where `s = sigmoid(z)`|
| `relu`    | `maximum(0, z)`         | `where(z > 0, 1.0, 0.0)`            |
| `tanh`    | `tanh(z)`               | `1 - tanh(z) ** 2`                  |
| `identity`| `z`                     | `ones_like(z)`                      |

`derivative` is not exercised by any forward-pass code in this spec; it is
covered by unit tests so the training spec can rely on it.

### `Layer` (layer.py)

```python
class Layer:
    W: np.ndarray            # shape (n_out, n_in), float64
    b: np.ndarray            # shape (n_out,),      float64
    activation: Activation

    def __init__(self, W, b, activation): ...
    @classmethod
    def from_neurons(cls, rows, bias, activation): ...
    def predict(self, x): ...
```

**Construction / validation**

- `__init__` coerces `W` and `b` to `float64` `np.ndarray`. Validates:
  - `W` is 2-D.
  - `b` is 1-D and `len(b) == W.shape[0]`.
  - `activation` is an `Activation` instance.
  - Raises `ValueError` (shape mismatch) / `TypeError` (wrong activation type)
    with a clear message.
- `from_neurons(rows, bias, activation)`:
  - `rows` is an iterable of equal-length weight sequences (`n_out × n_in`);
    non-rectangular input raises `ValueError`.
  - `bias` is an iterable of length `n_out`.
  - Delegates to `__init__` after assembling the matrix.

**Forward pass**

```python
def predict(self, x):
    x = np.asarray(x, dtype=float)
    z = x @ self.W.T + self.b        # (n_in,) -> (n_out,);  (m, n_in) -> (m, n_out)
    return self.activation.fn(z)
```

- Single vector `x` of shape `(n_in,)` returns `(n_out,)`.
- Batch `X` of shape `(m, n_in)` returns `(m, n_out)`.
- The trailing dimension of `x` must equal `W.shape[1]`; otherwise a clear
  `ValueError` is raised (preserving the intent of the old input-count-mismatch
  behavior, expressed as a shape check). The layer validates this explicitly
  rather than surfacing NumPy's raw matmul error.

## Data flow

```
inputs (n_in,) or (m, n_in)
      │
      ▼
 x @ W.T + b        # vectorized affine transform
      │
      ▼
 activation.fn      # elementwise
      │
      ▼
outputs (n_out,) or (m, n_out)
```

## Error handling

| Condition                                   | Error        |
|---------------------------------------------|--------------|
| `W` not 2-D                                 | `ValueError` |
| `b` not 1-D or `len(b) != W.shape[0]`       | `ValueError` |
| `from_neurons` rows non-rectangular         | `ValueError` |
| `predict` input trailing dim != `W.shape[1]`| `ValueError` |
| `activation` not an `Activation`            | `TypeError`  |

## Public API (`__init__.py`)

Remove: `Perceptron`, `PerceptronLayer`, `WeightNode`.
Add: `Layer`, `Activation`, `sigmoid`, `relu`, `tanh`, `identity`.

## Testing

Delete `tests/test_perceptron.py` and `tests/test_perceptron_layer.py`. Add:

- `tests/test_layer.py`
  - forward correctness for a single vector vs. hand-computed `W @ x + b`
    through the activation;
  - batch forward correctness (shape `(m, n_out)` and values);
  - `from_neurons` builds the expected matrix and bias;
  - shape-validation errors: bad `b` length, non-rectangular rows, wrong input
    width, non-2-D `W`, non-`Activation` activation.
- `tests/test_activation.py`
  - `fn` values at known points (e.g. `sigmoid(0) == 0.5`, `tanh(0) == 0`,
    `relu(-1) == 0`, `identity(z) == z`);
  - `derivative` values at known points (e.g. `relu'(-1) == 0`, `relu'(2) == 1`,
    `sigmoid'(0) == 0.25`, `identity'(z) == 1`);
  - each activation operates elementwise over an array input.

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = ["numpy"]
```

Default weight/bias dtype: `float64`.

## Deferred to follow-up spec

- `Network` — chaining layers so each layer's outputs feed the next layer's
  inputs.
- Training: `fit`, backpropagation, optimizers, loss functions.

The forward-looking hooks built now — `Activation.derivative` and the
batch-shaped `predict` — are the only concessions to that future work; no
training logic is implemented in this spec.
