# Note: Why backprop uses `Wᵀ · delta`

**Date:** 2026-07-05
**Context:** Follow-up explainer for the "core operation" mentioned in
[`2026-07-05-numpy-layer-representation-design.md`](../specs/2026-07-05-numpy-layer-representation-design.md#motivation),
for whoever implements the deferred training/backprop spec.

## Forward pass, as wires

Picture each input as having a wire to every output, and each wire has a
number stamped on it — the weight:

```
input1 ──(w11)──► output1
input1 ──(w21)──► output2
input1 ──(w31)──► output3
```

Forward, a signal leaves `input1`, travels down each wire, and gets
multiplied by that wire's weight on arrival:

```
output_i = Σ_j w_ij * input_j
```

## Backward pass, same wires, reversed

After a prediction, each output has an error signal `delta_i`
(`∂Loss/∂z_i`). To train the *previous* layer, we need to know how much each
*input* should be blamed for that error.

Since `input_j` only ever influenced the outputs through its own wires
(`w_1j`, `w_2j`, `w_3j`, ...), the fair share of blame is the weighted sum of
the deltas flowing back through exactly those wires:

```
blame_j = Σ_i w_ij * delta_i
```

That sum, for every input at once, is exactly `Wᵀ @ delta`. It's not a new
formula — it's the same weight values used in the forward pass, just
multiplying signal flowing in the opposite direction down the same
connections. `Wᵀ` is just "read the weight table by input instead of by
output" so the whole backward pass is one matrix multiply instead of manually
retracing wires.

This is why the NumPy matrix representation matters: `W.T @ delta` is a
single vectorized line, whereas the old linked-list-of-neurons representation
would need a hand-rolled multi-cursor traversal to do the same thing.
