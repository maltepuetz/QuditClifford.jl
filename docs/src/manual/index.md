```@meta
CurrentModule = QuditClifford
```

# Manual

The manual explains the package topic by topic. [Getting Started](../getting-started.md)
is the faster introduction; the [Examples](../examples/qutrit-measurement.md)
show complete workflows; the [API Reference](../api.md) lists every exported
function.

[**Conventions**](conventions.md) defines the operators, orderings and signs
that the rest of the documentation relies on: the Weyl operators ``X`` and
``Z``, phase exponents, the symplectic form, and the direction in which gates
act. Read it before building raw tableaux or interpreting phase exponents.

[**Tableaux**](tableaux.md) covers the two tableau types and when to use each,
preset and raw construction, the contract the generators satisfy, and
copying, comparing and resetting a tableau.

[**Measurements, Expectations and Channels**](measurements.md) covers
projective Pauli measurements and their outcomes, expectation values, and the
channels that discard information: dephasing, trace-out and reset.

[**Clifford Unitaries**](unitaries.md) lists the named gates with their action
on Paulis and on basis states, and covers stored Clifford operators with
conjugation, composition and inversion.

[**Entanglement Measures**](entanglement.md) defines the entropy, the
negativity and the mutual, conditional and tripartite information, all as
whole numbers of ``\log d`` units.

[**Random Sampling**](random.md) draws uniformly random Clifford operators and
stabilizer states, and shows how one seeded RNG reproduces a whole
monitored-circuit trajectory.

[**Inspecting States**](dense-states.md) expands a small tableau into an exact
ket, a state vector or a density matrix, for debugging and understanding.

[**Limitations and Performance**](performance.md) states which dimensions are
supported and how large `d` may be, and how to keep simulation loops fast and
free of allocation.
