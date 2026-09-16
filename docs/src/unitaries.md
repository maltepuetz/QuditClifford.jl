```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Clifford Unitaries

A Clifford unitary maps Paulis to Paulis under conjugation, so it acts on a
stabilizer tableau by transforming each generator's exponent vector through a
symplectic matrix and shifting its phase. [`apply!`](@ref) does this in place.

This page covers the named gates below, [`apply!`](@ref), and named-gate Pauli
conjugation; a general `CliffordOperator` type together with inversion,
composition, and uniform random Clifford sampling are not yet included.

## The gate set

Names are qudit-native. No qubit aliases are exported; the `d = 2` equivalents
are given below for orientation.

| Gate | Action | At `d = 2` |
|---|---|---|
| `Fourier(q)` | `X ↦ Z`, `Z ↦ X⁻¹` | Hadamard |
| `Phase(q)` | `X ↦ XZ`, `Z ↦ Z` | `S` gate, `diag(1, i)` |
| `Multiplier(q, a)` | `\|j⟩ ↦ \|aj⟩` | identity |
| `PauliGate(q, x, z)` | conjugation by `XˣZᶻ` | Pauli conjugation |
| `SUM(c, t, a)` | `\|u,v⟩ ↦ \|u, v+au⟩` | `CNOT` when `a = 1` |
| `CPhase(q₁, q₂, a)` | `\|u,v⟩ ↦ ω^{auv}\|u,v⟩` | `CZ` when `a = 1` |
| `SWAP(q₁, q₂)` | exchange the two qudits | `SWAP` |

`Multiplier` needs a coefficient invertible mod `d`; `SUM` and `CPhase` accept a
coefficient congruent to zero, which is the identity. In `SUM` it is the second
`Z` image that acquires the backward coupling, `Z₂ ↦ Z₁⁻ᵃZ₂`.

## Applying gates

Gates are dimension-agnostic values, like the Pauli types: the dimension comes
from the tableau.

```jldoctest
julia> tab = StabilizerTableau(2, 2; state = :product, basis = :Z);

julia> apply!(tab, Fourier(1));

julia> apply!(tab, SUM(1, 2));

julia> is_pure(tab)
true
```

That circuit prepares a Bell state, so the two qubits are maximally entangled:

```jldoctest
julia> tab = StabilizerTableau(2, 2; state = :product, basis = :Z);

julia> apply!(tab, Fourier(1)); apply!(tab, SUM(1, 2));

julia> entanglement_entropy(tab, [1])
1
```

`apply!` preserves the generator contract, the generator count `m`, and — for a
[`DestabilizerTableau`](@ref) — the dual basis, which the same symplectic map
transforms without any re-orthogonalization.

## Conjugating operators

[`conjugate`](@ref) gives `U P U†` for the same gates. Since a named gate does
not carry a dimension, pass one:

```jldoctest
julia> conjugate(Fourier(1), SinglePauli(1, 1, 0), 1; d = 3)
Z₁
```

At `d = 2` the phase gate sends `X` to `iXZ`, which the phase exponent records:

```jldoctest
julia> conjugate(Phase(1), SinglePauli(1, 1, 0), 1; d = 2)
[phase=1] X₁ Z₁
```

Applying a gate to a state and conjugating an observable by the same gate agree
on expectation values, which is the relation the test suite checks.

## Phase conventions

Phase exponents follow the package convention in
[Representation and Phase Conventions](@ref): `ω^k` with `k mod d` at odd prime
`d`, and `i^k` with `k mod 4` at `d = 2`. The two regimes genuinely differ — the
qudit phase gate `diag(ω^{j(j-1)/2})` degenerates to the identity if you
substitute `d = 2` — so each has its own evaluator internally.
