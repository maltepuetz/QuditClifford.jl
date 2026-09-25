```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Clifford Unitaries

A Clifford unitary maps Paulis to Paulis under conjugation, so it acts on a
stabilizer tableau by transforming each generator's exponent vector through a
symplectic matrix and shifting its phase. [`apply!`](@ref) does this in place.

This page covers named gates, stored [`CliffordOperator`](@ref) values,
tableau application, Pauli conjugation, inversion and composition. Uniform
random Clifford sampling is not yet included.

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

Named gates are dimension-agnostic values, like the Pauli types: the dimension
comes from the tableau.

```jldoctest clifford-bell
julia> tab = StabilizerTableau(2, 2; state = :product, basis = :Z);

julia> apply!(tab, Fourier(1));

julia> apply!(tab, SUM(1, 2));

julia> tab
Stabilizer Tableau:
    Qudit dimension:  d = 2
    Number of Qudits: n = 2
    Generators:       m = 2
    Tableau:
      X     Z 
     1 1 | 0 0 | 0
     0 0 | 1 1 | 0
```

That circuit prepares a Bell state, so the two qubits are maximally entangled:

```jldoctest clifford-bell
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

## Stored Clifford operators

The named gates are dimension-agnostic values. A [`CliffordOperator`](@ref) is
the general alternative: it stores an arbitrary Clifford on an ordered support
at a fixed dimension, as the conjugation images of the ordered generators
`X_t1 … X_tk, Z_t1 … Z_tk`. Column `i` of `F` is the image of generator `i`;
`a[i]` is that image's raw phase exponent.

```jldoctest
julia> U = CliffordOperator(Fourier(1), 3);

julia> U.d, U.targets
(3, [1])

julia> tab = StabilizerTableau(3, 2; state=:product, basis=:Z);

julia> apply!(tab, U) === tab
true
```

Operators compose and invert. `U ∘ V` applies `V` first and requires equal
dimensions and identically ordered targets. Here the composite `U = Phase(1) ∘
Fourier(1)` has raw phases `(0, 1)`, and `inv(U)` recovers raw phases `(1, 0)`:

```jldoctest
julia> U = CliffordOperator(Phase(1), 2) ∘ CliffordOperator(Fourier(1), 2);

julia> U.a
2-element Vector{Int64}:
 0
 1

julia> inv(U).a
2-element Vector{Int64}:
 1
 0
```

Heisenberg evolution of an observable under state evolution by `U` is
`conjugate(inv(U), op)`:

```jldoctest
julia> U = CliffordOperator(Fourier(1), 2);

julia> conjugate(inv(U), GeneralPauli([1, 0], 0))
Z₁
```

Raw construction validates the symplectic condition, and at `d = 2` also the
Hermiticity parity `a[i] ≡ x(F[:,i])·z(F[:,i]) (mod 2)`. `check=false` skips
those two algebraic checks only — shapes, normalization and independent storage
still happen — and makes their preconditions the caller's responsibility.
Nothing downstream re-derives them. Inputs must use one-based indexing;
one-based views, transposes and ranges are accepted and copied into owned dense
storage. Targets remain in the supplied order. For example:

```jldoctest
julia> U = CliffordOperator(2, [3], [0 1; 1 0], [0, 0]);

julia> conjugate(U, SinglePauli(3, 1, 0), 3).xz == [0, 0, 0, 0, 0, 1]
true
```

Stored conjugation infers `d`; an explicit keyword must match. Empty support
represents identity, preserves `iscanonical` under application, and still
returns a fresh normalized Pauli under conjugation:

```jldoctest
julia> E = CliffordOperator(3, Int[], zeros(Int, 0, 0), Int[]);

julia> p = GeneralPauli([-1, 4], -1);

julia> q = conjugate(E, p);

julia> q.xz == [2, 1] && q.phase == 2 && q.xz !== p.xz
true
```

Treat operator fields as read-only. Direct mutation of `targets`, `F`, `a`, or
derived data is unsupported because application trusts the constructor's
invariants and cached context. Construct a new `CliffordOperator` to change its
action.

An operator borrows its own scratch during `apply!`, so one operator must not be
applied concurrently from several tasks. Copies and materialized operators own
independent scratch. `inv`, `∘` and `conjugate` allocate and never touch their
operands' data or scratch.
