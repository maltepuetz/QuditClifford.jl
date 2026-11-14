```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Getting Started

This page is a first session with the package: build a state, read its
tableau, apply gates, measure, and compute expectation values and
entanglement. Each step links to the manual page that covers it in full.

## Installation

QuditClifford requires Julia 1.10 or later. Install it and load it with

```julia
using Pkg
Pkg.add("QuditClifford")
using QuditClifford
```

## A first state

A state of `n` qudits of prime dimension `d` is a tableau. Start with two
qutrits (`d = 3`) in the product state ``|00\rangle``:

```jldoctest first-session
julia> state = DestabilizerTableau(3, 2; state = :product)
Destabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 2
    Generators:       m = 2
    Tableau:
     Stabilizers:
      X     Z 
     0 0 | 1 0 | 0
     0 0 | 0 1 | 0
     ---------
     Destabilizers:
      X     Z 
     1 0 | 0 0
     0 1 | 0 0
```

Each row under **Stabilizers** is one generator of the stabilizer group: its
X exponents on qudits `1, …, n`, then its Z exponents, then its phase
exponent. Here the rows are ``Z_1`` and ``Z_2``, the operators that fix
``|00\rangle``. A pure state has `m = n` generators; a mixed state has fewer.
The **Destabilizers** are a dual basis that [`DestabilizerTableau`](@ref)
keeps up to date so that measurements stay fast; [`StabilizerTableau`](@ref)
has the same interface without them and uses less memory.

Other preset states are the maximally mixed state (`state = :mixed`, the
default), product states in other bases (`basis = :X`, or one basis per qudit),
and generalized GHZ states (`state = :ghz`). [Tableaux](manual/tableaux.md)
covers all of them, the choice between the two types, and building a tableau
from a raw matrix.

## Building Pauli operators

A Pauli operator is given by X and Z exponents per qudit: `(x, z)` on qudit
`q` stands for ``X_q^x Z_q^z``, with `X` to the left. Qudit indices are
1-based, and exponents are reduced mod `d` when the operator meets a tableau,
so the same value works at any `d`. Choose the type by the size of the
support:

```jldoctest pauli-construction
julia> SinglePauli(2, 1, 0)                           # (qudit, x, z): X₂
X₂

julia> DoublePauli(1, 0, 1, 2, 0, 2)                  # two (qudit, x, z) triplets
Z₁ Z₂²

julia> NPauli((1, 2, 4), (1, 0, 1), (0, 1, 1))        # qudits, x exponents, z exponents
X₁ Z₂ X₄ Z₄

julia> GeneralPauli([1, 0, 0, 0, 1, 0], 0)           # dense [x₁, …, xₙ, z₁, …, zₙ], phase
X₁ Z₂
```

Every type takes an optional phase exponent as its last argument, which means
``i^k`` for qubits and ``\omega^k = e^{2\pi i k/d}`` otherwise.
[Conventions](@ref) defines the operators, the ordering and the phases
precisely, including why the qubit `Y` is `SinglePauli(q, 1, 1, 1)`.

## Applying gates

[`apply!`](@ref) applies a Clifford gate in place. A Fourier gate on qutrit 1
followed by a `SUM` (the qudit CNOT) entangles the pair:

```jldoctest first-session
julia> apply!(state, Fourier(1)); apply!(state, SUM(1, 2));

julia> entanglement_entropy(state, [1])
1
```

The entropy is measured in units of ``\log d``, so `1` means the two qutrits
are maximally entangled. [Clifford Unitaries](@ref) lists every named gate
with its action, and covers general and random Clifford operators.

## Expectation values

[`expect!`](@ref) returns ``\langle P \rangle`` as a complex number. The pair is
now stabilized by ``Z_1 Z_2^2``, while ``X_1`` alone has no definite value:

```jldoctest first-session
julia> expect!(state, DoublePauli(1, 0, 1, 2, 0, 2))
1.0 + 0.0im

julia> expect!(state, SinglePauli(1, 1, 0))
0.0 + 0.0im
```

## Measuring

[`measure!`](@ref) performs a projective measurement and returns the outcome
`t`, which stands for the eigenvalue ``\omega^t``. Measuring ``Z_1`` on the
entangled pair gives a random outcome; afterwards ``Z_2`` is certain to agree
with it, and the entanglement is gone:

```jldoctest first-session
julia> t = measure!(state, SinglePauli(1, 0, 1));

julia> measure!(state, SinglePauli(2, 0, 1)) == t
true

julia> expect!(state, SinglePauli(1, 0, 1)) ≈ cis(2π * t / 3)
true

julia> entanglement_entropy(state, [1])
0
```

Random outcomes come from the global RNG. Pass an RNG as the first argument,
as in `measure!(rng, state, op)`, to make a simulation reproducible from a
seed; [Random Sampling](@ref) shows how one RNG reproduces a whole circuit.
[Measurements, Expectations and Channels](@ref) covers the three
measurement cases, the channels that forget outcomes, and the integer form
[`expect_int!`](@ref).

## Where next

- The [Examples](@ref) build complete workflows, from
  [A two-qutrit measurement](@ref) to the
  [measurement-only Ising circuit](@ref "A deep measurement-only Ising circuit").
- The [Manual](manual/index.md) covers each topic in depth;
  [Entanglement Measures](@ref) adds negativity and the information measures.
- The [API Reference](api.md) documents every exported function.
