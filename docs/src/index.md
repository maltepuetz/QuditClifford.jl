```@meta
CurrentModule = QuditClifford
DocTestSetup = quote
    using QuditClifford
    using Random
    Random.seed!(4)
end
```

# QuditClifford.jl

QuditClifford provides stabilizer-tableau tools for prime-dimensional qudits.
It includes qubits (`d = 2`) and odd prime dimensions, and supports pure and
mixed stabilizer states.

The core capabilities are:

- Stabilizer and dual destabilizer tableau representations.
- Preset mixed, product, and generalized GHZ states.
- Projective measurement of dense and few-qudit Pauli operators.
- Pauli expectation values, canonicalization, and purity checks.
- Stabilizer entanglement entropy for pure and mixed states.

## Quick start

Once QuditClifford is registered in Julia's General registry, install it with:

```julia
using Pkg
Pkg.add("QuditClifford")
```

Before registration, install it from the public repository instead:

```julia
using Pkg
Pkg.add(url="https://github.com/maltepuetz/QuditClifford.jl")
```

Then load the public API:

```julia
using QuditClifford
```

This example follows a generalized qutrit Bell state through a local
measurement. The random seed is fixed in the hidden doctest setup, so the
measurement remains genuinely sampled while the displayed result is
reproducible.

```jldoctest index-quick-start
# Construct a qutrit Bell state and inspect the entanglement between its sites.
# d = 3 selects qutrits; n = 2 creates two sites; :ghz gives their Bell state.
julia> state = DestabilizerTableau(3, 2; state=:ghz)
Destabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 2
    Generators:       m = 2
    Tableau:
     Stabilizers:
      X     Z
     1 1 | 0 0 | 0
     0 0 | 1 2 | 0
     ---------
     Destabilizers:
      X     Z
     0 0 | 2 0
     1 0 | 0 0

julia> entanglement_entropy(state, [1]) # Entropy of the subsystem containing qutrit 1.
1
```

```jldoctest index-quick-start
# Construct X₁ and calculate its expectation before measurement.
# X₁ is not fixed by the Bell stabilizers, so the expectation is zero.
julia> x1 = SinglePauli(1, 1, 0) # Qudit 1 with X exponent 1 and Z exponent 0.
X₁

julia> expect!(state, x1)
0.0 + 0.0im
```

```jldoctest index-quick-start
# X₁ fails to commute with one stabilizer, so measuring it updates the tableau.
# The outcome is sampled; a hidden fixed seed keeps the example reproducible.
julia> measure!(state, x1)
2
```

```jldoctest index-quick-start
# measure! mutates state; display it again to inspect the updated generators.
julia> state
Destabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 2
    Generators:       m = 2
    Tableau:
     Stabilizers:
      X     Z
     1 1 | 0 0 | 0
     1 0 | 0 0 | 1
     ---------
     Destabilizers:
      X     Z
     0 0 | 0 2
     0 0 | 2 1
```

```jldoctest index-quick-start
# Projection fixes X₁ to eigenvalue ω², changing its expectation from 0 to ω².
julia> expect!(state, x1)
-0.5000000000000004 - 0.8660254037844385im
```

```jldoctest index-quick-start
# The local measurement separates the pair, so qutrit 1's entropy falls to zero.
julia> entanglement_entropy(state, [1])
0
```

Entropies are returned in log-`d` units, so the initial value `1` corresponds
to ``\log(3)`` for this maximally entangled qutrit pair. Outcome `2` denotes
the eigenvalue ``\omega^2``. Since ``X_1`` does not commute with the original
``Z_1 Z_2^2`` stabilizer, [`measure!`](@ref) replaces that generator with one
encoding the sampled eigenvalue and updates the dual destabilizers. The local
projection also removes the entanglement between the two qutrits.

Continue with [Getting Started](@ref) for state and operator construction,
explore complete workflows in [Examples](@ref), and read
[Representation and Phase Conventions](@ref) before constructing raw tableaux
or interpreting phase exponents.

## Manual

```@contents
Pages = [
    "getting-started.md",
    "examples.md",
    "conventions.md",
    "measurements.md",
    "api.md",
]
Depth = 2
```
