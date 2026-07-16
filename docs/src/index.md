```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "QuditClifford.jl"
  text: "Stabilizer tableaux for prime-dimensional qudits"
  tagline: Simulate qubits and qudits with measurements, expectations, and entanglement workflows in pure Julia.
  actions:
    - theme: brand
      text: Getting Started
      link: /getting-started
    - theme: alt
      text: Examples
      link: /examples
    - theme: alt
      text: API Reference
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/maltepuetz/QuditClifford.jl

features:
  - icon: "⚛️"
    title: Qubits and qudits
    details: One interface for qubits and odd prime dimensions, with pure and mixed stabilizer states.
    link: /getting-started
  - icon: "⚡"
    title: Measurement-ready tableaux
    details: Efficient projective Pauli measurements, expectation values, canonicalization, and purity checks.
  - icon: "📈"
    title: Entanglement workflows
    details: Track stabilizer entanglement entropy through reproducible circuits and measurement trajectories.
---
```

```@meta
CurrentModule = QuditClifford
DocTestSetup = quote
    using QuditClifford
    using Random
    Random.seed!(4)
end
```

```@raw html
<div style="height: 3rem" aria-hidden="true"></div>
```

# What is QuditClifford.jl?

QuditClifford provides stabilizer-tableau tools for prime-dimensional qudits,
including projective Pauli measurements, expectation values, canonicalization,
purity checks, and entanglement-entropy calculations. It supports qubits
(`d = 2`) and odd prime dimensions (`d = 3, 5, 7, ...`), as well as pure and
mixed stabilizer states.

!!! note "Development status"
    QuditClifford.jl is under active development and has not yet reached a
    stable release. Additional functionality is planned, including support
    for Clifford unitaries. APIs may change before version 1.0.

Use [`DestabilizerTableau`](@ref) for repeated measurement and expectation
workflows, or [`StabilizerTableau`](@ref) when compact storage is more
important than maintaining a dual basis.

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
measurement.

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
