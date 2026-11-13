```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

# Search-result title and snippet: without these the tab title comes from the
# first heading ("What is QuditClifford.jl?") and never spells out "qudit
# Clifford" as separate words, which search engines do not split out of the
# package name.
title: "Qudit Clifford circuit simulation in Julia"
description: "QuditClifford.jl simulates qudit and qubit Clifford circuits in Julia: stabilizer tableaux for prime dimensions, random Cliffords, Pauli measurements, expectation values and entanglement entropy."

hero:
  name: "QuditClifford.jl"
  text: "Stabilizer tableaux for prime-dimensional qudits"
  tagline: Simulate qubit and qudit Clifford circuits with measurements, expectations, and entanglement workflows in pure Julia.
  image:
    light: /logo-light.svg
    dark: /logo-dark.svg
    alt: QuditClifford.jl
  actions:
    - theme: brand
      text: Getting Started
      link: /getting-started
    - theme: alt
      text: Manual
      link: /manual/
    - theme: alt
      text: Examples
      link: /examples/qutrit-measurement
    - theme: alt
      text: API Reference
      link: /api

features:
  - title: Any prime dimension
    details: Qubits, qutrits and every prime d through one interface, for pure and mixed stabilizer states.
    link: /getting-started
    linkText: Get started
  - title: Measurements
    details: Projective Pauli measurements, exact expectation values, and dephasing, trace-out and reset channels.
    link: /manual/measurements
    linkText: Measurements
  - title: Clifford unitaries
    details: Named gates for every prime d, general Clifford operators with composition and inversion, and uniformly random Cliffords and states.
    link: /manual/unitaries
    linkText: Unitaries
  - title: Entanglement
    details: Entropy, negativity, and mutual, conditional and tripartite information as exact integers in log d units.
    link: /manual/entanglement
    linkText: Entanglement
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

QuditClifford.jl simulates qubit and qudit Clifford circuits with stabilizer
tableaux, for any prime local dimension `d` and for pure and mixed stabilizer
states. It covers Clifford unitaries and uniformly random Cliffords, projective
Pauli measurements, exact expectation values, dephasing, trace-out and reset
channels, entanglement entropy, negativity and the information measures, and
exact or dense expansions of small states for debugging.

!!! note "Development status"
    QuditClifford.jl is under active development and has not yet reached a
    stable release. APIs may change before version 1.0.

## Quick start

Install QuditClifford with

```julia
using Pkg
Pkg.add("QuditClifford")
```

then prepare an entangled pair of qutrits, measure one of them, and watch the
entanglement disappear:

```jldoctest index-quick-start
julia> using QuditClifford

julia> state = DestabilizerTableau(3, 2; state = :product);  # two qutrits in |00⟩

julia> apply!(state, Fourier(1)); apply!(state, SUM(1, 2));  # entangle them

julia> entanglement_entropy(state, [1])  # in units of log 3
1

julia> t = measure!(state, SinglePauli(1, 0, 1));  # Z on qutrit 1: random t ∈ 0:2

julia> measure!(state, SinglePauli(2, 0, 1)) == t  # qutrit 2 now gives the same outcome
true

julia> entanglement_entropy(state, [1])
0
```

[Getting Started](@ref) walks through states, Pauli operators, gates and
measurements step by step; the [Manual](manual/index.md) and the
[Examples](@ref) go further.

## Manual

```@contents
Pages = [
    "getting-started.md",
    "manual/conventions.md",
    "manual/tableaux.md",
    "manual/measurements.md",
    "manual/unitaries.md",
    "manual/entanglement.md",
    "manual/random.md",
    "manual/dense-states.md",
    "manual/performance.md",
    "api.md",
]
Depth = 2
```

## Examples

```@contents
Pages = [
    "examples/qutrit-measurement.md",
    "examples/ghz-measurement.md",
    "examples/entropy-cuts.md",
    "examples/mixed-correlations.md",
    "examples/purification.md",
    "examples/boundary-dephasing.md",
    "examples/measurement-only-ising.md",
]
Depth = 1
```
