# QuditClifford.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://maltepuetz.github.io/QuditClifford.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maltepuetz.github.io/QuditClifford.jl/dev/)
[![Build Status](https://github.com/maltepuetz/QuditClifford.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/maltepuetz/QuditClifford.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/maltepuetz/QuditClifford.jl/graph/badge.svg?token=6I4UJ47VOH)](https://codecov.io/gh/maltepuetz/QuditClifford.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

QuditClifford provides stabilizer-tableau tools for prime-dimensional qudits,
including projective Pauli measurements, expectation values, canonicalization,
purity checks, and entanglement-entropy calculations. It supports qubits
(`d = 2`) and odd prime dimensions (`d = 3, 5, 7, ...`), as well as pure and
mixed stabilizer states.

> **Development status:** QuditClifford.jl is under active development and has
> not yet reached a stable release. APIs may change before version 1.0, and
> planned functionality includes support for Clifford unitaries.

Use `DestabilizerTableau` for repeated measurement and expectation workflows,
or `StabilizerTableau` when compact storage is more important than maintaining
a dual basis.

## Installation

After registration in Julia's General registry:

```julia
using Pkg
Pkg.add("QuditClifford")
```

Until registration is complete, install the public repository directly:

```julia
using Pkg
Pkg.add(url="https://github.com/maltepuetz/QuditClifford.jl")
```

QuditClifford requires Julia 1.10 or later.

## Quick start

This example follows a generalized qutrit Bell state through a local
measurement.

```julia
julia> using QuditClifford

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

```julia
# Construct X₁ and calculate its expectation before measurement.
# X₁ is not fixed by the Bell stabilizers, so the expectation is zero.
julia> x1 = SinglePauli(1, 1, 0) # Qudit 1 with X exponent 1 and Z exponent 0.
X₁

julia> expect!(state, x1)
0.0 + 0.0im
```

```julia
# X₁ fails to commute with one stabilizer, so measuring it updates the tableau.
julia> measure!(state, x1)
2
```

```julia
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

```julia
# Projection fixes X₁ to eigenvalue ω², changing its expectation from 0 to ω².
julia> expect!(state, x1)
-0.5000000000000004 - 0.8660254037844385im
```

```julia
# The local measurement separates the pair, so qutrit 1's entropy falls to zero.
julia> entanglement_entropy(state, [1])
0
```

Entropies are returned in log-`d` units, so the initial value `1` corresponds
to `log(3)` for this maximally entangled qutrit pair. Outcome `2` denotes the
eigenvalue `ω²`. Since `X₁` does not commute with the original `Z₁Z₂²`
stabilizer, `measure!` replaces that generator with one encoding the sampled
eigenvalue and updates the dual destabilizers. The local projection also
removes the entanglement between the two qutrits.

Continue with [Getting Started](https://maltepuetz.github.io/QuditClifford.jl/dev/getting-started)
for state and operator construction, explore complete workflows in
[Examples](https://maltepuetz.github.io/QuditClifford.jl/dev/examples), and
read [Representation and Phase Conventions](https://maltepuetz.github.io/QuditClifford.jl/dev/conventions)
before constructing raw tableaux or interpreting phase exponents.

## Documentation and support

- [Stable documentation](https://maltepuetz.github.io/QuditClifford.jl/stable/)
- [Development documentation](https://maltepuetz.github.io/QuditClifford.jl/dev/)
- [Issue tracker](https://github.com/maltepuetz/QuditClifford.jl/issues)
- [Changelog](CHANGELOG.md)

When using QuditClifford in research, cite the repository or archived release
and include the package version used. Formal paper or DOI metadata is not yet
available.

QuditClifford is maintained by Malte Pütz and distributed under the
[MIT License](LICENSE).
