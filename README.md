# QuditClifford.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://maltepuetz.github.io/QuditClifford.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maltepuetz.github.io/QuditClifford.jl/dev/)
[![Build Status](https://github.com/maltepuetz/QuditClifford.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/maltepuetz/QuditClifford.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/maltepuetz/QuditClifford.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/maltepuetz/QuditClifford.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

QuditClifford provides stabilizer and destabilizer tableaux for
prime-dimensional qudits. It supports qubits (`d = 2`) and odd prime
dimensions, pure and mixed stabilizer states, projective Pauli measurements,
expectation values, canonicalization, purity checks, and stabilizer
entanglement entropy.

The package is deliberately focused on prime-dimensional qudit tableau
operations. [QuantumClifford.jl](https://github.com/QuantumSavory/QuantumClifford.jl)
is a separate, mature toolkit centered on qubit stabilizer states, Clifford
circuits, graph states, and quantum-error-correction workflows.

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

Create a two-qutrit X-basis product state, measure `Z₁Z₂`, and inspect the
post-measurement state:

```julia
using QuditClifford

tab = DestabilizerTableau(3, 2; state=:product, basis=:X)
op = DoublePauli(1, 0, 1, 2, 0, 1)

outcome = measure!(tab, op; outcome=2)  # 2
expect!(tab, op)                        # ω² ≈ -0.5 - 0.866im
is_pure(tab)                            # true
```

Qubits use the same interface. Entropies are returned in log-`d` units:

```julia
bell = DestabilizerTableau(2, 2; state=:ghz)
entanglement_entropy(bell, [1])  # 1, i.e. log(2)
```

`DestabilizerTableau` is the recommended default: its maintained dual basis
makes repeated measurement, membership, and expectation operations efficient.
`StabilizerTableau` uses less memory and can suit storage-heavy or occasional-
query workflows. Both implement the same high-level measurement, expectation,
purity, canonicalization, reset, and entropy interfaces.

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
