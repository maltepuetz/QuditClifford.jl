```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Getting Started

## Installation and loading

Install the registered package with `Pkg.add("QuditClifford")`, or install the
repository directly before registration:

```julia
using Pkg
Pkg.add(url="https://github.com/maltepuetz/QuditClifford.jl")
```

Then load the public API:

```julia
using QuditClifford
```

QuditClifford requires Julia 1.10 or later.

## Constructing states

Use [`DestabilizerTableau`](@ref) for the usual workflow. Its preset states
cover empty stabilizer information, product states, and generalized GHZ states:

```jldoctest state-construction
# d = 3 selects qutrits; n = 4 creates four sites.
# :mixed gives the maximally mixed state.
julia> mixed = DestabilizerTableau(3, 4; state=:mixed)
Destabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 4
    Generators:       m = 0
    Mixed tableau (m < n generators):
    (no generators; maximally mixed on the full space)

julia> is_pure(mixed) # Test whether the represented state is pure.
false
```

```jldoctest state-construction
# d = 3, n = 4; create an X-basis product state.
julia> product = DestabilizerTableau(3, 4; state=:product, basis=:X)
Destabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 4
    Generators:       m = 4
    Tableau:
     Stabilizers:
     -- X --   -- Z --
     1 0 0 0 | 0 0 0 0 | 0
     0 1 0 0 | 0 0 0 0 | 0
     0 0 1 0 | 0 0 0 0 | 0
     0 0 0 1 | 0 0 0 0 | 0
     -----------------
     Destabilizers:
     -- X --   -- Z --
     0 0 0 0 | 2 0 0 0
     0 0 0 0 | 0 2 0 0
     0 0 0 0 | 0 0 2 0
     0 0 0 0 | 0 0 0 2

julia> is_pure(product) # Product states are pure.
true
```

```jldoctest state-construction
# d = 3, n = 4; create a generalized GHZ state.
julia> ghz = DestabilizerTableau(3, 4; state=:ghz)
Destabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 4
    Generators:       m = 4
    Tableau:
     Stabilizers:
     -- X --   -- Z --
     1 1 1 1 | 0 0 0 0 | 0
     0 0 0 0 | 1 2 0 0 | 0
     0 0 0 0 | 0 1 2 0 | 0
     0 0 0 0 | 0 0 1 2 | 0
     -----------------
     Destabilizers:
     -- X --   -- Z --
     0 0 0 0 | 2 0 0 0
     1 0 0 0 | 0 0 0 0
     1 1 0 0 | 0 0 0 0
     1 1 1 0 | 0 0 0 0
```

```jldoctest state-construction
# Compute the entropy of the subsystem containing qutrits 1 and 2.
julia> entanglement_entropy(ghz, [1, 2])
1
```

For a product state, `basis` can be `:X`, `:Y`, or `:Z`, or a vector/tuple
giving one basis per qudit. The `:Y` preset is available for qubits. The
constructor aliases `state=:X`, `state=:Y`, and `state=:Z` are shorthand for
product states in the corresponding basis.

Pass `storephase=false` only when phase-sensitive outcomes and expectation
values are unimportant. See [Phase exponents](@ref) for the resulting
convention.

## Constructing Pauli operators

Each local Pauli factor is represented by X and Z exponents: `(x, z)` denotes
``X^x Z^z`` on that qudit. Qudit indices are 1-based, and exponents are reduced
modulo the tableau dimension when the operator is used.

Choose the representation by the size of the operator's support:

```jldoctest pauli-construction
julia> SinglePauli(2, 1, 0) # (qudit, X exponent, Z exponent): X₂.
X₂

julia> DoublePauli(1, 0, 1, 3, 0, 1) # Two (qudit, X, Z) triplets: Z₁Z₃.
Z₁ Z₃

julia> TriplePauli(1, 1, 0, 2, 0, 1, 3, 1, 1) # Three (qudit, X, Z) triplets.
X₁ Z₂ X₃ Z₃
```

```jldoctest pauli-construction
# The tuples contain qudit indices, X exponents, and Z exponents.
julia> NPauli((1, 2, 4, 7), (1, 0, 1, 0), (0, 1, 1, 2))
X₁ Z₂ X₄ Z₄ Z₇²
```

The same compact Unicode representation is used in the REPL and other plain
text displays.

[`SinglePauli`](@ref), [`DoublePauli`](@ref), [`TriplePauli`](@ref), and
[`NPauli`](@ref) avoid storing zeros for operators with small support. Their
final optional argument is a phase exponent:

```jldoctest pauli-construction
julia> SinglePauli(1, 0, 1, 2) # (qudit, X, Z, phase): i²Z₁ = -Z₁ for d=2.
[phase=2] Z₁
```

The phase prefix remains dimension-neutral because the same stored exponent is
interpreted as ``i^k`` for qubits and ``\omega^k`` for odd-prime qudits.

Use [`GeneralPauli`](@ref) for dense support. Its vector is ordered as
`[x₁, …, xₙ, z₁, …, zₙ]`:

```jldoctest pauli-construction
# The vector contains all X exponents, then all Z exponents; the final argument is phase.
julia> dense = GeneralPauli([1, 0, 0, 0, 1, 0], 0)
X₁ Z₂
```

Measurement and expectation functions also accept that dense vector directly.
An optional final entry supplies the phase exponent:

```julia
raw = [1, 0, 0, 0, 1, 0, 0]  # six X/Z entries, then phase 0
```

Prefer a few-qudit type inside local circuits; use a dense representation when
the operator naturally acts on many sites.

Continue with [Examples](@ref) for complete measurement and entanglement
workflows, including the measurement-only Ising/percolation circuit.
