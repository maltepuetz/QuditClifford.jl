"""
    AbstractClifford

Abstract supertype for Clifford unitaries supported by QuditClifford.

A Clifford is fixed, up to a global phase, by the conjugation images of the
ordered generators `X₁, …, X_k, Z₁, …, Z_k` on its support. Concrete named
gates store only qudit indices and parameters, and are **dimension-agnostic**
in the same sense as [`SinglePauli`](@ref): the same value denotes different
operators at different `d`, and may be invalid at some of them. Dimension is
supplied when the gate meets a tableau in [`apply!`](@ref), or explicitly in
[`conjugate`](@ref).

# The named gates

The complete set of concrete subtypes. Names are qudit-native; no qubit
aliases are exported, and the `d = 2` column is for orientation only.

| Constructor | Action | At `d = 2` |
|:---|:---|:---|
| [`Fourier(qudit)`](@ref Fourier) | `X ↦ Z`, `Z ↦ X⁻¹` | Hadamard |
| [`Phase(qudit)`](@ref Phase) | `X ↦ XZ`, `Z ↦ Z` | `S` gate, `diag(1, i)` |
| [`Multiplier(qudit, a)`](@ref Multiplier) | `\\|j⟩ ↦ \\|aj⟩`, needs `a` invertible mod `d` | identity |
| [`PauliGate(qudit, x, z)`](@ref PauliGate) | conjugation by `XˣZᶻ` | Pauli conjugation |
| [`SUM(control, target, a = 1)`](@ref SUM) | `\\|u,v⟩ ↦ \\|u, v+au⟩` | `CNOT` when `a = 1` |
| [`CPhase(qudit1, qudit2, a = 1)`](@ref CPhase) | `\\|u,v⟩ ↦ ω^{auv}\\|u,v⟩` | `CZ` when `a = 1` |
| [`SWAP(qudit1, qudit2)`](@ref SWAP) | exchange the two qudits | `SWAP` |

`SUM` and `CPhase` accept a coefficient congruent to zero, which is the
identity; `Multiplier` rejects one, since it would not be invertible. Two-qudit
gates require distinct qudits. A general `CliffordOperator`, along with
composition, inversion and uniform random sampling, is not yet implemented.

# Examples
```julia
g = Fourier(1)
g isa AbstractClifford
```
"""
abstract type AbstractClifford end

@inline function _check_qudit(q::Int)
    q > 0 || throw(ArgumentError("Qudit index must be positive, got $q."))
    return q
end

@inline function _check_pair(q1::Int, q2::Int)
    _check_qudit(q1); _check_qudit(q2)
    q1 == q2 && throw(ArgumentError(
        "A two-qudit gate needs distinct qudits, got $q1 and $q2."))
    return nothing
end

"""
    Fourier(qudit::Int)

Fourier gate on one qudit: `X ↦ Z`, `Z ↦ X^{-1}`.

At `d = 2` this is the Hadamard gate.

# Examples
```julia
apply!(tab, Fourier(1))
```
"""
struct Fourier <: AbstractClifford
    qudit::Int
    Fourier(qudit::Int) = new(_check_qudit(qudit))
end

"""
    Phase(qudit::Int)

Quadratic phase gate on one qudit: `X ↦ XZ`, `Z ↦ Z`.

For odd prime `d` this is `diag(ω^{j(j-1)/2})`; at `d = 2` it is `diag(1, i)`,
the qubit `S` gate, and its `X` image carries phase exponent 1.

# Examples
```julia
apply!(tab, Phase(1))
```
"""
struct Phase <: AbstractClifford
    qudit::Int
    Phase(qudit::Int) = new(_check_qudit(qudit))
end

"""
    Multiplier(qudit::Int, a::Int)

Multiplier gate on one qudit: `X ↦ X^a`, `Z ↦ Z^{a^{-1}}`, acting as
`|j⟩ ↦ |aj⟩`.

`a` must be invertible modulo the dimension it is used at; a residue congruent
to zero throws `ArgumentError` when the gate is applied. At `d = 2` the only
valid coefficient is odd, and the gate is the identity.

# Examples
```julia
apply!(tab, Multiplier(1, 2))   # valid for d ≥ 3
```
"""
struct Multiplier <: AbstractClifford
    qudit::Int
    a::Int
    Multiplier(qudit::Int, a::Int) = new(_check_qudit(qudit), a)
end

"""
    PauliGate(qudit::Int, x::Int, z::Int)

Conjugation by the Pauli `X^x Z^z` on one qudit, up to a global phase.

Conjugation leaves every exponent unchanged and shifts phases by `ω^{zx' - xz'}`
on a Pauli with exponents `(x', z')`.

# Examples
```julia
apply!(tab, PauliGate(2, 1, 0))   # conjugate by X on qudit 2
```
"""
struct PauliGate <: AbstractClifford
    qudit::Int
    x::Int
    z::Int
    PauliGate(qudit::Int, x::Int, z::Int) = new(_check_qudit(qudit), x, z)
end

"""
    SUM(control::Int, target::Int, a::Int = 1)

Generalized controlled addition, `|u, v⟩ ↦ |u, v + a·u⟩`.

Images are `X₁ ↦ X₁X₂^a`, `X₂ ↦ X₂`, `Z₁ ↦ Z₁`, `Z₂ ↦ Z₁^{-a}Z₂`: the second
`Z` image is the one that acquires the backward coupling. At `d = 2` with
`a = 1` this is `CNOT`. A coefficient congruent to zero is a valid identity.

# Examples
```julia
apply!(tab, SUM(1, 3))
```
"""
struct SUM <: AbstractClifford
    control::Int
    target::Int
    a::Int
    function SUM(control::Int, target::Int, a::Int)
        _check_pair(control, target)
        return new(control, target, a)
    end
end
SUM(control::Int, target::Int) = SUM(control, target, 1)

"""
    CPhase(qudit1::Int, qudit2::Int, a::Int = 1)

Controlled phase, `|u, v⟩ ↦ ω^{a·u·v}|u, v⟩`.

Images are `X₁ ↦ X₁Z₂^a`, `X₂ ↦ Z₁^aX₂`, with both `Z` images fixed. At `d = 2`
with `a = 1` this is `CZ`. A coefficient congruent to zero is a valid identity.

# Examples
```julia
apply!(tab, CPhase(1, 2))
```
"""
struct CPhase <: AbstractClifford
    qudit1::Int
    qudit2::Int
    a::Int
    function CPhase(qudit1::Int, qudit2::Int, a::Int)
        _check_pair(qudit1, qudit2)
        return new(qudit1, qudit2, a)
    end
end
CPhase(qudit1::Int, qudit2::Int) = CPhase(qudit1, qudit2, 1)

"""
    SWAP(qudit1::Int, qudit2::Int)

Exchange two qudits: `X₁ ↦ X₂`, `X₂ ↦ X₁`, `Z₁ ↦ Z₂`, `Z₂ ↦ Z₁`.

# Examples
```julia
apply!(tab, SWAP(1, 4))
```
"""
struct SWAP <: AbstractClifford
    qudit1::Int
    qudit2::Int
    function SWAP(qudit1::Int, qudit2::Int)
        _check_pair(qudit1, qudit2)
        return new(qudit1, qudit2)
    end
end

########################################
# Raw gate data                        #
########################################

# `_clifford_data(g, d, inversemod) -> (targets, F, a)`
#
# `F[i]` is COLUMN `i`: the exponent vector of the image of generator `i`, in
# all-X-then-all-Z order, already reduced mod `d`. `a[i]` is the RAW phase
# exponent of that image, reduced mod `phase_modulus(d)`. Generators are ordered
# `X₁…X_K, Z₁…Z_K`, matching the tableau's row order on the target block.
#
# Every parameter is normalized mod `d` BEFORE any multiplication or negation,
# so `typemin(Int)` from a caller never enters a product. Modulus-dependent
# validation lives here because a named gate does not know `d` at construction.
# Read support without arithmetic, allocations, or a dimension. Public
# consumers validate these indices before preparing any Clifford data.
@inline _clifford_targets(g::Union{Fourier,Phase,Multiplier,PauliGate}) = (g.qudit,)
@inline _clifford_targets(g::SUM) = (g.control, g.target)
@inline _clifford_targets(g::Union{CPhase,SWAP}) = (g.qudit1, g.qudit2)

function _clifford_data end

@inline function _clifford_data(g::Fourier, d::Int, ::InverseMod)
    return (_clifford_targets(g), ((0, 1), (d - 1, 0)), (0, 0))
end

@inline function _clifford_data(g::Phase, d::Int, ::InverseMod)
    # At d = 2 the image of X is i·XZ = Y, hence raw phase 1; C(v,2) vanishes
    # for bits, so this linear term is the whole qubit difference.
    a = d == 2 ? (1, 0) : (0, 0)
    return (_clifford_targets(g), ((1, 1), (0, 1)), a)
end

@inline function _clifford_data(g::Multiplier, d::Int, inversemod::InverseMod)
    am = mod(g.a, d)
    am == 0 && throw(ArgumentError(
        "Multiplier coefficient must be invertible mod d; got $(g.a) ≡ 0 (mod $d)."))
    ai = inversemod(am, d)
    return (_clifford_targets(g), ((am, 0), (0, ai)), (0, 0))
end

@inline function _clifford_data(g::PauliGate, d::Int, ::InverseMod)
    xm = mod(g.x, d)
    zm = mod(g.z, d)
    # Conjugation by X^x0 Z^z0 sends P(v) to ω^{z0·x - x0·z} P(v).
    a = d == 2 ? (mod(2 * zm, 4), mod(2 * xm, 4)) : (zm, mod(-xm, d))
    return (_clifford_targets(g), ((1, 0), (0, 1)), a)
end

@inline function _clifford_data(g::SUM, d::Int, ::InverseMod)
    am = mod(g.a, d)
    na = mod(-am, d)
    F = ((1, am, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, na, 1))
    return (_clifford_targets(g), F, (0, 0, 0, 0))
end

@inline function _clifford_data(g::CPhase, d::Int, ::InverseMod)
    am = mod(g.a, d)
    F = ((1, 0, 0, am), (0, 1, am, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    return (_clifford_targets(g), F, (0, 0, 0, 0))
end

@inline function _clifford_data(g::SWAP, ::Int, ::InverseMod)
    F = ((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))
    return (_clifford_targets(g), F, (0, 0, 0, 0))
end

########################################
# Printing                             #
########################################

Base.show(io::IO, g::Fourier) = print(io, "Fourier(", g.qudit, ")")
Base.show(io::IO, g::Phase) = print(io, "Phase(", g.qudit, ")")
Base.show(io::IO, g::Multiplier) = print(io, "Multiplier(", g.qudit, ", ", g.a, ")")
Base.show(io::IO, g::PauliGate) =
    print(io, "PauliGate(", g.qudit, ", ", g.x, ", ", g.z, ")")
Base.show(io::IO, g::SUM) =
    print(io, "SUM(", g.control, ", ", g.target, ", ", g.a, ")")
Base.show(io::IO, g::CPhase) =
    print(io, "CPhase(", g.qudit1, ", ", g.qudit2, ", ", g.a, ")")
Base.show(io::IO, g::SWAP) = print(io, "SWAP(", g.qudit1, ", ", g.qudit2, ")")
