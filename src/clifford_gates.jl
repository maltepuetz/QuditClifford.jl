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

The named gate subtypes. `CliffordOperator` stores general operators at a
fixed dimension. Names are qudit-native; no qubit aliases are exported, and
the `d = 2` column is for orientation only. Here `ω = exp(2πi/d)`, which is
`ω = −1` at `d = 2`.

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
gates require distinct qudits.

The named gates above are dimension-agnostic. [`CliffordOperator`](@ref) is the
other subtype: it stores an arbitrary Clifford on an ordered support at a fixed
dimension, and supports composition and inversion. [`random_clifford`](@ref)
samples one uniformly.

# Examples
```jldoctest
julia> g = Fourier(1)
Fourier(1)

julia> g isa AbstractClifford
true
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

On basis states it is `|k⟩ ↦ d^{-1/2} Σ_j ω^{jk} |j⟩` with `ω = exp(2πi/d)`.
At `d = 2` this is the Hadamard gate.

# Examples
```jldoctest
julia> conjugate(Fourier(1), SinglePauli(1, 1, 0), 1; d = 3)
Z₁

julia> conjugate(Fourier(1), SinglePauli(1, 0, 1), 1; d = 3)
X₁²

julia> tab = StabilizerTableau(3, 1; state = :product, basis = :Z);

julia> apply!(tab, Fourier(1)) == StabilizerTableau(3, 1; state = :product, basis = :X)
true
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
```jldoctest
julia> conjugate(Phase(1), SinglePauli(1, 1, 0), 1; d = 3)
X₁ Z₁

julia> conjugate(Phase(1), SinglePauli(1, 1, 0), 1; d = 2)   # S X S† = Y = iXZ
[phase=1] X₁ Z₁

julia> tab = StabilizerTableau(2, 1; state = :product, basis = :X);

julia> apply!(tab, Phase(1)) == StabilizerTableau(2, 1; state = :product, basis = :Y)
true
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
```jldoctest
julia> conjugate(Multiplier(1, 2), SinglePauli(1, 1, 0), 1; d = 3)
X₁²

julia> conjugate(Multiplier(1, 2), SinglePauli(1, 0, 1), 1; d = 3)   # 2⁻¹ = 2 mod 3
Z₁²

julia> conjugate(Multiplier(1, 2), SinglePauli(1, 0, 1), 1; d = 5)   # 2⁻¹ = 3 mod 5
Z₁³
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
on a Pauli with exponents `(x', z')`, where `ω = exp(2πi/d)`; at `d = 2`,
`ω = −1`, so the shift is a sign.

# Examples
```jldoctest
julia> conjugate(PauliGate(2, 1, 0), SinglePauli(2, 0, 1), 2; d = 3)   # conjugate Z₂ by X₂
[phase=2] Z₂

julia> conjugate(PauliGate(2, 1, 0), SinglePauli(2, 0, 1), 2; d = 2)   # X Z X† = -Z
[phase=2] Z₂
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
```jldoctest
julia> conjugate(SUM(1, 3), SinglePauli(1, 1, 0), 3; d = 3)
X₁ X₃

julia> conjugate(SUM(1, 3), SinglePauli(3, 0, 1), 3; d = 3)
Z₁² Z₃

julia> SUM(1, 3) == SUM(1, 3, 1)
true
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

Controlled phase, `|u, v⟩ ↦ ω^{a·u·v}|u, v⟩` with `ω = exp(2πi/d)`; at
`d = 2`, `ω = −1`.

Images are `X₁ ↦ X₁Z₂^a`, `X₂ ↦ Z₁^aX₂`, with both `Z` images fixed. At `d = 2`
with `a = 1` this is `CZ`. A coefficient congruent to zero is a valid identity.

# Examples
```jldoctest
julia> conjugate(CPhase(1, 2), SinglePauli(1, 1, 0), 2; d = 3)
X₁ Z₂

julia> conjugate(CPhase(1, 2, 2), SinglePauli(2, 1, 0), 2; d = 3)
Z₁² X₂
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
```jldoctest
julia> conjugate(SWAP(1, 4), DoublePauli(1, 1, 0, 2, 0, 1), 4; d = 3)
Z₂ X₄
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
