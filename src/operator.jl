"""
    Operator

Abstract supertype for operators supported by QuditClifford.
Concrete operators (currently Pauli operators on a few qudits) can be used with
[`measure!`](@ref) and [`expect!`](@ref).

# Examples
```julia
op = SinglePauli(1, 1, 0)
op isa Operator
```
"""
abstract type Operator end

"""
    FewQuditOperator <: Operator

Abstract supertype for Pauli operators acting on a small number of qudits.
All concrete subtypes store 1-based qudit indices and X/Z exponents; the phase
is stored as an exponent (interpreted by the consuming tableau's dimension).

# Examples
```julia
op = DoublePauli(1, 1, 0, 3, 0, 1)  # X1 * Z3
op isa FewQuditOperator
```
"""
abstract type FewQuditOperator <: Operator end

"""
    SinglePauli(qudit::Int, x::Int, z::Int)
    SinglePauli(qudit::Int, x::Int, z::Int, phase::Int)

Pauli operator acting on a single qudit, with X/Z exponents and an optional phase.

# Arguments
- `qudit::Int`: 1-based qudit index.
- `x::Int`: Exponent of `X` on that qudit (reduced mod `d` by consuming functions).
- `z::Int`: Exponent of `Z` on that qudit (reduced mod `d` by consuming functions).
- `phase::Int`: Phase exponent `k`. Interpreted as `ω^k` for odd prime `d` and `i^k` for `d=2`.

# Examples
```julia
op = SinglePauli(2, 1, 0)        # X on qudit 2
op = SinglePauli(1, 0, 1, 2)     # i^2 Z on qudit 1 when d=2
```
"""
struct SinglePauli <: FewQuditOperator
    qudit::Int
    x::Int
    z::Int
    phase::Int
end
SinglePauli(qudit::Int, x::Int, z::Int) = SinglePauli(qudit, x, z, 0)

"""
    DoublePauli(qudit1::Int, x1::Int, z1::Int, qudit2::Int, x2::Int, z2::Int)
    DoublePauli(qudit1::Int, x1::Int, z1::Int, qudit2::Int, x2::Int, z2::Int, phase::Int)

Pauli operator acting on two qudits.

# Arguments
- `qudit1::Int`, `qudit2::Int`: 1-based qudit indices.
- `x1::Int`, `z1::Int`: Exponents for the first qudit.
- `x2::Int`, `z2::Int`: Exponents for the second qudit.
- `phase::Int`: Phase exponent `k` (interpreted as `ω^k` or `i^k` depending on `d`).

# Examples
```julia
op = DoublePauli(1, 1, 0, 2, 0, 1)    # X1 * Z2
op = DoublePauli(1, 0, 1, 2, 0, 1, 3) # i^3 Z1 * Z2 when d=2
```
"""
struct DoublePauli <: FewQuditOperator
    qudit1::Int
    x1::Int
    z1::Int
    qudit2::Int
    x2::Int
    z2::Int
    phase::Int
end
DoublePauli(qudit1::Int, x1::Int, z1::Int, qudit2::Int, x2::Int, z2::Int) = DoublePauli(qudit1, x1, z1, qudit2, x2, z2, 0)

"""
    TriplePauli(qudit1::Int, x1::Int, z1::Int,
                qudit2::Int, x2::Int, z2::Int,
                qudit3::Int, x3::Int, z3::Int)
    TriplePauli(qudit1::Int, x1::Int, z1::Int,
                qudit2::Int, x2::Int, z2::Int,
                qudit3::Int, x3::Int, z3::Int, phase::Int)

Pauli operator acting on three qudits.

# Arguments
- `qudit1::Int`, `qudit2::Int`, `qudit3::Int`: 1-based qudit indices.
- `x1::Int`, `z1::Int`, `x2::Int`, `z2::Int`, `x3::Int`, `z3::Int`: Exponents for each qudit.
- `phase::Int`: Phase exponent `k` (interpreted as `ω^k` or `i^k` depending on `d`).

# Examples
```julia
op = TriplePauli(1, 1, 0, 2, 0, 1, 3, 1, 1)  # X1 * Z2 * X3Z3
```
"""
struct TriplePauli <: FewQuditOperator
    qudit1::Int
    x1::Int
    z1::Int
    qudit2::Int
    x2::Int
    z2::Int
    qudit3::Int
    x3::Int
    z3::Int
    phase::Int
end
TriplePauli(qudit1::Int, x1::Int, z1::Int, qudit2::Int, x2::Int, z2::Int, qudit3::Int, x3::Int, z3::Int) = TriplePauli(qudit1, x1, z1, qudit2, x2, z2, qudit3, x3, z3, 0)

"""
    NPauli(qudits::NTuple{D,Int}, xs::NTuple{D,Int}, zs::NTuple{D,Int}) where D
    NPauli(qudits::NTuple{D,Int}, xs::NTuple{D,Int}, zs::NTuple{D,Int}, phase::Int) where D

Pauli operator acting on `D` qudits specified by tuples.

# Arguments
- `qudits::NTuple{D,Int}`: 1-based qudit indices (should be distinct).
- `xs::NTuple{D,Int}`: X exponents for each qudit.
- `zs::NTuple{D,Int}`: Z exponents for each qudit.
- `phase::Int`: Phase exponent `k` (interpreted as `ω^k` or `i^k` depending on `d`).

# Examples
```julia
op = NPauli((1, 3), (1, 0), (0, 1))  # X1 * Z3
```
"""
struct NPauli{D} <: FewQuditOperator
    qudits::NTuple{D,Int}
    xs::NTuple{D,Int}
    zs::NTuple{D,Int}
    phase::Int
end
NPauli(qudits::NTuple{D,Int}, xs::NTuple{D,Int}, zs::NTuple{D,Int}) where D = NPauli{D}(qudits, xs, zs, 0)

"""
    set_operator!(stabtab::StabilizerTableau, i::Int, op::SinglePauli)
    set_operator!(stabtab::StabilizerTableau, i::Int, op::DoublePauli)
    set_operator!(stabtab::StabilizerTableau, i::Int, op::TriplePauli)
    set_operator!(stabtab::StabilizerTableau, i::Int, op::NPauli{D}) where D
    set_operator!(stabtab::StabilizerTableau, i::Int, op::AbstractVector{<:Integer})
    set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::SinglePauli)
    set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::DoublePauli)
    set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::TriplePauli)
    set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::NPauli{D}) where D
    set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::AbstractVector{<:Integer})
Set the i-th generator in the stabtab.tableau to the given operator `op`, or set
the vector dst to the given operator `op`.
"""
function set_operator! end

######################################
# copy into arbitrary AbstractVector #
######################################
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::SinglePauli)
    n = stabtab.n
    d = stabtab.d
    len = length(dst)
    @assert (len == 2n || len == 2n + 1)

    @turbo for j in eachindex(dst)
        dst[j] = 0
    end

    dst[op.qudit] = mod(op.x, d)
    dst[op.qudit+n] = mod(op.z, d)
    if stabtab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::DoublePauli)
    n = stabtab.n
    d = stabtab.d
    len = length(dst)
    @assert (len == 2n || len == 2n + 1)

    @turbo for j in eachindex(dst)
        dst[j] = 0
    end

    dst[op.qudit1] = mod(op.x1, d)
    dst[op.qudit2] = mod(op.x2, d)
    dst[op.qudit1+n] = mod(op.z1, d)
    dst[op.qudit2+n] = mod(op.z2, d)
    if stabtab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::TriplePauli)
    n = stabtab.n
    d = stabtab.d
    len = length(dst)
    @assert (len == 2n || len == 2n + 1)
    
    @turbo for j in eachindex(dst)
        dst[j] = 0
    end
    
    dst[op.qudit1] = mod(op.x1, d)
    dst[op.qudit2] = mod(op.x2, d)
    dst[op.qudit3] = mod(op.x3, d)
    dst[op.qudit1+n] = mod(op.z1, d)
    dst[op.qudit2+n] = mod(op.z2, d)
    dst[op.qudit3+n] = mod(op.z3, d)
    if stabtab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::NPauli{D}) where D
    n = stabtab.n
    d = stabtab.d
    len = length(dst)
    @assert (len == 2n || len == 2n + 1)
    
    @turbo for j in eachindex(dst)
        dst[j] = 0
    end
    
    @turbo for k in 1:D
        dst[op.qudits[k]] = mod(op.xs[k], d)
    end
    @turbo for k in 1:D
        dst[op.qudits[k]+n] = mod(op.zs[k], d)
    end
    if stabtab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab::StabilizerTableau, op::AbstractVector{<:Integer})
    n = stabtab.n
    d = stabtab.d
    len = length(dst)
    len_op = length(op)
    @assert (len == 2n || len == 2n + 1)
    @assert (len_op == 2n || len_op == 2n + 1)
    
    @turbo for j in 1:2n
        dst[j] = mod(op[j], d)
    end
    if len == len_op == 2n + 1
        dst[2n+1] = mod(op[2n+1], phase_modulus(d))
    elseif len == 2n + 1
        dst[2n+1] = 0
    end
    nothing
end

#######################################
# copy into StabilizerTableau.tableau #
#######################################
@inline function set_operator!(stabtab::StabilizerTableau, i::Int, op::SinglePauli)
    n = stabtab.n
    d = stabtab.d
    @turbo for j in axes(stabtab.tableau, 1)
        stabtab.tableau[j, i] = 0
    end
    stabtab.tableau[op.qudit, i] = mod(op.x, d)
    stabtab.tableau[op.qudit+n, i] = mod(op.z, d)
    if stabtab.storephase
        stabtab.tableau[2n+1, i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(stabtab::StabilizerTableau, i::Int, op::DoublePauli)
    n = stabtab.n
    d = stabtab.d
    @turbo for j in axes(stabtab.tableau, 1)
        stabtab.tableau[j, i] = 0
    end
    stabtab.tableau[op.qudit1, i] = mod(op.x1, d)
    stabtab.tableau[op.qudit2, i] = mod(op.x2, d)
    stabtab.tableau[op.qudit1+n, i] = mod(op.z1, d)
    stabtab.tableau[op.qudit2+n, i] = mod(op.z2, d)
    if stabtab.storephase
        stabtab.tableau[2n+1, i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(stabtab::StabilizerTableau, i::Int, op::TriplePauli)
    n = stabtab.n
    d = stabtab.d
    @turbo for j in axes(stabtab.tableau, 1)
        stabtab.tableau[j, i] = 0
    end
    stabtab.tableau[op.qudit1, i] = mod(op.x1, d)
    stabtab.tableau[op.qudit2, i] = mod(op.x2, d)
    stabtab.tableau[op.qudit3, i] = mod(op.x3, d)
    stabtab.tableau[op.qudit1+n, i] = mod(op.z1, d)
    stabtab.tableau[op.qudit2+n, i] = mod(op.z2, d)
    stabtab.tableau[op.qudit3+n, i] = mod(op.z3, d)
    if stabtab.storephase
        stabtab.tableau[2n+1, i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(stabtab::StabilizerTableau, i::Int, op::NPauli{D}) where D
    n = stabtab.n
    d = stabtab.d
    @turbo for j in axes(stabtab.tableau, 1)
        stabtab.tableau[j, i] = 0
    end
    @turbo for k in 1:D
        stabtab.tableau[op.qudits[k], i] = mod(op.xs[k], d)
    end
    @turbo for k in 1:D
        stabtab.tableau[op.qudits[k]+n, i] = mod(op.zs[k], d)
    end
    if stabtab.storephase
        stabtab.tableau[2n+1, i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(stabtab::StabilizerTableau, i::Int, op::AbstractVector{<:Integer})
    n = stabtab.n
    d = stabtab.d
    len_op = length(op)
    @assert length(op) == 2n || length(op) == 2n + 1

    tab = stabtab.tableau
    @turbo for j in 1:2n
        tab[j, i] = mod(op[j], d)
    end
    if stabtab.storephase && (len_op == 2n + 1)
        tab[2n+1, i] = mod(op[2n+1], phase_modulus(d))
    elseif stabtab.storephase
        tab[2n+1, i] = 0
    end
    nothing
end

"""
Return the phase exponent kP of `op` reduced mod phase_modulus(d), if it exists.
If phase is not available (e.g. vector of length 2n), returns 0.

This is only used when `stabtab.storephase == true`.
"""
@inline function op_phase_exponent(stabtab::StabilizerTableau, op)::Int
    stabtab.storephase || return 0
    d_phase = phase_modulus(stabtab.d)
    n = stabtab.n

    if op isa AbstractVector{<:Integer}
        return (length(op) == 2n + 1) ? mod(op[2n+1], d_phase) : 0
    end

    # Pauli structs: assume they have a `phase` field (your code does)
    return mod(op.phase, d_phase)
end
