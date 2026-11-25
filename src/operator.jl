"""
    AbstractPauli

Abstract supertype for Pauli operators supported by QuditClifford.
Concrete operators can be used with [`measure!`](@ref) and [`expect!`](@ref).

# Examples
```jldoctest
julia> op = SinglePauli(1, 1, 0)
X₁

julia> op isa AbstractPauli
true
```
"""
abstract type AbstractPauli end

"""
    FewQuditPauli <: AbstractPauli

Abstract supertype for Pauli operators acting on a small number of qudits.
All concrete subtypes store 1-based qudit indices, X/Z exponents, and a phase
exponent, but not the local qudit dimension. These values are interpreted only
when the operator is used with a tableau: X/Z exponents are reduced modulo the
tableau dimension `d`, so the same operator object can represent different
physical operators for different `d`.

For odd prime `d`, phase exponent `k` represents ``\\omega^k``, where
``\\omega = \\exp(2\\pi i/d)``, with `k` reduced modulo `d`. For `d = 2`, it
represents ``i^k``, with `k` reduced modulo `4`.

# Examples
```jldoctest
julia> op = DoublePauli(1, 1, 0, 3, 0, 1)
X₁ Z₃

julia> op isa FewQuditPauli
true
```
"""
abstract type FewQuditPauli <: AbstractPauli end

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
```jldoctest
julia> SinglePauli(2, 1, 0)        # X on qudit 2
X₂

julia> SinglePauli(1, 0, 1, 2)     # i² Z on qudit 1 when d = 2
[phase=2] Z₁
```
"""
struct SinglePauli <: FewQuditPauli
    qudit::Int
    x::Int
    z::Int
    phase::Int
end
SinglePauli(qudit::Int, x::Int, z::Int) = begin
    SinglePauli(qudit, x, z, 0)
end

"""
    DoublePauli(qudit1::Int, x1::Int, z1::Int, qudit2::Int, x2::Int, z2::Int)
    DoublePauli(qudit1::Int, x1::Int, z1::Int, qudit2::Int, x2::Int, z2::Int, phase::Int)

Pauli operator acting on two qudits.

# Arguments
- `qudit1::Int`, `qudit2::Int`: 1-based qudit indices. They must be distinct.
  The constructor does not check this; a function that takes the operator
  together with a register size, such as [`measure!`](@ref), [`expect!`](@ref),
  [`expect_int!`](@ref), [`dephase!`](@ref) or [`conjugate`](@ref), throws an
  `ArgumentError` for a repeated index before it changes any state.
- `x1::Int`, `z1::Int`: Exponents for the first qudit.
- `x2::Int`, `z2::Int`: Exponents for the second qudit.
- `phase::Int`: Phase exponent `k` (interpreted as `ω^k` or `i^k` depending on `d`).

# Examples
```jldoctest
julia> DoublePauli(1, 1, 0, 2, 0, 1)
X₁ Z₂

julia> DoublePauli(1, 0, 1, 2, 0, 1, 3)   # i³ Z₁ Z₂ when d = 2
[phase=3] Z₁ Z₂
```
"""
struct DoublePauli <: FewQuditPauli
    qudit1::Int
    x1::Int
    z1::Int
    qudit2::Int
    x2::Int
    z2::Int
    phase::Int
end
DoublePauli(qudit1::Int, x1::Int, z1::Int, qudit2::Int, x2::Int, z2::Int) = begin
    DoublePauli(qudit1, x1, z1, qudit2, x2, z2, 0)
end

"""
    TriplePauli(qudit1::Int, x1::Int, z1::Int,
                qudit2::Int, x2::Int, z2::Int,
                qudit3::Int, x3::Int, z3::Int)
    TriplePauli(qudit1::Int, x1::Int, z1::Int,
                qudit2::Int, x2::Int, z2::Int,
                qudit3::Int, x3::Int, z3::Int, phase::Int)

Pauli operator acting on three qudits.

# Arguments
- `qudit1::Int`, `qudit2::Int`, `qudit3::Int`: 1-based qudit indices. They must
  be pairwise distinct. The constructor does not check this; a function that
  takes the operator together with a register size, such as [`measure!`](@ref),
  [`expect!`](@ref), [`expect_int!`](@ref), [`dephase!`](@ref) or
  [`conjugate`](@ref), throws an `ArgumentError` for a repeated index before it
  changes any state.
- `x1::Int`, `z1::Int`, `x2::Int`, `z2::Int`, `x3::Int`, `z3::Int`: Exponents for each qudit.
- `phase::Int`: Phase exponent `k` (interpreted as `ω^k` or `i^k` depending on `d`).

# Examples
```jldoctest
julia> TriplePauli(1, 1, 0, 2, 0, 1, 3, 1, 1)
X₁ Z₂ X₃ Z₃
```
"""
struct TriplePauli <: FewQuditPauli
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
TriplePauli(qudit1::Int, x1::Int, z1::Int, qudit2::Int, x2::Int, z2::Int, qudit3::Int, x3::Int, z3::Int) = begin
    TriplePauli(qudit1, x1, z1, qudit2, x2, z2, qudit3, x3, z3, 0)
end

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
```jldoctest
julia> NPauli((1, 3), (1, 0), (0, 1))
X₁ Z₃
```
"""
struct NPauli{D} <: FewQuditPauli
    qudits::NTuple{D,Int}
    xs::NTuple{D,Int}
    zs::NTuple{D,Int}
    phase::Int
end
NPauli(qudits::NTuple{D,Int}, xs::NTuple{D,Int}, zs::NTuple{D,Int}) where D = begin
    NPauli{D}(qudits, xs, zs, 0)
end

########################################
# Pauli representations                #
########################################

const _PAULI_SUBSCRIPT_DIGITS =
    ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')
const _PAULI_SUPERSCRIPT_DIGITS =
    ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')

@inline function _show_script_integer(
    io::IO,
    value::Int,
    digits::NTuple{10,Char},
    minus::Char,
)
    for digit in string(value)
        if digit == '-'
            print(io, minus)
        else
            print(io, digits[Int(digit) - Int('0') + 1])
        end
    end
    return nothing
end

@inline function _show_pauli_power(
    io::IO,
    axis::Char,
    qudit::Int,
    exponent::Int,
    wrote_term::Bool,
)
    exponent == 0 && return wrote_term
    wrote_term && print(io, ' ')
    print(io, axis)
    _show_script_integer(io, qudit, _PAULI_SUBSCRIPT_DIGITS, '₋')
    if exponent != 1
        _show_script_integer(io, exponent, _PAULI_SUPERSCRIPT_DIGITS, '⁻')
    end
    return true
end

function _show_pauli(
    io::IO,
    qudits,
    xs,
    zs,
    phase::Int,
)
    wrote_term = false
    if phase != 0
        print(io, "[phase=", phase, ']')
        wrote_term = true
    end

    wrote_local_term = false
    @inbounds for i in eachindex(qudits)
        if xs[i] != 0
            wrote_term = _show_pauli_power(io, 'X', qudits[i], xs[i], wrote_term)
            wrote_local_term = true
        end
        if zs[i] != 0
            wrote_term = _show_pauli_power(io, 'Z', qudits[i], zs[i], wrote_term)
            wrote_local_term = true
        end
    end

    if !wrote_local_term
        wrote_term && print(io, ' ')
        print(io, 'I')
    end
    return nothing
end

@inline _few_qudit_pauli_data(op::SinglePauli) =
    ((op.qudit,), (op.x,), (op.z,), op.phase)
@inline _few_qudit_pauli_data(op::DoublePauli) = (
    (op.qudit1, op.qudit2),
    (op.x1, op.x2),
    (op.z1, op.z2),
    op.phase,
)
@inline _few_qudit_pauli_data(op::TriplePauli) = (
    (op.qudit1, op.qudit2, op.qudit3),
    (op.x1, op.x2, op.x3),
    (op.z1, op.z2, op.z3),
    op.phase,
)
@inline _few_qudit_pauli_data(op::NPauli) = (op.qudits, op.xs, op.zs, op.phase)

const _BuiltinFewQuditPauli = Union{SinglePauli,DoublePauli,TriplePauli,NPauli}

function Base.show(io::IO, op::_BuiltinFewQuditPauli)
    return _show_pauli(io, _few_qudit_pauli_data(op)...)
end

"""
    GeneralPauli(xz::AbstractVector{<:Integer}, phase::Int)
    GeneralPauli(n::Int, d::Int, op::AbstractVector{<:Integer})

General Pauli operator backed by a dense `xz` vector of length `2n` and a phase exponent.
The vector is ordered as `[x₁, …, xₙ, z₁, …, zₙ]`, and the operator is
`ζ^phase ⊗_q X_q^{x_q} Z_q^{z_q}` with `ζ = i` at `d = 2` and `ζ = exp(2πi/d)`
otherwise.

# Arguments
- `xz`: the `2n` X and Z exponents. Exponents are reduced mod `d` only when a
  tableau consumes the operator, as for the sparse Pauli types.
- `phase::Int`: the phase exponent.
- `n::Int`: the number of qudits, for the second form.
- `d::Int`: not used; the dimension comes from the tableau that consumes the
  operator.
- `op`: `2n` exponents, or `2n + 1` with the phase exponent last (phase `0`
  otherwise).

# Ownership
`xz` is stored as a `Vector{Int}`. A `Vector{Int}` input is stored itself, not
copied, by `GeneralPauli(xz, phase)` and by `GeneralPauli(n, d, op)` when
`length(op) == 2n`: later changes to the caller's vector change the operator.
Every other input, including a `Vector{Int}` of length `2n + 1`, is copied.
"""
mutable struct GeneralPauli <: AbstractPauli
    xz::Vector{Int}
    phase::Int
end

function GeneralPauli(xz::AbstractVector{<:Integer}, phase::Int)
    n = length(xz) ÷ 2
    (2n == length(xz)) || throw(ArgumentError("xz must have even length (2n)."))
    xz_copy = Vector{Int}(undef, length(xz))
    @inbounds for i in eachindex(xz)
        xz_copy[i] = Int(xz[i])
    end
    return GeneralPauli(xz_copy, phase)
end

function GeneralPauli(n::Int, d::Int, op::AbstractVector{<:Integer})
    len = length(op)
    (len == 2n || len == 2n + 1) || throw(ArgumentError("Operator must have length 2n or 2n+1."))
    if len == 2n + 1
        phase = Int(op[2n + 1])
        if op isa Vector{Int}
            return GeneralPauli(op[1:2n], phase)
        end
        xz_copy = Vector{Int}(undef, 2n)
        @inbounds for i in 1:2n
            xz_copy[i] = Int(op[i])
        end
        return GeneralPauli(xz_copy, phase)
    else
        phase = 0
        if op isa Vector{Int}
            return GeneralPauli(op, phase)
        end
        xz_copy = Vector{Int}(undef, 2n)
        @inbounds for i in 1:2n
            xz_copy[i] = Int(op[i])
        end
        return GeneralPauli(xz_copy, phase)
    end
end

function Base.show(io::IO, op::GeneralPauli)
    n = length(op.xz) ÷ 2
    return _show_pauli(
        io,
        Base.OneTo(n),
        @view(op.xz[1:n]),
        @view(op.xz[(n + 1):(2n)]),
        op.phase,
    )
end

"""
    set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::SinglePauli)
    set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::DoublePauli)
    set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::TriplePauli)
    set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::NPauli{D}) where D
    set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::GeneralPauli)
    set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::AbstractVector{<:Integer})
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::SinglePauli)
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::DoublePauli)
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::TriplePauli)
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::NPauli{D}) where D
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::GeneralPauli)
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::AbstractVector{<:Integer})
Set the i-th stabilizer generator of the tableau to the given operator `op`, or set
the vector dst to the given operator `op`. `L` is the tableau's layout marker.
"""
function set_operator! end

#############################################
# Operand validation                        #
#############################################

# Throw an `ArgumentError` unless `op` is a Pauli on a register of `n` qudits:
# every qudit index in `1:n` and none repeated, or a dense vector of length `2n`.
# Callers run it before touching any state, since the kernels index under
# `@turbo`/`@inbounds` and a sparse index of `n + 1` lands in a padding row or
# a neighbouring column of the same block without tripping a bounds check. O(k²) for a k-qudit sparse Pauli,
# O(1) for a dense one, and allocation-free.
function _check_pauli_support(op::FewQuditPauli, n::Int)
    qudits = _few_qudit_pauli_data(op)[1]
    @inbounds for i in eachindex(qudits)
        q = qudits[i]
        1 <= q <= n || throw(ArgumentError(
            "Pauli acts on qudit $q, outside the register of n = $n qudits."))
        for r in 1:(i - 1)
            qudits[r] == q && throw(ArgumentError(
                "Pauli acts on qudit $q more than once; repeated qudit indices are not supported."))
        end
    end
    return nothing
end

function _check_pauli_support(op::GeneralPauli, n::Int)
    len = length(op.xz)
    len == 2n || throw(ArgumentError(
        "Pauli vector has length $len, expected 2n = $(2n)."))
    return nothing
end

_check_pauli_support(::AbstractPauli, ::Int) = nothing

_check_pauli_support(tab::AbstractTableau, op::AbstractPauli) = _check_pauli_support(op, tab.n)

######################################
# copy into arbitrary AbstractVector #
######################################
@inline function set_operator!(dst::AbstractVector{<:Integer}, tab::AbstractTableau, op::SinglePauli)
    n = tab.n
    d = tab.d
    len = length(dst)
    @assert (len == 2n || len == 2n + 1)

    @turbo for j in eachindex(dst)
        dst[j] = 0
    end

    dst[op.qudit] = mod(op.x, d)
    dst[op.qudit+n] = mod(op.z, d)
    if tab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(dst::AbstractVector{<:Integer}, tab::AbstractTableau, op::DoublePauli)
    n = tab.n
    d = tab.d
    len = length(dst)
    @assert (len == 2n || len == 2n + 1)

    @turbo for j in eachindex(dst)
        dst[j] = 0
    end

    dst[op.qudit1] = mod(op.x1, d)
    dst[op.qudit2] = mod(op.x2, d)
    dst[op.qudit1+n] = mod(op.z1, d)
    dst[op.qudit2+n] = mod(op.z2, d)
    if tab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(dst::AbstractVector{<:Integer}, tab::AbstractTableau, op::TriplePauli)
    n = tab.n
    d = tab.d
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
    if tab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(dst::AbstractVector{<:Integer}, tab::AbstractTableau, op::NPauli{D}) where D
    n = tab.n
    d = tab.d
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
    if tab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end

@inline function set_operator!(dst::AbstractVector{<:Integer}, tab::AbstractTableau, op::GeneralPauli)
    n = tab.n
    d = tab.d
    len = length(dst)
    @assert (len == 2n || len == 2n + 1)
    _check_pauli_support(op, n)

    @turbo for j in eachindex(dst)
        dst[j] = 0
    end

    @turbo for j in 1:(2n)
        dst[j] = mod(op.xz[j], d)
    end
    if tab.storephase && (len == 2n + 1)
        dst[2n+1] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(dst::AbstractVector{<:Integer}, tab::AbstractTableau, op::AbstractVector{<:Integer})
    n = tab.n
    d = tab.d
    len = length(dst)
    len_op = length(op)
    @assert (len == 2n || len == 2n + 1)
    len_op == 2n || len_op == 2n + 1 || throw(ArgumentError(
        "Pauli vector has length $len_op, expected 2n = $(2n) or 2n + 1 = $(2n + 1)."))
    
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
# copy into the stabilizer generators #
#######################################
@inline function set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::SinglePauli)
    d = tab.d
    stab = _stab(tab, L)
    _zero_gen!(stab, i)
    _xset!(stab, mod(op.x, d), op.qudit, i)
    _zset!(stab, mod(op.z, d), op.qudit, i)
    if tab.storephase
        _phase(tab)[i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::DoublePauli)
    d = tab.d
    stab = _stab(tab, L)
    _zero_gen!(stab, i)
    _xset!(stab, mod(op.x1, d), op.qudit1, i)
    _xset!(stab, mod(op.x2, d), op.qudit2, i)
    _zset!(stab, mod(op.z1, d), op.qudit1, i)
    _zset!(stab, mod(op.z2, d), op.qudit2, i)
    if tab.storephase
        _phase(tab)[i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::TriplePauli)
    d = tab.d
    stab = _stab(tab, L)
    _zero_gen!(stab, i)
    _xset!(stab, mod(op.x1, d), op.qudit1, i)
    _xset!(stab, mod(op.x2, d), op.qudit2, i)
    _xset!(stab, mod(op.x3, d), op.qudit3, i)
    _zset!(stab, mod(op.z1, d), op.qudit1, i)
    _zset!(stab, mod(op.z2, d), op.qudit2, i)
    _zset!(stab, mod(op.z3, d), op.qudit3, i)
    if tab.storephase
        _phase(tab)[i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::NPauli{D}) where D
    d = tab.d
    stab = _stab(tab, L)
    _zero_gen!(stab, i)
    @inbounds for k in 1:D
        _xset!(stab, mod(op.xs[k], d), op.qudits[k], i)
    end
    @inbounds for k in 1:D
        _zset!(stab, mod(op.zs[k], d), op.qudits[k], i)
    end
    if tab.storephase
        _phase(tab)[i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::AbstractVector{<:Integer})
    n = tab.n
    d = tab.d
    len_op = length(op)
    len_op == 2n || len_op == 2n + 1 || throw(ArgumentError(
        "Pauli vector has length $len_op, expected 2n = $(2n) or 2n + 1 = $(2n + 1)."))

    stab = _stab(tab, L)
    xg = _xgen(stab, i)
    zg = _zgen(stab, i)
    E = eltype(xg)
    @turbo for q in 1:n
        xg[q] = E(mod(op[q], d))
    end
    @turbo for q in 1:n
        zg[q] = E(mod(op[n+q], d))
    end
    if tab.storephase && (len_op == 2n + 1)
        _phase(tab)[i] = mod(op[2n+1], phase_modulus(d))
    elseif tab.storephase
        _phase(tab)[i] = 0
    end
    nothing
end

# Packed storage: generator `i` is zeroed and the odd exponents set.
@inline function set_operator!(tab::_BitTableau, L::TableauLayout, i::Int,
                               op::AbstractVector{<:Integer})
    n = tab.n
    len_op = length(op)
    len_op == 2n || len_op == 2n + 1 || throw(ArgumentError(
        "Pauli vector has length $len_op, expected 2n = $(2n) or 2n + 1 = $(2n + 1)."))
    _set_bit_generator!(_stab(tab, L), i, op, n)
    if tab.storephase && (len_op == 2n + 1)
        _phase(tab)[i] = mod(op[2n+1], phase_modulus(tab.d))
    elseif tab.storephase
        _phase(tab)[i] = 0
    end
    nothing
end

@inline function set_operator!(tab::_BitTableau, L::TableauLayout, i::Int, op::GeneralPauli)
    n = tab.n
    _check_pauli_support(op, n)
    _set_bit_generator!(_stab(tab, L), i, op.xz, n)
    if tab.storephase
        _phase(tab)[i] = mod(op.phase, phase_modulus(tab.d))
    end
    nothing
end

# Overwrite generator `i` of `b` with the exponents `xz[1:2n]` mod 2.
@inline function _set_bit_generator!(b::BitBlock, i::Int, xz::AbstractVector{<:Integer}, n::Int)
    _zero_gen!(b, i)
    return _set_bit_exponents!(b, i, xz, n)
end

# Set the odd exponents of `xz[1:2n]` in generator `i` of `b`, which is zero.
@inline function _set_bit_exponents!(b::BitBlock, i::Int, xz::AbstractVector{<:Integer}, n::Int)
    @inbounds for q in 1:n
        isodd(xz[q]) && _xflip!(b, UInt64(1), q, i)
        isodd(xz[n+q]) && _zflip!(b, UInt64(1), q, i)
    end
    return b
end

@inline function set_operator!(tab::AbstractTableau, L::TableauLayout, i::Int, op::GeneralPauli)
    n = tab.n
    d = tab.d
    stab = _stab(tab, L)
    xg = _xgen(stab, i)
    zg = _zgen(stab, i)
    _check_pauli_support(op, n)

    # `@simd`, not `@turbo`: `Int` loads with stores of the narrower `E`. A
    # residue mod `d` fits `E`, so `% E` is exact.
    E = eltype(xg)
    xz = op.xz
    @inbounds @simd for q in 1:n
        xg[q] = mod(xz[q], d) % E
    end
    @inbounds @simd for q in 1:n
        zg[q] = mod(xz[n+q], d) % E
    end
    if tab.storephase
        _phase(tab)[i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end

@inline _xdotz_parity(op::SinglePauli) = (op.x * op.z) & 1
@inline _xdotz_parity(op::DoublePauli) = (op.x1 * op.z1 + op.x2 * op.z2) & 1
@inline _xdotz_parity(op::TriplePauli) = (op.x1 * op.z1 + op.x2 * op.z2 + op.x3 * op.z3) & 1
@inline function _xdotz_parity(op::NPauli{D}) where D
    s = 0
    @inbounds for i in 1:D
        s += op.xs[i] * op.zs[i]
    end
    return s & 1
end
@inline function _xdotz_parity(op::GeneralPauli)
    n = length(op.xz) ÷ 2
    s = 0
    @inbounds for i in 1:n
        s += op.xz[i] * op.xz[n + i]
    end
    return s & 1
end

@inline function _fix_qubit_phase(phase::Int, parity::Int)
    return (phase & 2) | (parity & 1)
end

@inline function _is_valid_measurement(op::AbstractPauli, d::Int)
    d != 2 && return true
    return ((op.phase - _xdotz_parity(op)) & 1) == 0
end
