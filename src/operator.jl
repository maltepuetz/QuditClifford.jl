"""
    AbstractPauli

Abstract supertype for Pauli operators supported by QuditClifford.
Concrete operators can be used with [`measure!`](@ref) and [`expect!`](@ref).

# Examples
```julia
op = SinglePauli(1, 1, 0)
op isa AbstractPauli
```
"""
abstract type AbstractPauli end

"""
    FewQuditPauli <: AbstractPauli

Abstract supertype for Pauli operators acting on a small number of qudits.
All concrete subtypes store 1-based qudit indices and X/Z exponents; the phase
is stored as an exponent (interpreted by the consuming tableau's dimension).

# Examples
```julia
op = DoublePauli(1, 1, 0, 3, 0, 1)  # X1 * Z3
op isa FewQuditPauli
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
```julia
op = SinglePauli(2, 1, 0)        # X on qudit 2
op = SinglePauli(1, 0, 1, 2)     # i^2 Z on qudit 1 when d=2
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
- `qudit1::Int`, `qudit2::Int`, `qudit3::Int`: 1-based qudit indices.
- `x1::Int`, `z1::Int`, `x2::Int`, `z2::Int`, `x3::Int`, `z3::Int`: Exponents for each qudit.
- `phase::Int`: Phase exponent `k` (interpreted as `ω^k` or `i^k` depending on `d`).

# Examples
```julia
op = TriplePauli(1, 1, 0, 2, 0, 1, 3, 1, 1)  # X1 * Z2 * X3Z3
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
```julia
op = NPauli((1, 3), (1, 0), (0, 1))  # X1 * Z3
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

"""
    GeneralPauli(xz::AbstractVector{<:Integer}, phase::Int)
    GeneralPauli(n::Int, d::Int, op::AbstractVector{<:Integer})

General Pauli operator backed by a dense `xz` vector of length `2n` and a phase exponent.

Performance note: `xz` is stored as a `Vector{Int}`. If the input is not already
`Vector{Int}` with length `2n` (including the `2n+1` case), the constructor will
allocate and copy.
"""
mutable struct GeneralPauli <: AbstractPauli
    xz::Vector{Int}
    phase::Int
end

function GeneralPauli(xz::AbstractVector{<:Integer}, phase::Int)
    n = length(xz) ÷ 2
    (2n == length(xz)) || throw(ArgumentError("xz must have even length (2n)."))
    if xz isa Vector{Int}
        return GeneralPauli(xz, phase)
    end
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

"""
    set_operator!(tab::AbstractTableau, i::Int, op::SinglePauli)
    set_operator!(tab::AbstractTableau, i::Int, op::DoublePauli)
    set_operator!(tab::AbstractTableau, i::Int, op::TriplePauli)
    set_operator!(tab::AbstractTableau, i::Int, op::NPauli{D}) where D
    set_operator!(tab::AbstractTableau, i::Int, op::GeneralPauli)
    set_operator!(tab::AbstractTableau, i::Int, op::AbstractVector{<:Integer})
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::SinglePauli)
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::DoublePauli)
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::TriplePauli)
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::NPauli{D}) where D
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::GeneralPauli)
    set_operator!(dst::AbstractVector{<:Integer}, tab::StabilizerTableau, op::AbstractVector{<:Integer})
Set the i-th generator in the tab.stab to the given operator `op`, or set
the vector dst to the given operator `op`.
"""
function set_operator! end

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
# copy into StabilizerTableau.stab #
#######################################
@inline function set_operator!(tab::AbstractTableau, i::Int, op::SinglePauli)
    n = tab.n
    d = tab.d
    @turbo for j in axes(tab.stab, 1)
        tab.stab[j, i] = 0
    end
    tab.stab[op.qudit, i] = mod(op.x, d)
    tab.stab[op.qudit+n, i] = mod(op.z, d)
    if tab.storephase
        tab.stab[2n+1, i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(tab::AbstractTableau, i::Int, op::DoublePauli)
    n = tab.n
    d = tab.d
    @turbo for j in axes(tab.stab, 1)
        tab.stab[j, i] = 0
    end
    tab.stab[op.qudit1, i] = mod(op.x1, d)
    tab.stab[op.qudit2, i] = mod(op.x2, d)
    tab.stab[op.qudit1+n, i] = mod(op.z1, d)
    tab.stab[op.qudit2+n, i] = mod(op.z2, d)
    if tab.storephase
        tab.stab[2n+1, i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(tab::AbstractTableau, i::Int, op::TriplePauli)
    n = tab.n
    d = tab.d
    @turbo for j in axes(tab.stab, 1)
        tab.stab[j, i] = 0
    end
    tab.stab[op.qudit1, i] = mod(op.x1, d)
    tab.stab[op.qudit2, i] = mod(op.x2, d)
    tab.stab[op.qudit3, i] = mod(op.x3, d)
    tab.stab[op.qudit1+n, i] = mod(op.z1, d)
    tab.stab[op.qudit2+n, i] = mod(op.z2, d)
    tab.stab[op.qudit3+n, i] = mod(op.z3, d)
    if tab.storephase
        tab.stab[2n+1, i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(tab::AbstractTableau, i::Int, op::NPauli{D}) where D
    n = tab.n
    d = tab.d
    @turbo for j in axes(tab.stab, 1)
        tab.stab[j, i] = 0
    end
    @turbo for k in 1:D
        tab.stab[op.qudits[k], i] = mod(op.xs[k], d)
    end
    @turbo for k in 1:D
        tab.stab[op.qudits[k]+n, i] = mod(op.zs[k], d)
    end
    if tab.storephase
        tab.stab[2n+1, i] = mod(op.phase, phase_modulus(d))
    end
    nothing
end
@inline function set_operator!(tab::AbstractTableau, i::Int, op::AbstractVector{<:Integer})
    n = tab.n
    d = tab.d
    len_op = length(op)
    @assert length(op) == 2n || length(op) == 2n + 1

    stab = tab.stab
    @turbo for j in 1:2n
        stab[j, i] = mod(op[j], d)
    end
    if tab.storephase && (len_op == 2n + 1)
        stab[2n+1, i] = mod(op[2n+1], phase_modulus(d))
    elseif tab.storephase
        stab[2n+1, i] = 0
    end
    nothing
end

@inline function set_operator!(tab::AbstractTableau, i::Int, op::GeneralPauli)
    n = tab.n
    d = tab.d
    stab = tab.stab
    @assert length(op.xz) == 2n

    @turbo for j in 1:2n
        stab[j, i] = mod(op.xz[j], d)
    end
    if tab.storephase
        stab[2n+1, i] = mod(op.phase, phase_modulus(d))
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
