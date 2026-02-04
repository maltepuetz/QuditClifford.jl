abstract type Operator end

abstract type FewQuditOperator <: Operator end
struct SinglePauli <: FewQuditOperator
    qudit::Int
    x::Int
    z::Int
    phase::Int
end
SinglePauli(qudit::Int, x::Int, z::Int) = SinglePauli(qudit, x, z, 0)
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
struct NPauli{D} <: FewQuditOperator
    qudits::NTuple{D,Int}
    xs::NTuple{D,Int}
    zs::NTuple{D,Int}
    phase::Int
end
NPauli(qudits::NTuple{D,Int}, xs::NTuple{D,Int}, zs::NTuple{D,Int}) where D = NPauli{D}(qudits, xs, zs, 0)

"""
    set_operator!(stabtab, i::Int, op::SinglePauli)
    set_operator!(stabtab, i::Int, op::DoublePauli)
    set_operator!(stabtab, i::Int, op::TriplePauli)
    set_operator!(stabtab, i::Int, op::NPauli{D}) where D
    set_operator!(stabtab, i::Int, op::AbstractVector{<:Integer})
    set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::SinglePauli)
    set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::DoublePauli)
    set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::TriplePauli)
    set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::NPauli{D}) where D
    set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::AbstractVector{<:Integer})
Set the i-th operator in the stabtab.tableau to the given operator `op`, or set
the vector dst to the given operator `op`.
"""
function set_operator! end

######################################
# copy into arbitrary AbstractVector #
######################################
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::SinglePauli)
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
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::DoublePauli)
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
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::TriplePauli)
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
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::NPauli{D}) where D
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
@inline function set_operator!(dst::AbstractVector{<:Integer}, stabtab, op::AbstractVector{<:Integer})
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
@inline function set_operator!(stabtab, i::Int, op::SinglePauli)
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
@inline function set_operator!(stabtab, i::Int, op::DoublePauli)
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
@inline function set_operator!(stabtab, i::Int, op::TriplePauli)
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
@inline function set_operator!(stabtab, i::Int, op::NPauli{D}) where D
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
@inline function set_operator!(stabtab, i::Int, op::AbstractVector{<:Integer})
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
