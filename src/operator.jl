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
Set the i-th operator in the stabtab.tableau to the given FewQuditOperator `op`.
"""
function set_operator! end
@inline function set_operator!(stabtab, i::Int, op::SinglePauli)
    n = stabtab.n
    @turbo for j in axes(stabtab.tableau, 1)
        stabtab.tableau[j, i] = 0
    end
    stabtab.tableau[op.qudit, i] = op.x
    stabtab.tableau[op.qudit+n, i] = op.z
    !stabtab.storephase && return
    stabtab.tableau[2n+1, i] = op.phase
    nothing
end
@inline function set_operator!(stabtab, i::Int, op::DoublePauli)
    n = stabtab.n
    @turbo for j in axes(stabtab.tableau, 1)
        stabtab.tableau[j, i] = 0
    end
    stabtab.tableau[op.qudit1, i] = op.x1
    stabtab.tableau[op.qudit2, i] = op.x2
    stabtab.tableau[op.qudit1+n, i] = op.z1
    stabtab.tableau[op.qudit2+n, i] = op.z2
    !stabtab.storephase && return
    stabtab.tableau[2n+1, i] = op.phase
    nothing
end
@inline function set_operator!(stabtab, i::Int, op::TriplePauli)
    n = stabtab.n
    @turbo for j in axes(stabtab.tableau, 1)
        stabtab.tableau[j, i] = 0
    end
    stabtab.tableau[op.qudit1, i] = op.x1
    stabtab.tableau[op.qudit2, i] = op.x2
    stabtab.tableau[op.qudit3, i] = op.x3
    stabtab.tableau[op.qudit1+n, i] = op.z1
    stabtab.tableau[op.qudit2+n, i] = op.z2
    stabtab.tableau[op.qudit3+n, i] = op.z3
    !stabtab.storephase && return
    stabtab.tableau[2n+1, i] = op.phase
    nothing
end
@inline function set_operator!(stabtab, i::Int, op::NPauli{D}) where D
    n = stabtab.n
    @turbo for j in axes(stabtab.tableau, 1)
        stabtab.tableau[j, i] = 0
    end
    @turbo for k in 1:D
        stabtab.tableau[op.qudits[k], i] = op.xs[k]
    end
    @turbo for k in 1:D
        stabtab.tableau[op.qudits[k]+n, i] = op.zs[k]
    end
    !stabtab.storephase && return
    stabtab.tableau[2n+1, i] = op.phase
    nothing
end
