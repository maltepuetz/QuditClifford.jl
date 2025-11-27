abstract type Operator end

abstract type FewQuditOperator <: Operator end
struct SinglePauli <: FewQuditOperator
    qudit::Int
    x::Int
    z::Int
end
struct DoublePauli <: FewQuditOperator
    qudit1::Int
    x1::Int
    z1::Int
    qudit2::Int
    x2::Int
    z2::Int
end
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
end
struct NPauli{D} <: FewQuditOperator
    qudits::NTuple{D,Int}
    xs::NTuple{D,Int}
    zs::NTuple{D,Int}
end

"""
    set_operator!(tableau, i::Int, op::SinglePauli, n::Int)
    set_operator!(tableau, i::Int, op::DoublePauli, n::Int)
    set_operator!(tableau, i::Int, op::TriplePauli, n::Int)
    set_operator!(tableau, i::Int, op::NPauli{D}, n::Int) where D
Set the i-th operator in the tableau to the given FewQuditOperator `op`.
"""
function set_operator! end
@inline function set_operator!(tableau, i::Int, op::SinglePauli, n::Int)
    @turbo for j in axes(tableau, 1)
        tableau[j, i] = 0
    end
    tableau[op.qudit, i] = op.x
    tableau[op.qudit+n, i] = op.z
    nothing
end
@inline function set_operator!(tableau, i::Int, op::DoublePauli, n::Int)
    @turbo for j in axes(tableau, 1)
        tableau[j, i] = 0
    end
    tableau[op.qudit1, i] = op.x1
    tableau[op.qudit2, i] = op.x2
    tableau[op.qudit1+n, i] = op.z1
    tableau[op.qudit2+n, i] = op.z2
    nothing
end
@inline function set_operator!(tableau, i::Int, op::TriplePauli, n::Int)
    @turbo for j in axes(tableau, 1)
        tableau[j, i] = 0
    end
    tableau[op.qudit1, i] = op.x1
    tableau[op.qudit2, i] = op.x2
    tableau[op.qudit3, i] = op.x3
    tableau[op.qudit1+n, i] = op.z1
    tableau[op.qudit2+n, i] = op.z2
    tableau[op.qudit3+n, i] = op.z3
    nothing
end
@inline function set_operator!(tableau, i::Int, op::NPauli{D}, n::Int) where D
    @turbo for j in axes(tableau, 1)
        tableau[j, i] = 0
    end
    @turbo for k in 1:D
        tableau[op.qudits[k], i] = op.xs[k]
    end
    @turbo for k in 1:D
        tableau[op.qudits[k]+n, i] = op.zs[k]
    end
    nothing
end
