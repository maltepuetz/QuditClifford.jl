##### modular-inverse strategies #####
#
# CONTRACT: an `InverseMod` maps `(x, d)` with `0 < x < d` to the inverse of `x`
# modulo `d`, and **must return an `Integer`**.
#
# The returned value's type is not incidental. It is the type the downstream
# multiplications are evaluated in -- `src/modular.jl` forms `a * src[i]` with
# `a` exactly as returned -- so the table's element type also sets the overflow
# ceiling: an `Int` table caps the package where `(d-1)^2` passes `typemax(Int)`,
# while an unsigned table of the same width raises it.
#
# Non-integer tables are rejected at construction. An exactly-valued `Float64`
# table happens to work, because the result converts on assignment into the
# `Int` tableau, but one off by a single ULP would round into a wrong-but-valid
# residue -- a silently incorrect state, which is far worse than a construction
# error. The check is at the constructor so the failure lands where the mistake
# is, rather than as a MethodError inside an inner loop.

abstract type InverseMod end

struct JustInTimeInvMod <: InverseMod end
struct PrecomputedInvMod{T <: AbstractVector} <: InverseMod
    lookuptable::T
    function PrecomputedInvMod(d::T) where T <: Integer
        !Primes.isprime(Int(d)) && throw(ArgumentError("Qudit dimension d must be a prime number."))
        lookuptable = ones(T, Int(d) - 1)
        for x in 2:Int(d)-1
            lookuptable[x] = invmod(x, Int(d))
        end
        new{typeof(lookuptable)}(lookuptable)
    end
    function PrecomputedInvMod(lookuptable::T) where T <: AbstractVector
        eltype(T) <: Integer || throw(ArgumentError(
            "InverseMod lookup tables must hold Integer values; got eltype $(eltype(T))."))
        new{typeof(lookuptable)}(lookuptable)
    end
end

@inline (jit::JustInTimeInvMod)(x::Int, d::Int) = Base.invmod(x, d)
@inline (pre::PrecomputedInvMod)(x::Int, d::Int) = (@inbounds pre.lookuptable[x])
