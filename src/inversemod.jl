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
        new{typeof(lookuptable)}(lookuptable)
    end
end

@inline (jit::JustInTimeInvMod)(x::Int, d::Int) = Base.invmod(x, d)
@inline (pre::PrecomputedInvMod)(x::Int, d::Int) = (@inbounds pre.lookuptable[x])
