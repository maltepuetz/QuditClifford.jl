abstract type InverseMod end

struct JustInTimeInvMod <: InverseMod end
struct PrecomputedInvMod{T <: AbstractVector} <: InverseMod
    lookuptable::T
    function PrecomputedInvMod(d::T) where T <: Integer
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


(jitInvMod::JustInTimeInvMod)(x::Int, d::Int) = invmod(x, d)
(preInvMod::PrecomputedInvMod)(x::Int, d::Int) = preInvMod.lookuptable[x]
