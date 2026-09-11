##### modular-inverse strategies #####
#
# CONTRACT: an `InverseMod` maps `(x, d)` with `0 < x < d` to the inverse of `x`
# modulo `d`, as an `Int`.
#
# `Int` throughout, deliberately. The rest of the package is `Int` arithmetic --
# `binom2_mod_oddprime` takes `Int`, and `measure!` forms
# `mod(-commutator * inv, d)`. Handed an unsigned inverse, that negation and
# multiply evaluate in unsigned arithmetic and wrap BEFORE the `mod`, silently
# leaving the tableau with non-commuting generators. Keeping the table `Int`
# means that cannot arise anywhere, rather than being patched downstream.
#
# `PrecomputedInvMod` therefore stores a `Vector{Int}` and converts whatever it
# is given. A table of a narrower or unsigned integer type is converted on
# construction; a non-integer one is rejected there, so the failure lands where
# the mistake is. If you want to avoid storing a table at all, that is what
# `JustInTimeInvMod` is for.

abstract type InverseMod end

struct JustInTimeInvMod <: InverseMod end

struct PrecomputedInvMod <: InverseMod
    lookuptable::Vector{Int}

    function PrecomputedInvMod(d::Integer)
        di = Int(d)
        !Primes.isprime(di) && throw(ArgumentError("Qudit dimension d must be a prime number."))
        lookuptable = ones(Int, di - 1)
        for x in 2:(di - 1)
            lookuptable[x] = invmod(x, di)
        end
        new(lookuptable)
    end

    function PrecomputedInvMod(lookuptable::AbstractVector)
        eltype(lookuptable) <: Integer || throw(ArgumentError(
            "InverseMod lookup tables must hold Integer values; got eltype $(eltype(lookuptable))."))
        new(Vector{Int}(lookuptable))
    end
end

@inline (jit::JustInTimeInvMod)(x::Int, d::Int) = Base.invmod(x, d)
@inline (pre::PrecomputedInvMod)(x::Int, d::Int) = (@inbounds pre.lookuptable[x])
