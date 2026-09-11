##### modular-inverse strategies #####
#
# CONTRACT: an `InverseMod` maps `(x, d)` with `0 < x < d` to the inverse of `x`
# modulo `d`, and **every consumer receives an `Int`**.
#
# The lookup table may be stored in any integer type -- `Vector{Int32}` and
# `Vector{UInt64}` are both accepted, and a lazy `AbstractVector` works too --
# but the value is converted on read, at the call operator below, so no caller
# ever has to think about it. A valid inverse lies in `1:(d-1)` and `d` is an
# `Int`, so the conversion is always exact.
#
# That normalization is not cosmetic. The rest of the package is `Int`
# arithmetic throughout: `binom2_mod_oddprime` takes `Int`, and `measure!`
# forms `mod(-commutator * inv, d)`. Handed an unsigned inverse, that negation
# and multiply evaluate in unsigned arithmetic and wrap BEFORE the `mod`,
# which silently leaves the tableau with non-commuting generators -- no error,
# just a corrupt state. Converting once, here, is what prevents that.
#
# Non-integer tables are rejected at construction. An exactly-valued `Float64`
# table happens to work, because the result converts on assignment into the
# `Int` tableau, but one off by a single ULP would round into a wrong-but-valid
# residue. The check is at the constructor so the failure lands where the
# mistake is.

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

# The `::Int` return annotations are the normalization point for the whole
# package; see the contract note above. For an `Int` table the conversion is
# the identity, so the hot path is unchanged.
@inline (jit::JustInTimeInvMod)(x::Int, d::Int)::Int = Base.invmod(x, d)
@inline (pre::PrecomputedInvMod)(x::Int, d::Int)::Int = (@inbounds pre.lookuptable[x])
