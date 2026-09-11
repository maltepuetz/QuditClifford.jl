##### branch-free modular arithmetic for the tableau inner loops #####
#
# Every routine in this file assumes its inputs are ALREADY REDUCED into
# [0, d).  Tableau entries, destabilizer entries and the operator workspaces
# all satisfy that: `_build_stabilizer_tableau` reduces raw input, the preset
# fillers write reduced values, and every `set_operator!` method reduces each
# exponent on write.  That invariant is what turns the outer reduction into a
# single conditional add or subtract instead of a division.  Handing one of
# these functions an unreduced value gives a WRONG answer, not a slow one.
#
# `dst` and `src` must not alias.  Disjoint columns of one matrix are fine and
# are relied on; overlapping ranges are unsupported and unchecked.
#
# Phase rows use a different modulus (`phase_modulus`) and are deliberately not
# handled here.

########################
# Barrett reduction    #
########################

# For 0 <= x <= (d-1)^2, `(x*M) >> K == div(x, d)` whenever `_barrett_ok`.
# K = 20 keeps the largest accepted intermediate (d-1)^2 * M below 4.7e8, i.e.
# inside Int32 -- forward compatibility with a narrowed element type.
# K = 12 would suffice only up to d = 19 and is NOT enough.
const _BARRETT_K = 20

@inline _barrett_mul(d::Int) = cld(1 << _BARRETT_K, d)

# With e = M*d - 2^K (so 0 <= e < d) the identity holds for all x <= X exactly
# when e*X < 2^K, and here X = (d-1)^2.  Encoding that directly as
#     (M*d - (1 << K)) * (d-1)^2 < (1 << K)
# OVERFLOWS Int and wraps negative, which then compares as valid -- first at
# the prime d = 2511241.  Test the same condition by division instead; neither
# the division nor `isqrt` can overflow.
@inline function _barrett_ok(M::Int, d::Int)
    e = M * d - (1 << _BARRETT_K)      # M*d is about max(2^K, d); safe for all d
    e == 0 && return true              # d divides 2^K exactly (d == 2)
    q = ((1 << _BARRETT_K) - 1) ÷ e
    return (d - 1) <= isqrt(q)         # (d-1)^2 <= q, without forming the product
end

##############################################
# dst .= mod.(dst .- a .* src, d)            #
##############################################
#
# Tiers, resolved once per call:
#   a == 0    nothing to do
#   a == 1    subtract, then one conditional add        (no multiply)
#   a == d-1  -(d-1) == +1 (mod d): add, then one conditional subtract
#   Barrett   multiply, shift-reduce, then conditional add
#   otherwise the original divide-twice loop, unchanged
#
# The two multiply-free tiers cover every multiplier that occurs at d = 2 and
# d = 3, which is the whole `ci` benchmark profile: in `canonicalize!` the
# multiplier is a nonzero tableau entry, and in `measure!` it is
# mod(-commutator * inv(comm0, d), d) -- both in [1, d-1].
#
# The `a == d-1` body adds two stored values, so it is exact for d <= 2^62 and
# wraps above that; see the plan's correctness envelope. The `a == 1` body
# subtracts, so it is exact for every d.
@inline function submul_mod!(dst::AbstractVector{Int}, src::AbstractVector{Int}, a::Int, d::Int)
    a == 0 && return dst
    if a == 1
        @turbo for i in eachindex(dst)
            r = dst[i] - src[i]
            dst[i] = r + ifelse(r < 0, d, 0)
        end
    elseif a == d - 1
        @turbo for i in eachindex(dst)
            r = dst[i] + src[i]
            dst[i] = r - ifelse(r >= d, d, 0)
        end
    else
        M = _barrett_mul(d)
        if _barrett_ok(M, d)
            @turbo for i in eachindex(dst)
                t = a * src[i]
                t = t - ((t * M) >> _BARRETT_K) * d
                r = dst[i] - t
                dst[i] = r + ifelse(r < 0, d, 0)
            end
        else
            @inbounds @simd for i in eachindex(dst)
                dst[i] = mod(dst[i] - mod(a * src[i], d), d)
            end
        end
    end
    return dst
end

##############################################
# dst .= mod.(dst .+ a .* src, d)            #
##############################################
# Mirror of `submul_mod!` with the signs exchanged, so here `a == 1` is the add
# form (ceiling d <= 2^62) and `a == d-1` is the subtract form (exact always).
@inline function addmul_mod!(dst::AbstractVector{Int}, src::AbstractVector{Int}, a::Int, d::Int)
    a == 0 && return dst
    if a == 1
        @turbo for i in eachindex(dst)
            r = dst[i] + src[i]
            dst[i] = r - ifelse(r >= d, d, 0)
        end
    elseif a == d - 1
        @turbo for i in eachindex(dst)
            r = dst[i] - src[i]
            dst[i] = r + ifelse(r < 0, d, 0)
        end
    else
        M = _barrett_mul(d)
        if _barrett_ok(M, d)
            @turbo for i in eachindex(dst)
                t = a * src[i]
                t = t - ((t * M) >> _BARRETT_K) * d
                r = dst[i] + t
                dst[i] = r - ifelse(r >= d, d, 0)
            end
        else
            @inbounds @simd for i in eachindex(dst)
                dst[i] = mod(dst[i] + mod(a * src[i], d), d)
            end
        end
    end
    return dst
end
