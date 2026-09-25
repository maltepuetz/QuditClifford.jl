##### branch-free modular arithmetic for the tableau inner loops #####
#
# Every routine in this file assumes its inputs are ALREADY REDUCED into
# [0, d).  Tableau entries, destabilizer entries and the operator workspaces
# all satisfy that: `_build_stabilizer_tableau` and `_build_destabilizer_tableau`
# both reduce raw input (the latter covers `destab`, which these primitives also
# write), the preset fillers write reduced values, and every `set_operator!`
# method reduces each exponent on write.  That invariant is what turns the outer
# reduction into a single conditional add or subtract instead of a division.
# Handing one of these functions an unreduced value gives a WRONG answer, not a
# slow one.
#
# `dst` and `src` must not alias.  Disjoint columns of one matrix are fine and
# are relied on; overlapping ranges are unsupported and unchecked.
#
# `dst` and `src` must also have IDENTICAL AXES.  The loops run over
# `eachindex(dst)` and index `src` with the same `i`, so a shorter `src` is an
# out-of-bounds read that `@turbo` does not check -- a plausible wrong answer
# rather than an error -- and `mulcopy_mod!`'s `copyto!` tier would silently
# stop early where the other tiers ran on.  Unlike the two preconditions above
# this one is cheap to test, so it is a `@boundscheck`: active by default, and
# elided at a caller that has already proved it with `@inbounds`.
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

# `_barrett_ok` costs two divisions and an `isqrt`, which is why the tier check
# below is a table lookup rather than a call to it: the cost is then paid once
# per call instead of being re-derived, and it never competes with the loop it
# guards on a short column.  It can only be true for
# `d <= 2^(K/2)`: `q <= 2^K - 1`, so `isqrt(q) <= 1023` and `(d-1) <= isqrt(q)`
# fails above that -- unless `d` divides `2^K` exactly, which beyond `2^(K/2)`
# means `d` is a power of two, and so never a prime.  Precompute that range.
#
# The table is DERIVED from `_barrett_ok`; it is not a hardcoded prime list.
# Acceptance is non-monotonic (131 rejected, 137/139/443 accepted), so a `d <=
# limit` test would be wrong -- a test asserts the two agree over the whole
# range, which is what stops them drifting apart.
const _BARRETT_TABLE_MAX = 1 << (_BARRETT_K >> 1)
const _BARRETT_VALID = Bool[_barrett_ok(_barrett_mul(d), d) for d in 1:_BARRETT_TABLE_MAX]

# The lower bound on the table index is load-bearing, not decorative: the read
# is `@inbounds`, so a non-positive `d` would read off the front of the table
# and return whatever was there. No caller reaches that -- both tableau
# builders reject a non-prime `d` -- but the guard is one comparison and the
# alternative is undefined behaviour. `false` is the right answer there anyway:
# it routes to the fallback tier rather than claiming a Barrett constant for a
# modulus `_barrett_mul` could not even compute.
@inline function _barrett_valid(d::Int)
    d < 1 && return false
    d <= _BARRETT_TABLE_MAX && return @inbounds _BARRETT_VALID[d]
    return ispow2(d) && ((1 << _BARRETT_K) % d == 0)
end

# `@noinline` so the message construction stays out of the inlined bodies: the
# three two-argument primitives are `@inline` at a dozen call sites, and the
# error path is never taken. (`scale_mod!` has one vector, so nothing to check.)
@noinline function _throw_axes_mismatch(fn, dst, src)
    throw(DimensionMismatch(
        "$fn: dst and src must have identical axes; got $(axes(dst)) and $(axes(src))."))
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
#   otherwise the divide-twice loop
#
# The two multiply-free tiers cover every multiplier that occurs at d = 2 and
# d = 3: in `canonicalize!` the multiplier is a nonzero tableau entry, and in
# `measure!` it is mod(-commutator * inv(comm0, d), d) -- both in [1, d-1].
# The `ci` profile also runs d = 5, whose mid-circuit leaves do reach Barrett;
# its constructor-based leaves still do not, because a `:ghz` or `:product`
# tableau only ever holds 0, 1 and d-1.
#
# The `a == d-1` body adds two stored values, so it is exact for d <= 2^62 and
# wraps above that. That ceiling never binds in practice: the package-wide
# envelope (`max(n, 2) * (d-1)^2 <= typemax(Int)`, see `_warn_if_dimension_unsafe`
# in helper.jl and the conventions page) caps d near 2^31 already. The `a == 1`
# body subtracts, so it is exact for every d.
@inline function submul_mod!(dst::AbstractVector{Int}, src::AbstractVector{Int}, a::Int, d::Int)
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("submul_mod!", dst, src)
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
        if _barrett_valid(d)
            M = _barrett_mul(d)
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
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("addmul_mod!", dst, src)
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
        if _barrett_valid(d)
            M = _barrett_mul(d)
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

##############################################
# dst .= mod.(a .* src, d)   (overwrites)    #
##############################################
# Unlike the accumulating pair, `a == 0` must WRITE zeros rather than return.
@inline function mulcopy_mod!(dst::AbstractVector{Int}, src::AbstractVector{Int}, a::Int, d::Int)
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("mulcopy_mod!", dst, src)
    if a == 0
        @turbo for i in eachindex(dst)
            dst[i] = 0
        end
    elseif a == 1
        copyto!(dst, src)
    elseif a == d - 1
        @turbo for i in eachindex(dst)
            s = src[i]
            dst[i] = ifelse(s == 0, 0, d - s)
        end
    else
        if _barrett_valid(d)
            M = _barrett_mul(d)
            @turbo for i in eachindex(dst)
                t = a * src[i]
                dst[i] = t - ((t * M) >> _BARRETT_K) * d
            end
        else
            @inbounds @simd for i in eachindex(dst)
                dst[i] = mod(a * src[i], d)
            end
        end
    end
    return dst
end

##############################################
# dst .= mod.(a .* dst, d)   (in place)      #
##############################################
# `a == 1` returns immediately: given the reduced-input invariant the loop is
# the identity. At d = 2 that is EVERY call from `canonicalize!`, since the
# pivot is nonzero (so 1) and inv(1, 2) = 1 -- so the early return skips 2n^2
# identity divisions per `_canonicalize_tableau!`, and 4n^2 per
# `_canonicalize_tableau_with_destab!`, which scales `tab` and `destab` in turn.
@inline function scale_mod!(dst::AbstractVector{Int}, a::Int, d::Int)
    a == 1 && return dst
    if a == 0
        @turbo for i in eachindex(dst)
            dst[i] = 0
        end
    elseif a == d - 1
        @turbo for i in eachindex(dst)
            s = dst[i]
            dst[i] = ifelse(s == 0, 0, d - s)
        end
    else
        if _barrett_valid(d)
            M = _barrett_mul(d)
            @turbo for i in eachindex(dst)
                t = a * dst[i]
                dst[i] = t - ((t * M) >> _BARRETT_K) * d
            end
        else
            @inbounds @simd for i in eachindex(dst)
                dst[i] = mod(a * dst[i], d)
            end
        end
    end
    return dst
end
