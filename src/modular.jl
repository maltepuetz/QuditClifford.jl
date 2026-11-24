##### branch-free modular arithmetic for the tableau inner loops #####
#
# Every routine in this file assumes its inputs are ALREADY REDUCED into
# [0, d).  Tableau entries, destabilizer entries and the operator workspaces
# all satisfy that: `_build_stabilizer_tableau` and `_build_destabilizer_tableau`
# both reduce raw input (the latter also builds the destabilizers, which these
# primitives also write), the preset fillers write reduced values, and every
# `set_operator!` method reduces each exponent on write.  That invariant is what turns the outer
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

# Element types of the X and Z blocks of a tableau (the second type parameter
# of `StabilizerTableau` and `DestabilizerTableau`): `Int` for every d, and the
# two unsigned types for the d they hold (see `_eltype_admissible`).
const _NarrowEltype = Union{UInt8, UInt16}
const _StorageEltype = Union{UInt8, UInt16, Int}

########################
# Barrett reduction    #
########################

# For 0 <= x <= (d-1)^2, `(x*M) >> K == div(x, d)` whenever `_barrett_ok`.
# K = 20 keeps the largest accepted intermediate (d-1)^2 * M below 4.7e8, i.e.
# inside Int32.
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
# dst .= mod.(dst .+ a .* src, d)            #
##############################################
#
# Tiers, resolved once per call:
#   a == 0    nothing to do
#   a == 1    add, then one conditional subtract        (no multiply)
#   a == d-1  +(d-1) == -1 (mod d): subtract, then one conditional add
#   Barrett   multiply, shift-reduce, then conditional subtract
#   otherwise one `mod` per entry of `dst + a*src`
#
# The two multiply-free tiers cover every multiplier that occurs at d = 2 and
# d = 3: in `canonicalize!` the multiplier is a nonzero tableau entry, and in
# `measure!` it is mod(-commutator * inv(comm0, d), d) -- both in [1, d-1].
# The `ci` profile also runs d = 5, whose mid-circuit leaves do reach Barrett;
# its constructor-based leaves still do not, because a `:ghz` or `:product`
# tableau only ever holds 0, 1 and d-1.
#
# The `a == 1` body adds two stored values, so it is exact for d <= 2^62 and
# wraps above that. That ceiling never binds in practice: the package-wide
# envelope (`max(n, 2) * (d-1)^2 <= typemax(Int)`, see `_warn_if_dimension_unsafe`
# in helper.jl and the conventions page) caps d near 2^31 already. The
# `a == d-1` body subtracts, so it is exact for every d.
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
            # In [0, (d-1) + (d-1)^2], inside `Int` under the envelope, so one
            # `mod` is exact.
            @inbounds @simd for i in eachindex(dst)
                dst[i] = mod(dst[i] + a * src[i], d)
            end
        end
    end
    return dst
end

##############################################
# dst .= mod.(dst .- a .* src, d)            #
##############################################
# dst - a*src == dst + (d - a)*src (mod d), and d - a is reduced for 0 < a < d,
# so `addmul_mod!` with d - a serves every tier: a == 1 here is its a == d-1
# tier and a == d-1 its a == 1 tier. The narrow and mixed methods below forward
# the same way.
@inline function submul_mod!(dst::AbstractVector{Int}, src::AbstractVector{Int}, a::Int, d::Int)
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("submul_mod!", dst, src)
    a == 0 && return dst
    return @inbounds addmul_mod!(dst, src, d - a, d)
end

##################################################################
# dst[q] = mod(dst[q] ± a*src[q], d) on the qudits of a site set #
##################################################################
# The site-restricted forms of `addmul_mod!` and `submul_mod!` for a generator
# whose X and Z parts are separate vectors of length `n`: they touch only
# `dstx[q]` and `dstz[q]` for each site `q` in `sites[1:nsites]`, and leave every
# other entry alone. A caller that knows `src` vanishes outside those sites gets
# the same result as the full primitive at O(nsites) instead of O(n).
#
# Same preconditions as the full primitives (reduced entries and multiplier,
# `dstx` and `dstz` not aliased with `srcx` or `srcz`); additionally the sites
# are distinct and lie in `1:n`, which is checked by nobody since the loop runs
# under `@inbounds`. The gathered indices rule out `@turbo`, so the tier is
# chosen once and the loop body is scalar.
#
# `dst` and `src` may be any storage element types `T` and `S`: entries are
# read as `Int`, the tiers run in `Int` arithmetic, and results are stored as
# `T`.
@inline function addmul_mod_sites!(dstx::AbstractVector{T}, dstz::AbstractVector{T},
                                   srcx::AbstractVector{S}, srcz::AbstractVector{S},
                                   sites::AbstractVector{Int}, nsites::Int,
                                   a::Int, d::Int) where {T<:_StorageEltype, S<:_StorageEltype}
    a == 0 && return nothing
    if a == 1
        @inbounds for k in 1:nsites
            q = sites[k]
            r = Int(dstx[q]) + Int(srcx[q])
            dstx[q] = (r - ifelse(r >= d, d, 0)) % T
            r = Int(dstz[q]) + Int(srcz[q])
            dstz[q] = (r - ifelse(r >= d, d, 0)) % T
        end
    elseif a == d - 1
        @inbounds for k in 1:nsites
            q = sites[k]
            r = Int(dstx[q]) - Int(srcx[q])
            dstx[q] = (r + ifelse(r < 0, d, 0)) % T
            r = Int(dstz[q]) - Int(srcz[q])
            dstz[q] = (r + ifelse(r < 0, d, 0)) % T
        end
    elseif _barrett_valid(d)
        M = _barrett_mul(d)
        @inbounds for k in 1:nsites
            q = sites[k]
            t = a * Int(srcx[q])
            t = t - ((t * M) >> _BARRETT_K) * d
            r = Int(dstx[q]) + t
            dstx[q] = (r - ifelse(r >= d, d, 0)) % T
            t = a * Int(srcz[q])
            t = t - ((t * M) >> _BARRETT_K) * d
            r = Int(dstz[q]) + t
            dstz[q] = (r - ifelse(r >= d, d, 0)) % T
        end
    else
        @inbounds for k in 1:nsites
            q = sites[k]
            dstx[q] = mod(Int(dstx[q]) + mod(a * Int(srcx[q]), d), d) % T
            dstz[q] = mod(Int(dstz[q]) + mod(a * Int(srcz[q]), d), d) % T
        end
    end
    return nothing
end

@inline function submul_mod_sites!(dstx::AbstractVector{T}, dstz::AbstractVector{T},
                                   srcx::AbstractVector{S}, srcz::AbstractVector{S},
                                   sites::AbstractVector{Int}, nsites::Int,
                                   a::Int, d::Int) where {T<:_StorageEltype, S<:_StorageEltype}
    a == 0 && return nothing
    # dst - a*src == dst + (d - a)*src (mod d), and d - a is reduced for 0 < a < d.
    return addmul_mod_sites!(dstx, dstz, srcx, srcz, sites, nsites, d - a, d)
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
# identity divisions per `_canonicalize_tableau!` on a `StabilizerTableau`, and
# 4n^2 on a `DestabilizerTableau`, whose destabilizer generators it scales too.
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

##########################################################
# UInt8 and UInt16 vectors: the same four primitives     #
##########################################################
#
# Same contract as the `Int` methods above: reduced entries and multiplier,
# `d <= typemax(E)`, non-aliasing `dst` and `src` with identical axes (a
# `@boundscheck`). Tiers, resolved once per call:
#   a == 0    nothing to do (the writers write zeros)
#   d == 2    XOR for the accumulating pair (a is then 1)
#   a == 1    one add or subtract with a correction by d
#   a == d-1  the mirrored form: -s == d - s (mod d), and d - s lies in [1, d]
#   otherwise Shoup multiplication
# The `a == 1` and `a == d-1` tiers never form x + s: an add is written as
# x - (d - s), so no tier carries out of `E` even for d close to typemax(E).
#
# Shoup multiplication, with B = 8 sizeof(E) and q = floor(a 2^B / d)
# (`_shoup_q`): for every s < 2^B, floor(s q / 2^B) is floor(a s / d) or one
# less, so p = a s - floor(s q / 2^B) d lies in [0, 2d). s q < 2^(2B) is formed
# in widen(E). The rest runs in lane type L: `E` itself when 2d <= 2^B, where
# a s and floor(s q / 2^B) d wrap in `E` but their difference p < 2^B is exact;
# otherwise widen(E), where both products stay below 2^(2B).
# The final reductions use unsigned wraparound instead of a compare: for
# 0 <= u < 2d, min(u, u - d) is u mod d, and for x, p in [0, d), with
# t = x - p, min(t, t + d) is (x - p) mod d. Both need 2d <= 2^B in L.

# floor(a 2^B / d) for B = 8 sizeof(E), formed in UInt64 so that no word size
# overflows. Requires 0 <= a < d <= typemax(E), so the quotient is below 2^B.
@inline _shoup_q(::Type{E}, a::Int, d::Int) where {E<:_NarrowEltype} =
    ((UInt64(a) << (8 * sizeof(E))) ÷ UInt64(d)) % widen(E)

# The lane type of the Shoup tier: `E` when 2d <= 2^B, else widen(E).
@inline _shoup_in_eltype(::Type{E}, d::Int) where {E<:_NarrowEltype} =
    2d <= Int(typemax(E)) + 1

# p = a s mod d for one entry s < 2^B, in lane type L; see the Shoup notes.
# Before the final min, p lies in [0, 2d).
@inline function _shoup_mod(s::E, aL::L, q, dL::L) where {E<:_NarrowEltype, L}
    h = ((widen(E)(s) * q) >> (8 * sizeof(E))) % L
    p = aL * L(s) - h * dL
    return min(p, p - dL)
end

@inline function _addmul_shoup!(dst::AbstractVector{E}, src::AbstractVector{E},
                                a::Int, d::Int, ::Type{L}) where {E<:_NarrowEltype, L}
    aL = a % L; dL = d % L; q = _shoup_q(E, a, d)
    @inbounds @simd for i in eachindex(dst)
        p = _shoup_mod(src[i], aL, q, dL)       # in [0, d)
        u = L(dst[i]) + p                       # in [0, 2d)
        dst[i] = min(u, u - dL) % E
    end
    return dst
end

@inline function _mulcopy_shoup!(dst::AbstractVector{E}, src::AbstractVector{E},
                                 a::Int, d::Int, ::Type{L}) where {E<:_NarrowEltype, L}
    aL = a % L; dL = d % L; q = _shoup_q(E, a, d)
    @inbounds @simd for i in eachindex(dst)
        dst[i] = _shoup_mod(src[i], aL, q, dL) % E
    end
    return dst
end

@inline function submul_mod!(dst::AbstractVector{E}, src::AbstractVector{E},
                             a::Int, d::Int) where {E<:_NarrowEltype}
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("submul_mod!", dst, src)
    a == 0 && return dst
    return @inbounds addmul_mod!(dst, src, d - a, d)
end

@inline function addmul_mod!(dst::AbstractVector{E}, src::AbstractVector{E},
                             a::Int, d::Int) where {E<:_NarrowEltype}
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("addmul_mod!", dst, src)
    a == 0 && return dst
    dd = d % E
    if d == 2
        @inbounds @simd for i in eachindex(dst)
            dst[i] = xor(dst[i], src[i])
        end
    elseif a == 1
        @inbounds @simd for i in eachindex(dst)
            x = dst[i]
            t = dd - src[i]                     # in [1, d]
            dst[i] = (x - t) + ifelse(x < t, dd, zero(E))
        end
    elseif a == d - 1
        @inbounds @simd for i in eachindex(dst)
            x = dst[i]
            s = src[i]
            dst[i] = (x - s) + ifelse(x < s, dd, zero(E))
        end
    elseif _shoup_in_eltype(E, d)
        _addmul_shoup!(dst, src, a, d, E)
    else
        _addmul_shoup!(dst, src, a, d, widen(E))
    end
    return dst
end

@inline function mulcopy_mod!(dst::AbstractVector{E}, src::AbstractVector{E},
                              a::Int, d::Int) where {E<:_NarrowEltype}
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("mulcopy_mod!", dst, src)
    dd = d % E
    if a == 0
        fill!(dst, zero(E))
    elseif a == 1
        copyto!(dst, src)
    elseif a == d - 1
        @inbounds @simd for i in eachindex(dst)
            s = src[i]
            dst[i] = ifelse(s == zero(E), zero(E), dd - s)
        end
    elseif _shoup_in_eltype(E, d)
        _mulcopy_shoup!(dst, src, a, d, E)
    else
        _mulcopy_shoup!(dst, src, a, d, widen(E))
    end
    return dst
end

@inline function scale_mod!(dst::AbstractVector{E}, a::Int, d::Int) where {E<:_NarrowEltype}
    a == 1 && return dst
    dd = d % E
    if a == 0
        fill!(dst, zero(E))
    elseif a == d - 1
        @inbounds @simd for i in eachindex(dst)
            s = dst[i]
            dst[i] = ifelse(s == zero(E), zero(E), dd - s)
        end
    elseif _shoup_in_eltype(E, d)
        # `dst` is passed as its own `src`, the one exception to the
        # no-aliasing rule: the loop reads and writes entry i in the same
        # iteration and touches no other entry.
        _mulcopy_shoup!(dst, dst, a, d, E)
    else
        _mulcopy_shoup!(dst, dst, a, d, widen(E))
    end
    return dst
end

##############################################
# dst and src of different storage types     #
##############################################
# Chosen only when the element types differ; equal types take the methods
# above. Entries are read as `Int`, every tier runs in `Int` arithmetic, and
# results are stored as `eltype(dst)`. Tiers, resolved once per call, as for
# the `Int` methods: a == 0, a == 1 and a == d-1 (no multiply; sums and
# differences of two reduced entries lie in (-d, 2d)), Barrett where
# `_barrett_valid(d)` (a*s <= (d-1)^2, the range the constant is validated
# for), and otherwise one `mod` of a value within (d-1) + (d-1)^2 of zero,
# inside `Int` under the package-wide envelope
# `max(n, 2) * (d-1)^2 <= typemax(Int)`. `residual_from_coeffs!` subtracts
# narrow generators from the `Int` `res_workspace` through `submul_mod!`.

# dst .= mod.(dst .+ a .* src, d) for 0 < a < d.
@inline function _addmul_mixed!(dst::AbstractVector{<:_StorageEltype},
                                src::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    T = eltype(dst)
    if a == 1
        @inbounds @simd for i in eachindex(dst)
            u = Int(dst[i]) + Int(src[i])               # in [0, 2d - 2]
            dst[i] = (u - ifelse(u >= d, d, 0)) % T
        end
    elseif a == d - 1
        @inbounds @simd for i in eachindex(dst)
            u = Int(dst[i]) - Int(src[i])               # in [-(d - 1), d - 1]
            dst[i] = (u + ifelse(u < 0, d, 0)) % T
        end
    elseif _barrett_valid(d)
        M = _barrett_mul(d)
        @inbounds @simd for i in eachindex(dst)
            t = a * Int(src[i])                         # in [0, (d - 1)^2]
            u = Int(dst[i]) + (t - ((t * M) >> _BARRETT_K) * d)   # in [0, 2d - 2]
            dst[i] = (u - ifelse(u >= d, d, 0)) % T
        end
    else
        @inbounds @simd for i in eachindex(dst)
            dst[i] = mod(Int(dst[i]) + a * Int(src[i]), d) % T
        end
    end
    return dst
end

@inline function addmul_mod!(dst::AbstractVector{<:_StorageEltype},
                             src::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("addmul_mod!", dst, src)
    a == 0 && return dst
    return _addmul_mixed!(dst, src, a, d)
end

# dst - a*src == dst + (d - a)*src (mod d), and d - a is reduced for 0 < a < d.
@inline function submul_mod!(dst::AbstractVector{<:_StorageEltype},
                             src::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("submul_mod!", dst, src)
    a == 0 && return dst
    return _addmul_mixed!(dst, src, d - a, d)
end

@inline function mulcopy_mod!(dst::AbstractVector{<:_StorageEltype},
                              src::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("mulcopy_mod!", dst, src)
    T = eltype(dst)
    if a == 0
        fill!(dst, zero(T))
    elseif a == 1
        @inbounds @simd for i in eachindex(dst)
            dst[i] = Int(src[i]) % T
        end
    elseif a == d - 1
        @inbounds @simd for i in eachindex(dst)
            s = Int(src[i])
            dst[i] = ifelse(s == 0, 0, d - s) % T
        end
    elseif _barrett_valid(d)
        M = _barrett_mul(d)
        @inbounds @simd for i in eachindex(dst)
            t = a * Int(src[i])                         # in [0, (d - 1)^2]
            dst[i] = (t - ((t * M) >> _BARRETT_K) * d) % T
        end
    else
        @inbounds @simd for i in eachindex(dst)
            dst[i] = mod(a * Int(src[i]), d) % T
        end
    end
    return dst
end
