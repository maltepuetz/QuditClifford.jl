##### Narrow and bit-packed workspaces for rank elimination #####
#
# The tableau stores Int entries, but a rank only needs the entries mod d. The
# rank-based quantities (entropies, profiles, mutual information, negativity,
# generator rank) therefore copy the rows they need into a workspace of a
# narrower type and eliminate there:
#
#   d == 2      bits packed into UInt64 words (`bit_workspace`), when the
#               caller has a GF(2) variant, else the UInt8 workspace
#   d <= 255    one UInt8 per entry (`narrow_workspace`)
#   otherwise   the Int `workspace` and the primitives in src/modular.jl
#
# Each elimination streams the remaining matrix once per pivot, so the cost at
# large n is set by how many bytes an entry takes, not by the arithmetic.

const _NARROW_RANK_MAX_D = 255

@inline _uses_narrow_rank(d::Int) = d <= _NARROW_RANK_MAX_D

# 2n x n UInt8 for d <= 255, else empty: no caller reaches it at larger d.
_narrow_rank_workspace(d::Int, n::Int) =
    _uses_narrow_rank(d) ? zeros(UInt8, 2n, n) : zeros(UInt8, 0, 0)

# One column holds 2n bits, the X and Z exponents of n sites, for d == 2.
_bit_rank_workspace(d::Int, n::Int) =
    d == 2 ? zeros(UInt64, cld(2n, 64), n) : zeros(UInt64, 0, 0)

# A caller's vector that shares memory with a rank workspace is copied before
# the workspace is written.
@inline function _unalias_rank_workspaces(tab, v)
    v = Base.unalias(tab.bit_workspace, v)
    v = Base.unalias(tab.narrow_workspace, v)
    return Base.unalias(tab.workspace, v)
end

#############################################
# dst .= mod.(dst .- a .* src, d) on UInt8  #
#############################################
#
# Entries of `dst` and `src` lie in [0, d), d <= 255 and 1 <= a <= d - 1.
# Tiers, resolved once per call:
#   d == 2    XOR
#   a == 1    subtract, then add d back on borrow
#   a == d-1  add, written as dst - (d - src) so that it cannot carry out of
#             UInt8 for d > 128
#   otherwise Shoup multiplication: with q = floor(256 a / d), floor(s q / 256)
#             is floor(a s / d) or one less for every s < 256, so
#             p = a s - floor(s q / 256) d lies in [0, 2d). Only s q needs 16
#             bits. For d <= 127, p < 256, so everything else runs in UInt8
#             lanes and wraps harmlessly; above that, in UInt16 lanes, where
#             both products stay below 255 * 255 < 2^16.
# The Shoup tier's final reductions use unsigned wraparound instead of a
# compare: for 0 <= u < 2d, min(u, u - d) is u mod d, and for x, p in [0, d),
# with t = x - p, min(t, t + d) is (x - p) mod d. Both need 2d to fit the lane.
# `dst` and `src` must not alias.
@inline function _submul_u8!(dst::AbstractVector{UInt8}, src::AbstractVector{UInt8}, a::Int, d::Int)
    @boundscheck axes(dst) == axes(src) || _throw_axes_mismatch("_submul_u8!", dst, src)
    dd = d % UInt8
    if d == 2
        @inbounds @simd for i in eachindex(dst)
            dst[i] = xor(dst[i], src[i])
        end
    elseif a == 1
        @inbounds @simd for i in eachindex(dst)
            x = dst[i]
            s = src[i]
            dst[i] = (x - s) + ifelse(x < s, dd, 0x00)
        end
    elseif a == d - 1
        @inbounds @simd for i in eachindex(dst)
            x = dst[i]
            t = dd - src[i]                  # in [1, d]
            dst[i] = (x - t) + ifelse(x < t, dd, 0x00)
        end
    elseif d <= 127
        a8 = a % UInt8
        q16 = ((a << 8) ÷ d) % UInt16
        @inbounds @simd for i in eachindex(dst)
            s = src[i]
            h = ((UInt16(s) * q16) >> 8) % UInt8
            p = a8 * s - h * dd
            p = min(p, p - dd)
            t = dst[i] - p
            dst[i] = min(t, t + dd)
        end
    else
        a16 = a % UInt16
        q16 = ((a << 8) ÷ d) % UInt16
        d16 = d % UInt16
        @inbounds @simd for i in eachindex(dst)
            s = UInt16(src[i])
            p = a16 * s - ((s * q16) >> 8) * d16
            p = min(p, p - d16)
            t = UInt16(dst[i]) - p
            dst[i] = min(t, t + d16) % UInt8
        end
    end
    return dst
end

# `dst .= mod.(dst .- a .* src, d)` for either workspace element type.
@inline _submul_col!(dst::AbstractVector{Int}, src::AbstractVector{Int}, a::Int, d::Int) =
    submul_mod!(dst, src, a, d)
@inline _submul_col!(dst::AbstractVector{UInt8}, src::AbstractVector{UInt8}, a::Int, d::Int) =
    _submul_u8!(dst, src, a, d)

# Exchange columns `c` and `j` of `A` on `rows`.
@inline function _swap_cols!(A::AbstractMatrix, rows::UnitRange{Int}, c::Int, j::Int)
    @inbounds @simd for i in rows
        t = A[i, c]
        A[i, c] = A[i, j]
        A[i, j] = t
    end
    return A
end

# Copy the X and Z exponents of `sites[1:N]` from `stab` into `ws` after row
# `offset`: X to rows offset+1:offset+N, Z to offset+N+1:offset+2N, columns
# 1:m. Returns offset + 2N.
function _copy_sites!(
    ws::AbstractMatrix{Int},
    stab::AbstractMatrix{Int},
    n::Int,
    m::Int,
    sites::AbstractVector,
    N::Int,
    offset::Int,
)
    @turbo for j in 1:m, r in 1:N
        q = sites[r]
        ws[offset + r, j] = stab[q, j]
        ws[offset + N + r, j] = stab[n + q, j]
    end
    return offset + 2N
end

function _copy_sites!(
    ws::AbstractMatrix{UInt8},
    stab::AbstractMatrix{Int},
    n::Int,
    m::Int,
    sites::AbstractVector,
    N::Int,
    offset::Int,
)
    @inbounds for j in 1:m
        @simd for r in 1:N
            q = sites[r]
            ws[offset + r, j] = stab[q, j] % UInt8
            ws[offset + N + r, j] = stab[n + q, j] % UInt8
        end
    end
    return offset + 2N
end

##########################
# GF(2) on packed words  #
##########################
#
# A packed column holds the rows of one generator as bits, LSB first: row `i`
# is bit `(i - 1) & 63` of word `(i - 1) >> 6 + 1`. Site `sites[k]` takes rows
# 2k - 1 (X) and 2k (Z), so word `w` holds sites 32(w - 1) + 1 to 32w. Bits
# past the last row are zero.

# Pack the X and Z exponents of `sites[1:N]` into columns 1:m of `W`, and
# return the number of words per column, cld(2N, 64). Entries are 0 or 1.
function _pack_sites_gf2!(
    W::AbstractMatrix{UInt64},
    stab::AbstractMatrix{Int},
    n::Int,
    m::Int,
    sites::AbstractVector,
    N::Int,
)
    nw = cld(2N, 64)
    @inbounds for j in 1:m
        for w in 1:nw
            k0 = 32 * (w - 1)
            acc = zero(UInt64)
            @simd for t in 1:min(32, N - k0)
                q = sites[k0 + t]
                acc |= ((stab[q, j] % UInt64) << (2t - 2)) |
                       ((stab[n + q, j] % UInt64) << (2t - 1))
            end
            W[w, j] = acc
        end
    end
    return nw
end

# Rank over GF(2) of the columns of `W`, whose rows `1:nbits` are packed as
# above. Top-down column elimination like `rank_fp_cols!`: once row `r` has
# been processed, every column from the current one on is zero on rows 1:r,
# so the updates start at the word holding row r.
function _rank_gf2_cols!(W::AbstractMatrix{UInt64}, nbits::Int)
    nw, m = size(W)
    nbits <= 64nw || throw(DimensionMismatch(
        "_rank_gf2_cols!: $nbits rows do not fit in $nw words."))
    r = 1
    c = 1
    piv = 0
    @inbounds while r <= nbits && c <= m
        w = (r - 1) >> 6 + 1
        bit = one(UInt64) << ((r - 1) & 63)
        j = c
        while j <= m && iszero(W[w, j] & bit)
            j += 1
        end
        if j > m
            r += 1
            continue
        end
        j != c && _swap_cols!(W, w:nw, c, j)
        for jj in (c + 1):m
            iszero(W[w, jj] & bit) && continue
            @simd for l in w:nw
                W[l, jj] = xor(W[l, jj], W[l, c])
            end
        end
        piv += 1
        r += 1
        c += 1
    end
    return piv
end
