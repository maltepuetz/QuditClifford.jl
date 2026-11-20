##### Rank workspaces #####
#
# The rank-based quantities (entropies, profiles, mutual information,
# negativity, generator rank) copy the rows they need into a workspace and
# eliminate there, leaving the tableau unchanged. Each tableau holds one
# workspace for each case:
#
#   d == 2      bits packed into UInt64 words (`bit_workspace`)
#   d >= 3      `rank_workspace`, 2n x n of the storage element type `E`,
#               eliminated with the primitives in src/modular.jl
#
# Every entry point branches on `d == 2` and passes the selected workspace to
# one routine. Each elimination streams the remaining matrix once per pivot,
# so the cost at large n is set by how many bytes an entry takes, not by the
# arithmetic.

# 2n x n of `E` for d >= 3, else empty.
_storage_rank_workspace(::Type{E}, d::Int, n::Int) where {E} =
    d != 2 ? zeros(E, 2n, n) : zeros(E, 0, 0)

# For d == 2. A column holds the X and Z exponents of n sites as 2n bits, in
# up to three row blocks that each start on a fresh word (see
# `_copy_sites!`). Splitting the 2n bits into three word-aligned blocks takes
# at most two words more than cld(2n, 64); `_negativity_from_mask!` shows the
# same bound for its two blocks and the bit matrix it stacks below them.
_bit_rank_workspace(d::Int, n::Int) =
    d == 2 ? zeros(UInt64, cld(2n, 64) + 2, n) : zeros(UInt64, 0, 0)

# The rank workspace that `tab.d` selects.
@inline _rank_workspace(tab) = tab.d == 2 ? tab.bit_workspace : tab.rank_workspace

# A caller's vector that shares memory with a rank workspace is copied before
# the workspace is written.
@inline function _unalias_rank_workspaces(tab, v)
    v = Base.unalias(tab.bit_workspace, v)
    return Base.unalias(tab.rank_workspace, v)
end

# `dst .= mod.(dst .- a .* src, d)` for any storage element type.
@inline _submul_col!(dst::AbstractVector, src::AbstractVector, a::Int, d::Int) =
    submul_mod!(dst, src, a, d)

# Exchange columns `c` and `j` of `A` on `rows`.
@inline function _swap_cols!(A::AbstractMatrix, rows::UnitRange{Int}, c::Int, j::Int)
    @inbounds @simd for i in rows
        t = A[i, c]
        A[i, c] = A[i, j]
        A[i, j] = t
    end
    return A
end

# Copy the X and Z exponents of `sites[1:N]` from the generators of `b` into `ws` after row
# `offset`: X to rows offset+1:offset+N, Z to offset+N+1:offset+2N, columns
# 1:m. Returns offset + 2N.
function _copy_sites!(
    ws::AbstractMatrix{Int},
    b::_GMBlock,
    m::Int,
    sites::AbstractVector,
    N::Int,
    offset::Int,
)
    X, Z = b.x, b.z
    @turbo for j in 1:m, r in 1:N
        q = sites[r]
        ws[offset + r, j] = X[q, j]
        ws[offset + N + r, j] = Z[q, j]
    end
    return offset + 2N
end

function _copy_sites!(
    ws::AbstractMatrix{T},
    b::_GMBlock,
    m::Int,
    sites::AbstractVector,
    N::Int,
    offset::Int,
) where {T<:_NarrowEltype}
    X, Z = b.x, b.z
    @inbounds for j in 1:m
        @simd for r in 1:N
            q = sites[r]
            ws[offset + r, j] = X[q, j] % T
            ws[offset + N + r, j] = Z[q, j] % T
        end
    end
    return offset + 2N
end

# Qudit-major: site q's exponents over the generators are the contiguous
# columns `b.x[:, q]` and `b.z[:, q]`. A transpose in tiles of 64 sites by 64
# generators, sites outer: within a tile, 64 source columns are read as
# 64-entry runs, and the workspace is written one generator column at a time.
# Measured on an M3 Pro (UInt8, d = 3, half the sites, n = 1023-2049), this
# took 0.75-0.8x the time of tiles of 8 sites by 64 generators with the
# generators outer, and is still 5-8x slower than the generator-major copy.
function _copy_sites!(
    ws::AbstractMatrix{T},
    b::_QMBlock,
    m::Int,
    sites::AbstractVector,
    N::Int,
    offset::Int,
) where {T<:_StorageEltype}
    X, Z = b.x, b.z
    @inbounds for r0 in 1:64:N
        r1 = min(r0 + 63, N)
        for j0 in 1:64:m
            j1 = min(j0 + 63, m)
            for j in j0:j1
                @simd for r in r0:r1
                    q = sites[r]
                    ws[offset + r, j] = X[j, q] % T
                    ws[offset + N + r, j] = Z[j, q] % T
                end
            end
        end
    end
    return offset + 2N
end

# On a bit-packed workspace, `offset` counts bits and is a multiple of 64. The
# block takes rows offset+1:offset+2N as bits, packed as below, and the return
# value is rounded up to the next multiple of 64, so that the next block
# starts on a fresh word. The bits in between are zero.
function _copy_sites!(
    ws::AbstractMatrix{UInt64},
    b::XZBlock,
    m::Int,
    sites::AbstractVector,
    N::Int,
    offset::Int,
)
    offset & 63 == 0 || throw(ArgumentError(
        "a bit-packed block must start on a word boundary; got bit offset $offset."))
    nw = _pack_sites_gf2!(view(ws, ((offset >> 6) + 1):size(ws, 1), 1:m), b, m, sites, N)
    return offset + 64nw
end

# The row after a block of `N` sites that starts after row `offset`, as
# returned by `_copy_sites!`.
@inline _block_end(::AbstractMatrix, offset::Int, N::Int) = offset + 2N
@inline _block_end(::AbstractMatrix{UInt64}, offset::Int, N::Int) = offset + 64cld(2N, 64)

# `rank_fp_cols!` on rows r0+1:r1 and columns `cols` of a rank workspace,
# pivoting on rows r0+1:r0+pivot_rows. On a bit-packed workspace the rows are
# bits, and r0 is a multiple of 64 as `_copy_sites!` leaves it.
@inline _block_rank!(
    ws::AbstractMatrix{<:_StorageEltype}, r0::Int, r1::Int, cols::UnitRange{Int},
    d::Int, inversemod::InverseMod, pivot_rows::Int=r1 - r0,
) = rank_fp_cols!(view(ws, (r0 + 1):r1, cols), d, inversemod, pivot_rows)

@inline function _block_rank!(
    ws::AbstractMatrix{UInt64}, r0::Int, r1::Int, cols::UnitRange{Int},
    ::Int, ::InverseMod, pivot_rows::Int=r1 - r0,
)
    r0 & 63 == 0 || throw(ArgumentError(
        "a bit-packed block must start on a word boundary; got bit offset $r0."))
    return _rank_gf2_cols!(view(ws, ((r0 >> 6) + 1):cld(r1, 64), cols), pivot_rows)
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
    b::_GMBlock,
    m::Int,
    sites::AbstractVector,
    N::Int,
)
    nw = cld(2N, 64)
    X, Z = b.x, b.z
    @inbounds for j in 1:m
        for w in 1:nw
            k0 = 32 * (w - 1)
            acc = zero(UInt64)
            @simd for t in 1:min(32, N - k0)
                q = sites[k0 + t]
                acc |= ((X[q, j] % UInt64) << (2t - 2)) |
                       ((Z[q, j] % UInt64) << (2t - 1))
            end
            W[w, j] = acc
        end
    end
    return nw
end

# Qudit-major: tiles of 64 generators by the 32 sites of one word. Within a
# tile, each site's 64-entry runs of `b.x[:, q]` and `b.z[:, q]` are read in
# order and OR-ed into the 64 words of the tile, which stay in L1. Measured on
# an M3 Pro (d = 2, UInt8, n = 1024-2048, half the sites), this took 0.7x the
# time of forming each word in a register from 32 sites, which reads 32
# columns per generator.
function _pack_sites_gf2!(
    W::AbstractMatrix{UInt64},
    b::_QMBlock,
    m::Int,
    sites::AbstractVector,
    N::Int,
)
    nw = cld(2N, 64)
    X, Z = b.x, b.z
    @inbounds for j0 in 1:64:m
        j1 = min(j0 + 63, m)
        for w in 1:nw
            k0 = 32 * (w - 1)
            @simd for j in j0:j1
                W[w, j] = zero(UInt64)
            end
            for t in 1:min(32, N - k0)
                q = sites[k0 + t]
                @simd for j in j0:j1
                    W[w, j] |= ((X[j, q] % UInt64) << (2t - 2)) |
                               ((Z[j, q] % UInt64) << (2t - 1))
                end
            end
        end
    end
    return nw
end

# Rank over GF(2) of the columns of `W`, packed as above, on rows
# 1:pivot_bits. Top-down column elimination like `rank_fp_cols!`, with the
# same contract: pivots come from rows 1:pivot_bits only, the XOR updates run
# over every word of `W`, and with `piv` pivots found, columns piv+1:m are
# zero on rows 1:pivot_bits and a basis of the combinations of the input
# columns that vanish there. Once row `r` has been processed, every column
# from the current one on is zero on rows 1:r, so the updates start at the
# word holding row r.
function _rank_gf2_cols!(W::AbstractMatrix{UInt64}, pivot_bits::Int)
    nw, m = size(W)
    0 <= pivot_bits <= 64nw || throw(DimensionMismatch(
        "_rank_gf2_cols!: $pivot_bits rows do not fit in $nw words."))
    r = 1
    c = 1
    piv = 0
    @inbounds while r <= pivot_bits && c <= m
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

# The matrix K[i, j] = x_i·z_j + z_i·x_j over GF(2) of the columns
# piv+1:piv+k of `W`, restricted to the site rows in words w0+1:w0+nw, packed
# into rows 1:k of words wK+1:wK+cld(k, 64) and columns 1:k. Bits 2t-2 and
# 2t-1 of a word hold the X and Z exponents of one site, so masking the even
# bits after a one-bit shift pairs x of one column with z of the other. The
# parity of a sum of popcounts is the parity of the popcount of their XOR, so
# one accumulator per pair suffices. K is symmetric with a zero diagonal. The
# words of K must not overlap words w0+1:w0+nw.
function _symplectic_gram_gf2!(
    W::AbstractMatrix{UInt64},
    w0::Int,
    nw::Int,
    piv::Int,
    k::Int,
    wK::Int,
)
    even = 0x5555555555555555
    nwK = cld(k, 64)
    checkbounds(W, (w0 + 1):(w0 + nw), (piv + 1):(piv + k))
    checkbounds(W, (wK + 1):(wK + nwK), 1:k)
    @inbounds for j in 1:k, l in (wK + 1):(wK + nwK)
        W[l, j] = zero(UInt64)
    end
    @inbounds for j in 2:k
        cj = piv + j
        wj = wK + ((j - 1) >> 6) + 1
        bj = one(UInt64) << ((j - 1) & 63)
        for i in 1:(j - 1)
            ci = piv + i
            acc = zero(UInt64)
            @simd for l in (w0 + 1):(w0 + nw)
                a = W[l, ci]
                b = W[l, cj]
                acc = xor(acc, xor(a & (b >> 1), (a >> 1) & b))
            end
            p = (count_ones(acc & even) & 1) % UInt64
            W[wK + ((i - 1) >> 6) + 1, j] |= p << ((i - 1) & 63)
            W[wj, i] |= p * bj
        end
    end
    return W
end
