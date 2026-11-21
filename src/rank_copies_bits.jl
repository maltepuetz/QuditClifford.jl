##### packed blocks into the rank workspaces #####
#
# The word-level forms of `_pack_sites_gf2!` and `_copy_generator_rows!` for
# `BitBlock`, in both layouts. Each writes exactly the workspace words that
# the `UInt8` forms write, with the same contents.
#
# A 64 × 64 bit tile that has to be transposed is gathered into 64 entries of
# one workspace row, transposed there by `_t64!`, and left in place, so no
# scratch is needed. A tile that does not fill its 64 workspace entries is
# built bit by bit instead, so nothing outside the written region is touched.

# The low 32 bits of `v` moved to the even bit positions.
@inline function _spread32(v::UInt64)
    v = (v | (v << 16)) & 0x0000ffff0000ffff
    v = (v | (v << 8)) & 0x00ff00ff00ff00ff
    v = (v | (v << 4)) & 0x0f0f0f0f0f0f0f0f
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v
end

# The 32 bits of word column `j` of `A` starting at 0-based bit `p`, LSB
# first, in the low half of a word.
@inline function _bits32(A::AbstractMatrix{UInt64}, j::Int, p::Int)
    w = (p >> _WORD_SHIFT) + 1
    o = p & _WORD_MASK
    v = @inbounds A[w, j] >> o
    if o > _WORD_BITS >> 1 && w < size(A, 1)
        v |= @inbounds A[w + 1, j] << (_WORD_BITS - o)
    end
    return v & 0x00000000ffffffff
end

# Site `q` of generator `j`'s word column, as a bit in position `s` of a word.
@inline _site_bit(A::AbstractMatrix{UInt64}, q::Int, j::Int, s::Int) =
    @inbounds ((A[_word_index(q), j] >> _bit_offset(q)) & UInt64(1)) << s

# Generator-major. Site `k` of a generator takes bits `2k-2` and `2k-1`. When
# `sites` is a contiguous ascending range, the 32 sites of a word are 32
# consecutive bits of each of the X and Z columns: they are spread to the
# even bits and OR-ed. Any other site list is read bit by bit.
function _pack_sites_gf2!(
    W::AbstractMatrix{UInt64},
    b::_GMBits,
    m::Int,
    sites::AbstractVector,
    N::Int,
)
    nw = cld(2N, _WORD_BITS)
    checkbounds(W, 1:nw, 1:m)
    X, Z = b.x, b.z
    half = _WORD_BITS >> 1
    if sites isa AbstractUnitRange && N > 0 && first(sites) >= 1 && first(sites) + N - 1 <= b.n
        s0 = Int(first(sites)) - 1
        @inbounds for j in 1:m
            for w in 1:nw
                k0 = half * (w - 1)
                cnt = min(half, N - k0)
                mask = (UInt64(1) << cnt) - UInt64(1)
                xs = _bits32(X, j, s0 + k0) & mask
                zs = _bits32(Z, j, s0 + k0) & mask
                W[w, j] = _spread32(xs) | (_spread32(zs) << 1)
            end
        end
    else
        @inbounds for j in 1:m
            for w in 1:nw
                k0 = half * (w - 1)
                acc = zero(UInt64)
                for t in 1:min(half, N - k0)
                    q = Int(sites[k0 + t])
                    acc |= _site_bit(X, q, j, 2t - 2) | _site_bit(Z, q, j, 2t - 1)
                end
                W[w, j] = acc
            end
        end
    end
    return nw
end

# Qudit-major. The 64 generators of word `G` of site `q` are one word of
# column `q`. For the 32 sites of a workspace word `w`, the X and Z words of
# site `t` form rows `2t-2` and `2t-1` of a 64 × 64 bit tile; its transpose is
# the 64 workspace words of those generators.
function _pack_sites_gf2!(
    W::AbstractMatrix{UInt64},
    b::_QMBits,
    m::Int,
    sites::AbstractVector,
    N::Int,
)
    nw = cld(2N, _WORD_BITS)
    checkbounds(W, 1:nw, 1:m)
    X, Z = b.x, b.z
    half = _WORD_BITS >> 1
    @inbounds for G in 1:_nwords(m)
        j0 = _WORD_BITS * (G - 1)
        ng = min(_WORD_BITS, m - j0)
        for w in 1:nw
            k0 = half * (w - 1)
            cnt = min(half, N - k0)
            if ng == _WORD_BITS
                for t in 1:cnt
                    q = Int(sites[k0 + t])
                    W[w, j0 + 2t - 1] = X[G, q]
                    W[w, j0 + 2t] = Z[G, q]
                end
                for r in (2cnt + 1):_WORD_BITS
                    W[w, j0 + r] = zero(UInt64)
                end
                _t64!(view(W, w, (j0 + 1):(j0 + _WORD_BITS)), 0)
            else
                for g in 0:(ng - 1)
                    acc = zero(UInt64)
                    for t in 1:cnt
                        q = Int(sites[k0 + t])
                        acc |= (((X[G, q] >> g) & UInt64(1)) << (2t - 2)) |
                               (((Z[G, q] >> g) & UInt64(1)) << (2t - 1))
                    end
                    W[w, j0 + g + 1] = acc
                end
            end
        end
    end
    return nw
end

# Rows of G as bits, generator `j` at bit `(j-1) & 63` of word `(j-1) >> 6 + 1`:
# the X row of site `q` in words 1:cld(m, 64) of column `q`, the Z row from
# word `h + 1` on, `h` half the workspace height. Bits past `m` are zero.
#
# Qudit-major, a site's generators already are those words.
function _copy_generator_rows!(
    ws::AbstractMatrix{UInt64},
    b::_QMBits,
    m::Int,
)
    n = _nq(b)
    h = size(ws, 1) >> 1
    nw = _nwords(m)
    (nw <= h && size(ws, 2) >= n) || throw(DimensionMismatch(
        "_copy_generator_rows!: $n sites of $m generators do not fit a $(size(ws)) workspace."))
    X, Z = b.x, b.z
    tail = _tail_mask(m)
    @inbounds for q in 1:n
        for w in 1:nw
            mk = w == nw ? tail : typemax(UInt64)
            ws[w, q] = X[w, q] & mk
            ws[h + w, q] = Z[w, q] & mk
        end
    end
    return ws
end

# One 64 generator by 64 qudit tile of the generator-major block `A`, word
# `Wq` of generators `j0+1:j0+ng`, as the workspace words of sites
# `q0+1:q0+nq` in row `row`.
@inline function _gm_tile_to_row!(
    ws::AbstractMatrix{UInt64},
    A::AbstractMatrix{UInt64},
    row::Int,
    Wq::Int,
    j0::Int,
    ng::Int,
    q0::Int,
    nq::Int,
)
    if nq == _WORD_BITS
        @inbounds for t in 1:_WORD_BITS
            ws[row, q0 + t] = t <= ng ? A[Wq, j0 + t] : zero(UInt64)
        end
        _t64!(view(ws, row, (q0 + 1):(q0 + _WORD_BITS)), 0)
    else
        @inbounds for i in 0:(nq - 1)
            acc = zero(UInt64)
            for t in 1:ng
                acc |= ((A[Wq, j0 + t] >> i) & UInt64(1)) << (t - 1)
            end
            ws[row, q0 + i + 1] = acc
        end
    end
    return ws
end

# Generator-major: the 64 generators by 64 qudits of one word of generators
# and one word of qudits, transposed.
function _copy_generator_rows!(
    ws::AbstractMatrix{UInt64},
    b::_GMBits,
    m::Int,
)
    n = _nq(b)
    h = size(ws, 1) >> 1
    nw = _nwords(m)
    (nw <= h && size(ws, 2) >= n) || throw(DimensionMismatch(
        "_copy_generator_rows!: $n sites of $m generators do not fit a $(size(ws)) workspace."))
    X, Z = b.x, b.z
    @inbounds for w in 1:nw
        j0 = _WORD_BITS * (w - 1)
        ng = min(_WORD_BITS, m - j0)
        for Wq in 1:_nwords(n)
            q0 = _WORD_BITS * (Wq - 1)
            nq = min(_WORD_BITS, n - q0)
            _gm_tile_to_row!(ws, X, w, Wq, j0, ng, q0, nq)
            _gm_tile_to_row!(ws, Z, h + w, Wq, j0, ng, q0, nq)
        end
    end
    return ws
end
