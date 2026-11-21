##### packed qubit storage #####
#
# At d = 2 a tableau can hold its X and Z blocks as packed bits (`eltype =
# Bits`). Each block is an `nw × 64nw` matrix of `UInt64` words, `nw =
# _nwords(n)`, bits least significant first, in a padded parent of its own:
#
# - generator-major (`_GMBits`): generator `j` is column `j`, and qudit `q` is
#   bit `(q-1) & 63` of word `(q-1) >> 6 + 1`;
# - qudit-major (`_QMBits`): qudit `q` is column `q`, and generator `j` is bit
#   `(j-1) & 63` of word `(j-1) >> 6 + 1`.
#
# The bit matrix is square in 64 × 64 strips. Every bit of a qudit or
# generator past `n` (the tail of the last word of a column, and the columns
# `n+1:64nw`) is zero, and so is every parent word outside the view.
#
# `BitBlock{L}` is a struct of its own rather than an `XZBlock`, so no
# `XZBlock` method can read a word as an entry: a block operation without a
# bit method is a `MethodError`. Its leaves take and return entries as `Int`
# 0 or 1, like the `XZBlock` leaves.
#
# The phase and the x·z cache of a packed tableau are `_BitPlanes`: `k` bit
# planes indexed by generator in both layouts, so a layout switch leaves them
# alone.
#
# Words are `UInt64` on every host, and masks are built from `UInt64(1)`, so
# no shift goes through a 32-bit `Int`.

"""
    Bits

The `eltype` keyword value for packed qubit storage: `StabilizerTableau(2, n;
eltype = Bits)` and `DestabilizerTableau(2, n; eltype = Bits)` store the X and
Z exponents as single bits, 64 to a `UInt64` word, in either layout. It is
legal only at `d = 2`; any other `d` throws an `ArgumentError`.

The choice affects memory and speed only: every result, and every matrix the
tableau returns, is the same `Int` value as with `UInt8` storage.
[`tableau_eltype`](@ref) returns `Bits` for such a tableau.

# Examples
```jldoctest
julia> tab = StabilizerTableau(2, 3; state = :ghz, eltype = Bits);

julia> tableau_eltype(tab)
Bits

julia> tab == StabilizerTableau(2, 3; state = :ghz)
true
```
"""
struct Bits end

const _WORD_BITS = 8 * sizeof(UInt64)
const _WORD_SHIFT = trailing_zeros(_WORD_BITS)
const _WORD_MASK = _WORD_BITS - 1

# Words that hold `n` bits.
_nwords(n::Int) = cld(n, _WORD_BITS)

# The word of bit index `i` (1-based), and its offset within that word.
@inline _word_index(i::Int) = (i - 1) >> _WORD_SHIFT + 1
@inline _bit_offset(i::Int) = (i - 1) & _WORD_MASK

# The bits of a word that hold indices `≤ len` of its last word, all ones
# when `len` fills it.
@inline function _tail_mask(len::Int)
    r = len & _WORD_MASK
    return r == 0 ? typemax(UInt64) : (UInt64(1) << r) - UInt64(1)
end

##### bit planes #####

"""
    _BitPlanes <: AbstractVector{Int}

A vector of `len` small integers stored as `k = size(planes, 2)` bit planes:
entry `j` is `Σ_b bit(j) of plane b · 2^(b-1)`, with bit `j` at offset
`(j-1) & 63` of word `(j-1) >> 6 + 1`. `setindex!` stores the value mod `2^k`.
The bits past `len` in the last word stay zero.
"""
struct _BitPlanes <: AbstractVector{Int}
    planes::Matrix{UInt64}             # _nwords(len) × k
    len::Int
end

# A zeroed vector of `len` entries of `k` bits.
_BitPlanes(len::Int, k::Int) = _BitPlanes(zeros(UInt64, _nwords(len), k), len)

Base.size(v::_BitPlanes) = (v.len,)
Base.IndexStyle(::Type{_BitPlanes}) = IndexLinear()

@inline function Base.getindex(v::_BitPlanes, j::Int)
    @boundscheck checkbounds(v, j)
    w, o = _word_index(j), _bit_offset(j)
    s = 0
    @inbounds for b in axes(v.planes, 2)
        s |= (((v.planes[w, b] >> o) & UInt64(1)) % Int) << (b - 1)
    end
    return s
end

@inline function Base.setindex!(v::_BitPlanes, x::Integer, j::Int)
    @boundscheck checkbounds(v, j)
    w, o = _word_index(j), _bit_offset(j)
    xi = Int(x)
    @inbounds for b in axes(v.planes, 2)
        bit = ((xi >> (b - 1)) & 1) % UInt64
        v.planes[w, b] = (v.planes[w, b] & ~(UInt64(1) << o)) | (bit << o)
    end
    return v
end

Base.copy(v::_BitPlanes) = _BitPlanes(copy(v.planes), v.len)

function Base.copyto!(dst::_BitPlanes, src::_BitPlanes)
    size(dst.planes) == size(src.planes) && dst.len == src.len ||
        throw(DimensionMismatch("bit planes of different shapes"))
    copyto!(dst.planes, src.planes)
    return dst
end

function Base.fill!(v::_BitPlanes, x::Integer)
    xi = Int(x)
    P = v.planes
    nw = size(P, 1)
    for b in axes(P, 2)
        word = (xi >> (b - 1)) & 1 == 1 ? typemax(UInt64) : UInt64(0)
        fill!(view(P, :, b), word)
        nw > 0 && (P[nw, b] &= _tail_mask(v.len))
    end
    return v
end

_zero!(v::_BitPlanes) = (fill!(v.planes, UInt64(0)); v)

# Overwrite `dst` with the entries of `src`, of the same length, mod `2^k`.
function _pack_planes!(dst::_BitPlanes, src::AbstractVector{<:Integer})
    length(dst) == length(src) || throw(DimensionMismatch("lengths differ"))
    _zero!(dst)
    @inbounds for j in eachindex(src)
        dst[j] = src[j]
    end
    return dst
end

##### packed blocks #####

struct BitBlock{L<:TableauLayout}
    x::_BlockMatrix{UInt64}            # _nwords(n) × _WORD_BITS·_nwords(n)
    z::_BlockMatrix{UInt64}
    n::Int
end

const _GMBits = BitBlock{GeneratorMajor}
const _QMBits = BitBlock{QuditMajor}

# A zeroed packed block for `n` qudits, `_nwords(n) × _WORD_BITS·_nwords(n)`
# words in a parent of its own, padded and placed in line `slot` by the rule
# of `_padded_block` applied to its word columns.
_padded_bit_block(n::Int, slot::Int) =
    _padded_view(UInt64, _nwords(n), _WORD_BITS * _nwords(n), slot)

_stab_bit_blocks(n::Int) = (_padded_bit_block(n, 0), _padded_bit_block(n, 2048))
_destab_bit_blocks(n::Int) = (_padded_bit_block(n, 1024), _padded_bit_block(n, 3072))

# A copy of a packed block for `n` qudits in a fresh parent, in the same line slot.
_copy_bit_block(A::_BlockMatrix{UInt64}, n::Int) =
    _copy_block!(_padded_bit_block(n, _block_slot(A)), A)

# A copy of a tableau block for `n` qudits in a fresh parent, in the same line
# slot: unpacked `n × n`, or packed.
_copy_gen_block(A::_BlockMatrix{T}, ::Int) where {T<:_StorageEltype} = _copy_block(A)
_copy_gen_block(A::_BlockMatrix{UInt64}, n::Int) = _copy_bit_block(A, n)

function _zero!(b::BitBlock)
    fill!(b.x, UInt64(0))
    fill!(b.z, UInt64(0))
    return b
end

@inline _nq(b::BitBlock) = b.n
@inline _ncap(b::BitBlock) = b.n

# Bit `i` of word column `c`, as `Int` 0 or 1, and its overwrite with `v`.
@inline _getbit(A::_BlockMatrix{UInt64}, i::Int, c::Int) =
    @inbounds ((A[_word_index(i), c] >> _bit_offset(i)) & UInt64(1)) % Int
@inline function _setbit!(A::_BlockMatrix{UInt64}, v::Int, i::Int, c::Int)
    w, o = _word_index(i), _bit_offset(i)
    @inbounds A[w, c] = (A[w, c] & ~(UInt64(1) << o)) | ((v & 1) % UInt64) << o
    return v
end

@inline _xget(b::_GMBits, q::Int, j::Int) = _getbit(b.x, q, j)
@inline _zget(b::_GMBits, q::Int, j::Int) = _getbit(b.z, q, j)
@inline _xset!(b::_GMBits, v::Int, q::Int, j::Int) = _setbit!(b.x, v, q, j)
@inline _zset!(b::_GMBits, v::Int, q::Int, j::Int) = _setbit!(b.z, v, q, j)
@inline _xget(b::_QMBits, q::Int, j::Int) = _getbit(b.x, j, q)
@inline _zget(b::_QMBits, q::Int, j::Int) = _getbit(b.z, j, q)
@inline _xset!(b::_QMBits, v::Int, q::Int, j::Int) = _setbit!(b.x, v, j, q)
@inline _zset!(b::_QMBits, v::Int, q::Int, j::Int) = _setbit!(b.z, v, j, q)

# The x·z dot products of src/xz_block.jl, mod `d`. Generator-major, two
# generators pair word by word; qudit-major, bit by bit along the strided rows.
@inline function dot_xz_col(b::_GMBits, i::Int, j::Int, d::Int)
    s = 0
    x, z = b.x, b.z
    @inbounds for w in axes(x, 1)
        s += count_ones(x[w, i] & z[w, j])
    end
    return mod(s, d)
end

@inline function dot_xz_col(b::_QMBits, i::Int, j::Int, d::Int)
    s = 0
    wi, oi = _word_index(i), _bit_offset(i)
    wj, oj = _word_index(j), _bit_offset(j)
    x, z = b.x, b.z
    @inbounds for q in 1:b.n
        s += (((x[wi, q] >> oi) & (z[wj, q] >> oj)) & UInt64(1)) % Int
    end
    return mod(s, d)
end
@inline dot_xz_col(b::BitBlock, j::Int, d::Int) = dot_xz_col(b, j, j, d)

@inline function dot_xz_ws_vs_col(genws::AbstractVector{<:_StorageEltype}, b::BitBlock,
                                  tgt::Int, d::Int)
    s = 0
    @inbounds for q in 1:b.n
        s += Int(genws[q]) * _zget(b, q, tgt)
    end
    return mod(s, d)
end

@inline function dot_xz_col_vs_zacc(b::BitBlock, j::Int, zacc::AbstractVector{<:_StorageEltype},
                                    d::Int)
    s = 0
    @inbounds for q in 1:b.n
        s += _xget(b, q, j) * Int(zacc[q])
    end
    return mod(s, d)
end

##### unpacked copies and packing #####

# Write the generators of the generator-major `src`, entries 0 or 1, into the
# packed `dst` in the layout of `dst`. Every bit of `dst` inside the `n × n`
# square is written.
function _pack_generator_major!(dst::BitBlock, src::_GMBlock)
    n = _nq(dst)
    _nq(src) == n && _ncap(src) == n || throw(DimensionMismatch("block sizes differ"))
    _zero!(dst)
    @inbounds for j in 1:n, q in 1:n
        _xget(src, q, j) == 0 || _xset!(dst, 1, q, j)
        _zget(src, q, j) == 0 || _zset!(dst, 1, q, j)
    end
    return dst
end

# An unpacked generator-major copy of `b` in fresh `UInt8` matrices, for code
# that works on a private generator-major scratch.
function _generator_major_copy(b::BitBlock)
    n = b.n
    x = Matrix{UInt8}(undef, n, n)
    z = Matrix{UInt8}(undef, n, n)
    @inbounds for j in 1:n, q in 1:n
        x[q, j] = _xget(b, q, j) % UInt8
        z[q, j] = _zget(b, q, j) % UInt8
    end
    return XZBlock{GeneratorMajor}(x, z)
end

# The blocks of `b` stacked generator-major as a fresh `Int` matrix, X rows
# first, with `extra` further rows left for the caller to fill.
function _stacked_int_matrix(b::BitBlock, extra::Int)
    n = b.n
    S = Matrix{Int}(undef, 2n + extra, n)
    @inbounds for j in 1:n, q in 1:n
        S[q, j] = _xget(b, q, j)
        S[n + q, j] = _zget(b, q, j)
    end
    return S
end
