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

##### packed blocks #####

struct BitBlock{L<:TableauLayout}
    x::_BlockMatrix{UInt64}            # _nwords(n) × _WORD_BITS·_nwords(n)
    z::_BlockMatrix{UInt64}
    n::Int
end

const _GMBits = BitBlock{GeneratorMajor}
const _QMBits = BitBlock{QuditMajor}

# Either kind of block, for code that calls only the leaves.
const _AnyBlock = Union{XZBlock, BitBlock}

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

##### composites #####
#
# The generator operations of src/xz_block.jl on packed blocks. At d = 2 every
# multiplier `a` acts through its parity: adding or subtracting `a` times a
# generator flips the bits where that generator is 1 when `a` is odd, and
# multiply-copy writes the generator or zero. Stacked vectors `[x; z]` hold
# reduced entries 0 or 1 of a storage element type, so bit 0 of an entry is
# the entry.
#
# Generator-major, a generator is a contiguous word column: block-to-block
# forms are word loops, and vector forms pack or unpack 64 entries per word.
# Qudit-major, a generator is bit `j` of every column word, so each form sets
# or flips that bit column by column.

# Bit `i` of a word built from a stacked entry, 0 or 1.
@inline _entry_bit(v::AbstractVector{<:_StorageEltype}, i::Int) =
    @inbounds (v[i] % UInt64) & UInt64(1)

# Word `w` of the entries `v[off+1:off+n]`, packed least significant first.
@inline function _pack_word(v::AbstractVector{<:_StorageEltype}, off::Int, w::Int, n::Int)
    lo = (w - 1) << _WORD_SHIFT
    word = UInt64(0)
    @inbounds for k in 1:min(_WORD_BITS, n - lo)
        word |= _entry_bit(v, off + lo + k) << (k - 1)
    end
    return word
end

# Write bit `k-1` of `word` to entry `v[off + 64(w-1) + k]` for every such entry
# within `v[off+1:off+n]`; with `flip`, add it mod 2 instead.
@inline function _unpack_word!(v::AbstractVector{E}, off::Int, w::Int, n::Int, word::UInt64,
                               flip::Bool) where {E<:_StorageEltype}
    lo = (w - 1) << _WORD_SHIFT
    @inbounds for k in 1:min(_WORD_BITS, n - lo)
        bit = (word >> (k - 1)) & UInt64(1)
        i = off + lo + k
        v[i] = (flip ? _entry_bit(v, i) ⊻ bit : bit) % E
    end
    return v
end

# Flip bit `i` of word column `c` when `v` is 1 (`v` is 0 or 1), and entry
# `(q, j)` of a block likewise.
@inline function _flipbit!(A::_BlockMatrix{UInt64}, v::UInt64, i::Int, c::Int)
    @inbounds A[_word_index(i), c] ⊻= v << _bit_offset(i)
    return A
end
@inline _xflip!(b::_GMBits, v::UInt64, q::Int, j::Int) = _flipbit!(b.x, v, q, j)
@inline _zflip!(b::_GMBits, v::UInt64, q::Int, j::Int) = _flipbit!(b.z, v, q, j)
@inline _xflip!(b::_QMBits, v::UInt64, q::Int, j::Int) = _flipbit!(b.x, v, j, q)
@inline _zflip!(b::_QMBits, v::UInt64, q::Int, j::Int) = _flipbit!(b.z, v, j, q)

# Combined row `r` in `1:2n`, X rows first.
@inline function _rget(b::BitBlock, r::Int, j::Int)
    n = b.n
    return r <= n ? _xget(b, r, j) : _zget(b, r - n, j)
end
@inline function _rset!(b::BitBlock, v::Int, r::Int, j::Int)
    n = b.n
    r <= n ? _xset!(b, v, r, j) : _zset!(b, v, r - n, j)
    return v
end

## generator-major

function _zero_gen!(b::_GMBits, j::Int)
    x, z = b.x, b.z
    @inbounds @simd for w in axes(x, 1)
        x[w, j] = UInt64(0)
        z[w, j] = UInt64(0)
    end
    return b
end

function _swap_gens!(b::_GMBits, i::Int, j::Int)
    i == j && return b
    for A in (b.x, b.z)
        @inbounds @simd for w in axes(A, 1)
            t = A[w, i]
            A[w, i] = A[w, j]
            A[w, j] = t
        end
    end
    return b
end

function _move_gen!(b::_GMBits, j::Int, k::Int)
    j == k && return b
    x, z = b.x, b.z
    @inbounds @simd for w in axes(x, 1)
        x[w, j] = x[w, k]
        z[w, j] = z[w, k]
    end
    return b
end

function _copy_gen!(v::AbstractVector{<:_StorageEltype}, b::_GMBits, j::Int)
    n = b.n
    @inbounds for w in axes(b.x, 1)
        _unpack_word!(v, 0, w, n, b.x[w, j], false)
        _unpack_word!(v, n, w, n, b.z[w, j], false)
    end
    return v
end

function _load_gen!(b::_GMBits, j::Int, v::AbstractVector{<:_StorageEltype})
    n = b.n
    @inbounds for w in axes(b.x, 1)
        b.x[w, j] = _pack_word(v, 0, w, n)
        b.z[w, j] = _pack_word(v, n, w, n)
    end
    return b
end

# generator j += v (mod 2)
function _gen_xor!(b::_GMBits, j::Int, v::AbstractVector{<:_StorageEltype})
    n = b.n
    @inbounds for w in axes(b.x, 1)
        b.x[w, j] ⊻= _pack_word(v, 0, w, n)
        b.z[w, j] ⊻= _pack_word(v, n, w, n)
    end
    return b
end

# generator j of `b` += generator k of `src` (mod 2)
function _gen_xor!(b::_GMBits, j::Int, src::_GMBits, k::Int)
    x, z, sx, sz = b.x, b.z, src.x, src.z
    @inbounds @simd for w in axes(x, 1)
        x[w, j] ⊻= sx[w, k]
        z[w, j] ⊻= sz[w, k]
    end
    return b
end

# v += generator j (mod 2)
function _vec_xor!(v::AbstractVector{<:_StorageEltype}, b::_GMBits, j::Int)
    n = b.n
    @inbounds for w in axes(b.x, 1)
        _unpack_word!(v, 0, w, n, b.x[w, j], true)
        _unpack_word!(v, n, w, n, b.z[w, j], true)
    end
    return v
end

## qudit-major

function _zero_gen!(b::_QMBits, j::Int)
    w, mask = _word_index(j), ~(UInt64(1) << _bit_offset(j))
    x, z = b.x, b.z
    @inbounds for q in 1:b.n
        x[w, q] &= mask
        z[w, q] &= mask
    end
    return b
end

function _swap_gens!(b::_QMBits, i::Int, j::Int)
    i == j && return b
    wi, oi = _word_index(i), _bit_offset(i)
    wj, oj = _word_index(j), _bit_offset(j)
    for A in (b.x, b.z)
        @inbounds for q in 1:b.n
            t = ((A[wi, q] >> oi) ⊻ (A[wj, q] >> oj)) & UInt64(1)
            A[wi, q] ⊻= t << oi
            A[wj, q] ⊻= t << oj
        end
    end
    return b
end

function _move_gen!(b::_QMBits, j::Int, k::Int)
    j == k && return b
    wj, oj = _word_index(j), _bit_offset(j)
    wk, ok = _word_index(k), _bit_offset(k)
    mask = ~(UInt64(1) << oj)
    for A in (b.x, b.z)
        @inbounds for q in 1:b.n
            A[wj, q] = (A[wj, q] & mask) | (((A[wk, q] >> ok) & UInt64(1)) << oj)
        end
    end
    return b
end

function _copy_gen!(v::AbstractVector{E}, b::_QMBits, j::Int) where {E<:_StorageEltype}
    n = b.n
    w, o = _word_index(j), _bit_offset(j)
    @inbounds for q in 1:n
        v[q] = ((b.x[w, q] >> o) & UInt64(1)) % E
        v[n + q] = ((b.z[w, q] >> o) & UInt64(1)) % E
    end
    return v
end

function _load_gen!(b::_QMBits, j::Int, v::AbstractVector{<:_StorageEltype})
    n = b.n
    w, o = _word_index(j), _bit_offset(j)
    mask = ~(UInt64(1) << o)
    @inbounds for q in 1:n
        b.x[w, q] = (b.x[w, q] & mask) | (_entry_bit(v, q) << o)
        b.z[w, q] = (b.z[w, q] & mask) | (_entry_bit(v, n + q) << o)
    end
    return b
end

function _gen_xor!(b::_QMBits, j::Int, v::AbstractVector{<:_StorageEltype})
    n = b.n
    w, o = _word_index(j), _bit_offset(j)
    @inbounds for q in 1:n
        b.x[w, q] ⊻= _entry_bit(v, q) << o
        b.z[w, q] ⊻= _entry_bit(v, n + q) << o
    end
    return b
end

function _gen_xor!(b::_QMBits, j::Int, src::_QMBits, k::Int)
    wj, oj = _word_index(j), _bit_offset(j)
    wk, ok = _word_index(k), _bit_offset(k)
    x, z, sx, sz = b.x, b.z, src.x, src.z
    @inbounds for q in 1:b.n
        x[wj, q] ⊻= ((sx[wk, q] >> ok) & UInt64(1)) << oj
        z[wj, q] ⊻= ((sz[wk, q] >> ok) & UInt64(1)) << oj
    end
    return b
end

function _vec_xor!(v::AbstractVector{E}, b::_QMBits, j::Int) where {E<:_StorageEltype}
    n = b.n
    w, o = _word_index(j), _bit_offset(j)
    @inbounds for q in 1:n
        v[q] = (_entry_bit(v, q) ⊻ ((b.x[w, q] >> o) & UInt64(1))) % E
        v[n + q] = (_entry_bit(v, n + q) ⊻ ((b.z[w, q] >> o) & UInt64(1))) % E
    end
    return v
end

## both layouts, through the forms above

_gen_addmul!(b::BitBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, ::Int) =
    isodd(a) ? _gen_xor!(b, j, v) : b
_gen_submul!(b::BitBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, ::Int) =
    isodd(a) ? _gen_xor!(b, j, v) : b
_gen_mulcopy!(b::BitBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, ::Int) =
    isodd(a) ? _load_gen!(b, j, v) : _zero_gen!(b, j)

# `src` may be `b` itself when `k ≠ j`.
_gen_addmul!(b::BitBlock{L}, j::Int, src::BitBlock{L}, k::Int, a::Int, ::Int) where {L} =
    isodd(a) ? _gen_xor!(b, j, src, k) : b
_gen_submul!(b::BitBlock{L}, j::Int, src::BitBlock{L}, k::Int, a::Int, ::Int) where {L} =
    isodd(a) ? _gen_xor!(b, j, src, k) : b

_gen_scale!(b::BitBlock, j::Int, a::Int, ::Int) = isodd(a) ? b : _zero_gen!(b, j)

_vec_addmul!(v::AbstractVector{<:_StorageEltype}, b::BitBlock, j::Int, a::Int, ::Int) =
    isodd(a) ? _vec_xor!(v, b, j) : v
_vec_submul!(v::AbstractVector{<:_StorageEltype}, b::BitBlock, j::Int, a::Int, ::Int) =
    isodd(a) ? _vec_xor!(v, b, j) : v

# Site-restricted forms: only qudits `sites[1:nsites]` of generator j change.
function _gen_addmul_sites!(b::BitBlock, j::Int, v::AbstractVector{<:_StorageEltype},
                            sites::AbstractVector{Int}, nsites::Int, a::Int, ::Int)
    isodd(a) || return b
    n = b.n
    @inbounds for k in 1:nsites
        q = sites[k]
        _xflip!(b, _entry_bit(v, q), q, j)
        _zflip!(b, _entry_bit(v, n + q), q, j)
    end
    return b
end
_gen_submul_sites!(b::BitBlock, j::Int, v::AbstractVector{<:_StorageEltype},
                   sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int) =
    _gen_addmul_sites!(b, j, v, sites, nsites, a, d)

# `src` may be `b` itself when `k ≠ j`.
function _gen_addmul_sites!(b::BitBlock{L}, j::Int, src::BitBlock{L}, k::Int,
                            sites::AbstractVector{Int}, nsites::Int, a::Int, ::Int) where {L}
    isodd(a) || return b
    @inbounds for s in 1:nsites
        q = sites[s]
        _xflip!(b, _xget(src, q, k) % UInt64, q, j)
        _zflip!(b, _zget(src, q, k) % UInt64, q, j)
    end
    return b
end
_gen_submul_sites!(b::BitBlock{L}, j::Int, src::BitBlock{L}, k::Int,
                   sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int) where {L} =
    _gen_addmul_sites!(b, j, src, k, sites, nsites, a, d)

##### equality #####

# Whether entries `1:m` of two plane vectors with the same number of planes
# agree, word by word with the last word masked to `m`.
function _first_entries_equal(pa::_BitPlanes, pb::_BitPlanes, m::Int)
    A, B = pa.planes, pb.planes
    nw = _nwords(m)
    nw == 0 && return true
    t = _tail_mask(m)
    @inbounds for k in axes(A, 2)
        for w in 1:(nw - 1)
            A[w, k] == B[w, k] || return false
        end
        (A[nw, k] ⊻ B[nw, k]) & t == UInt64(0) || return false
    end
    return true
end

# `_active_columns_equal` for two packed blocks of the same layout, by words:
# generator-major the word columns `1:m` (bits past `n` are zero in both),
# qudit-major each site's words over generators `1:m`, the last masked to `m`.
function _active_columns_equal(a::_GMBits, pa::_BitPlanes,
                               b::_GMBits, pb::_BitPlanes,
                               m::Int, storephase::Bool)
    storephase && !_first_entries_equal(pa, pb, m) && return false
    ax, az, bx, bz = a.x, a.z, b.x, b.z
    nw = size(ax, 1)
    @inbounds for j in 1:m, w in 1:nw
        (ax[w, j] == bx[w, j] && az[w, j] == bz[w, j]) || return false
    end
    return true
end
function _active_columns_equal(a::_QMBits, pa::_BitPlanes,
                               b::_QMBits, pb::_BitPlanes,
                               m::Int, storephase::Bool)
    storephase && !_first_entries_equal(pa, pb, m) && return false
    nw = _nwords(m)
    nw == 0 && return true
    t = _tail_mask(m)
    ax, az, bx, bz = a.x, a.z, b.x, b.z
    @inbounds for q in 1:a.n
        for w in 1:(nw - 1)
            (ax[w, q] == bx[w, q] && az[w, q] == bz[w, q]) || return false
        end
        ((ax[nw, q] ⊻ bx[nw, q]) | (az[nw, q] ⊻ bz[nw, q])) & t == UInt64(0) || return false
    end
    return true
end

##### switching layouts #####

# The masks of the six rounds of `_t64!`, for exchanges at distances 32, 16, 8,
# 4, 2 and 1.
const _T64_MASKS = (0x00000000ffffffff, 0x0000ffff0000ffff, 0x00ff00ff00ff00ff,
                    0x0f0f0f0f0f0f0f0f, 0x3333333333333333, 0x5555555555555555)

# The exchanges of `_t64!`, written out: round `r` exchanges word `i` with word
# `i + j` for every `i` without bit `j`, `j = 32 >> (r - 1)`. The unrolled body
# transposes a block faster than the same exchanges in loops.
function _t64_body()
    ex = Expr[]
    j = _WORD_BITS >> 1
    for m in _T64_MASKS
        for i in 0:(_WORD_BITS - 1)
            i & j == 0 || continue
            push!(ex, quote
                t = ((a[o + $(i + 1)] >> $j) ⊻ a[o + $(i + j + 1)]) & $m
                a[o + $(i + 1)] ⊻= t << $j
                a[o + $(i + j + 1)] ⊻= t
            end)
        end
        j >>= 1
    end
    return Expr(:block, ex...)
end

# Transpose, in place, the 64 × 64 bit matrix held in `a[o+1:o+64]` (word `i`
# is row `i-1`, bits least significant first): bit `b` of word `i` swaps with
# bit `i` of word `b`. It is not marked `@inline`: inlined into the strip loop
# of `_transpose_bit_block!` it is about 4% slower.
@eval function _t64!(a::AbstractArray{UInt64}, o::Int)
    @inbounds $(_t64_body())
    return a
end

# Transpose the `_WORD_BITS·nw` square bit matrix of the block `A` in place,
# `nw` the number of its rows. Each 64 × 64 strip is moved through the first
# `2·_WORD_BITS` words of `buf`, a vector or a matrix indexed linearly (a
# matrix is passed as it is, because a view of it would be allocated on the
# heap at the call): a diagonal strip is transposed where it is,
# and the strips (word row `W`, block `G`) and (word row `G`, block `W`) with
# `W < G` are transposed and stored in each other's place. Nothing outside the
# view is read or written.
function _transpose_bit_block!(A::_BlockMatrix{UInt64}, buf::AbstractArray{UInt64})
    nw = size(A, 1)
    P = parent(A)
    off = first(parentindices(A)[2]) - 1
    @inbounds for W in 1:nw
        c0 = _WORD_BITS * (W - 1) + off
        for k in 1:_WORD_BITS
            buf[k] = P[W, c0 + k]
        end
        _t64!(buf, 0)
        for k in 1:_WORD_BITS
            P[W, c0 + k] = buf[k]
        end
        for G in (W + 1):nw
            cA = _WORD_BITS * (G - 1) + off
            for k in 1:_WORD_BITS
                buf[k] = P[W, cA + k]
                buf[_WORD_BITS + k] = P[G, c0 + k]
            end
            _t64!(buf, 0)
            _t64!(buf, _WORD_BITS)
            for k in 1:_WORD_BITS
                P[W, cA + k] = buf[_WORD_BITS + k]
                P[G, c0 + k] = buf[k]
            end
        end
    end
    return A
end

##### unpacked copies #####

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
