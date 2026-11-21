##### generator-level access to a tableau block #####
#
# An `XZBlock` holds the X and Z exponents of a set of generators as two
# matrices. In the `GeneratorMajor` layout generator `j` is column `j` of both:
# `x[q, j]` and `z[q, j]` are its exponents on qudit `q`. In the `QuditMajor`
# layout generator `j` is row `j`: its exponents on qudit `q` are `x[j, q]` and
# `z[j, q]`. Every operation below works on whole generators or single entries,
# and calls the `src/modular.jl` primitives on the per-generator vectors
# (contiguous columns, or strided rows), so the arithmetic tiers and their
# preconditions (reduced entries, no aliasing) are those of the primitives.
#
# Two kinds of methods. The leaves (`_nq`, `_ncap`, `_xget`, `_zget`, `_xset!`,
# `_zset!`, `_xgen`, `_zgen`, `_zero_gen!`, `_swap_gens!` and the `dot_xz_*`
# products) know the layout: each has one method for `XZBlock{GeneratorMajor}`
# and one for `XZBlock{QuditMajor}`, and none for a bare `XZBlock`. Every other
# operation is a single method on `XZBlock` that calls only leaves, so it works
# in both layouts. This file touches only `b.x`, `b.z` and plain arrays, never
# a tableau.
#
# The matrices hold one of the storage element types `E` (`_StorageEltype`).
# Single entries go in and out as `Int` (`_xget`, `_xset!` and their kin), so
# callers do their arithmetic in `Int` whatever `E` is; the generator views
# `_xgen` and `_zgen` expose the raw `E` storage.

abstract type TableauLayout end
struct GeneratorMajor <: TableauLayout end
struct QuditMajor <: TableauLayout end

struct XZBlock{L<:TableauLayout, M<:AbstractMatrix{<:_StorageEltype}}
    x::M
    z::M
end
XZBlock{L}(x::M, z::M) where {L, M} = XZBlock{L, M}(x, z)

##### padded generator storage #####
#
# Each generator block (`stab_x`, `stab_z`, `destab_x`, `destab_z`) is an
# `n × n` view into a parent matrix of its own. Sweeps that read the same qudit
# of every generator (`apply!`'s gather and scatter, the row scans in
# `measure!` and `expect!`) step through memory by the column stride, and two
# placements, both stated in bytes so that they hold for any element type,
# decide whether those accesses spread over the cache sets:
#
# - the parent has `_padded_rows(n, sizeof(T))` rows, which makes the column
#   stride an odd multiple of `_LINE_BYTES` (128). Consecutive columns then land
#   on distinct cache lines modulo every power-of-two set period. An unpadded
#   stride is a multiple of 4 KiB at n = 512, 1024, … (8-byte entries) and puts
#   every column of a sweep into a handful of sets; strides within a few
#   elements of such a multiple (n = 511, 513, 1023, …) are nearly as bad.
# - the view starts `c` columns into its parent, `0 ≤ c < 32`, chosen so that
#   its first entry sits in the 128-byte line `slot ÷ 128` modulo 4096 bytes.
#   The four blocks use different slots, so `x[q, j]` and `z[q, j]` of one
#   generator (and the destabilizer entries) never share a set. Large arrays
#   are page-aligned, so unshifted blocks would put them in the same set for
#   every `j`.
#
# Columns shorter than `_PAD_MIN_COL_BYTES` are neither padded nor shifted:
# n < 256 for `UInt8` entries, n < 32 for 8-byte `Int`. From 256 bytes on, the
# padding is a trade-off. With `UInt8` entries it makes the one-qudit gate
# sweeps up to 1.5x faster near a power of two (n = 256 and 511-513), and it
# makes random two-qudit gates at d = 3 up to 1.2x slower, most likely because
# an unpadded stride keeps the in-page offsets that the hardware prefetchers
# follow when a sweep reads two qudits per generator. Of the cutoffs 256, 512
# and 1024 bytes, 256 has the smallest worst-case slowdown against the best of
# the three at each n (monitored circuits differ by at most 1.13x).

const _BlockMatrix{T} = SubArray{T, 2, Matrix{T}, Tuple{Base.OneTo{Int}, UnitRange{Int}}, false}

const _LINE_BYTES = 128
const _SET_SPAN_BYTES = 4096
const _PAD_MIN_COL_BYTES = 256

# Rows of a parent for `n`-entry columns of `sz`-byte elements: the smallest
# count `≥ n` whose stride `rows · sz` is an odd multiple of `_LINE_BYTES`, or
# `n` for short columns. `sz` must divide `_LINE_BYTES`.
function _padded_rows(n::Int, sz::Int)
    (sz > 0 && _LINE_BYTES % sz == 0) || throw(ArgumentError(
        "element size $sz must divide the $_LINE_BYTES-byte line unit"))
    n * sz < _PAD_MIN_COL_BYTES && return n
    u = _LINE_BYTES ÷ sz
    return n + mod(u - n, 2u)
end

# A zeroed `n × n` block of element type `T` in a parent of its own, starting in
# line `slot ÷ _LINE_BYTES` modulo `_SET_SPAN_BYTES` (see above).
function _padded_block(::Type{T}, n::Int, slot::Int) where {T}
    sz = sizeof(T)
    L = _padded_rows(n, sz)
    L == n && return view(zeros(T, n, n), Base.OneTo(n), 1:n)
    nshift = _SET_SPAN_BYTES ÷ _LINE_BYTES    # 32: the stride steps through every line
    P = zeros(T, L, n + nshift - 1)
    base = UInt(pointer(P))
    want = mod(slot, _SET_SPAN_BYTES) ÷ _LINE_BYTES
    c = 0
    while (base + c * L * sz) % _SET_SPAN_BYTES ÷ _LINE_BYTES != want
        c += 1
    end
    return view(P, Base.OneTo(n), c+1:c+n)
end

# The line slot a block's first entry sits in, as `_padded_block` takes it.
_block_slot(A::_BlockMatrix) =
    Int(UInt(pointer(A)) % _SET_SPAN_BYTES ÷ _LINE_BYTES * _LINE_BYTES)

# The stabilizer X and Z blocks, and the destabilizer blocks of a
# `DestabilizerTableau`, in the line slots between them.
_stab_blocks(::Type{E}, n::Int) where {E} = (_padded_block(E, n, 0), _padded_block(E, n, 2048))
_destab_blocks(::Type{E}, n::Int) where {E} = (_padded_block(E, n, 1024), _padded_block(E, n, 3072))

# Copy the entries of block `src` into block `dst` of the same size, one
# contiguous column at a time.
function _copy_block!(dst::_BlockMatrix{T}, src::_BlockMatrix{T}) where {T}
    @inbounds for j in axes(src, 2)
        copyto!(view(dst, :, j), view(src, :, j))
    end
    return dst
end

# A copy of a block in a fresh parent, in the same line slot.
_copy_block(A::_BlockMatrix{T}) where {T} =
    _copy_block!(_padded_block(T, size(A, 1), _block_slot(A)), A)

# Zero an array of a storage element type in any layout; the `Array{Int}`
# method in helper.jl is one `memset`.
_zero!(A::AbstractArray{<:_StorageEltype}) = fill!(A, zero(eltype(A)))

# One `memset` per column of a block.
function _zero!(A::_BlockMatrix{T}) where {T<:_StorageEltype}
    P = parent(A)
    L = size(P, 1)
    nb = size(A, 1) * sizeof(T)
    GC.@preserve P for c in parentindices(A)[2]
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t),
              pointer(P, (c - 1) * L + 1), 0, nb)
    end
    return A
end
function _zero!(b::XZBlock)
    _zero!(b.x)
    _zero!(b.z)
    return b
end

# Views into a stacked `(2n + storephase) × n'` matrix: X rows `1:n`, Z rows
# `n+1:2n`, and the phase row `2n+1` when `storephase`. Without a phase row the
# phase is an empty view of the same type.
function _stacked_views(S::Matrix{Int}, n::Int, storephase::Bool)
    nc = size(S, 2)
    b = XZBlock{GeneratorMajor}(view(S, 1:n, :), view(S, n+1:2n, :))
    # The empty view must also exist for a matrix with no rows, so it is built
    # without the bounds check of `view` (which `@inbounds` does not reach on
    # Julia 1.10). Both branches have the same type.
    phase = storephase ? view(S, 2n+1, 1:nc) : SubArray(S, (1, 1:0))
    return b, phase
end

const _GMBlock = XZBlock{GeneratorMajor}
const _QMBlock = XZBlock{QuditMajor}

# A generator-major copy of `b` in fresh plain matrices, for code that works on
# a private generator-major scratch whatever the layout of its source.
_generator_major_copy(b::XZBlock{GeneratorMajor, <:_BlockMatrix{E}}) where {E} =
    XZBlock{GeneratorMajor}(copy(b.x), copy(b.z))
function _generator_major_copy(b::XZBlock{QuditMajor, <:_BlockMatrix{E}}) where {E}
    x = _transpose_into!(Matrix{E}(undef, reverse(size(b.x))), 0, b.x)
    z = _transpose_into!(Matrix{E}(undef, reverse(size(b.z))), 0, b.z)
    return XZBlock{GeneratorMajor}(x, z)
end

@inline _nq(b::_GMBlock) = size(b.x, 1)
@inline _ncap(b::_GMBlock) = size(b.x, 2)
@inline _nq(b::_QMBlock) = size(b.x, 2)
@inline _ncap(b::_QMBlock) = size(b.x, 1)

# Entries are read as `Int` and written from an `Int` `v` with `0 ≤ v < d`,
# which every storage element type of the block holds (see `_eltype_admissible`),
# so the conversion `v % E` is exact.
@inline _xget(b::_GMBlock, q::Int, j::Int) = @inbounds b.x[q, j] % Int
@inline _zget(b::_GMBlock, q::Int, j::Int) = @inbounds b.z[q, j] % Int
@inline function _xset!(b::_GMBlock, v::Int, q::Int, j::Int)
    @inbounds b.x[q, j] = v % eltype(b.x)
    return v
end
@inline function _zset!(b::_GMBlock, v::Int, q::Int, j::Int)
    @inbounds b.z[q, j] = v % eltype(b.z)
    return v
end
@inline _xget(b::_QMBlock, q::Int, j::Int) = @inbounds b.x[j, q] % Int
@inline _zget(b::_QMBlock, q::Int, j::Int) = @inbounds b.z[j, q] % Int
@inline function _xset!(b::_QMBlock, v::Int, q::Int, j::Int)
    @inbounds b.x[j, q] = v % eltype(b.x)
    return v
end
@inline function _zset!(b::_QMBlock, v::Int, q::Int, j::Int)
    @inbounds b.z[j, q] = v % eltype(b.z)
    return v
end

# Whether generators `1:m` of two qudit-major blocks agree entry for entry,
# walked in storage order: each site's column over the generators.
function _qm_generators_equal(a::_QMBlock, b::_QMBlock, m::Int)
    @inbounds for q in 1:_nq(a)
        for j in 1:m
            a.x[j, q] % Int == b.x[j, q] % Int || return false
        end
        for j in 1:m
            a.z[j, q] % Int == b.z[j, q] % Int || return false
        end
    end
    return true
end

# Combined row `r` in `1:2n`, X rows first.
@inline function _rget(b::XZBlock, r::Int, j::Int)
    n = _nq(b)
    return r <= n ? _xget(b, r, j) : _zget(b, r - n, j)
end
@inline function _rset!(b::XZBlock, v::Int, r::Int, j::Int)
    n = _nq(b)
    r <= n ? _xset!(b, v, r, j) : _zset!(b, v, r - n, j)
    return v
end

@inline _xgen(b::_GMBlock, j::Int) = view(b.x, :, j)
@inline _zgen(b::_GMBlock, j::Int) = view(b.z, :, j)
@inline _xgen(b::_QMBlock, j::Int) = view(b.x, j, :)
@inline _zgen(b::_QMBlock, j::Int) = view(b.z, j, :)

# Stacked vectors `[x; z]` of length at least `2n`, of any storage element
# type. Operations between a vector and a block of different element types go
# to the mixed-type methods of the `src/modular.jl` primitives.
@inline _vx(v::AbstractVector{<:_StorageEltype}, n::Int) = view(v, 1:n)
@inline _vz(v::AbstractVector{<:_StorageEltype}, n::Int) = view(v, n+1:2n)

function _copy_gen!(v::AbstractVector{<:_StorageEltype}, b::XZBlock, j::Int)
    n = _nq(b)
    copyto!(_vx(v, n), _xgen(b, j))
    copyto!(_vz(v, n), _zgen(b, j))
    return v
end

function _load_gen!(b::XZBlock, j::Int, v::AbstractVector{<:_StorageEltype})
    n = _nq(b)
    copyto!(_xgen(b, j), _vx(v, n))
    copyto!(_zgen(b, j), _vz(v, n))
    return b
end

function _zero_gen!(b::_GMBlock, j::Int)
    fill!(_xgen(b, j), 0)
    fill!(_zgen(b, j), 0)
    return b
end

function _zero_gen!(b::_QMBlock, j::Int)
    fill!(_xgen(b, j), 0)
    fill!(_zgen(b, j), 0)
    return b
end

function _swap_gens!(b::_GMBlock, i::Int, j::Int)
    i == j && return b
    for blk in (b.x, b.z)
        @inbounds @simd for q in axes(blk, 1)
            t = blk[q, i]
            blk[q, i] = blk[q, j]
            blk[q, j] = t
        end
    end
    return b
end

function _swap_gens!(b::_QMBlock, i::Int, j::Int)
    i == j && return b
    for blk in (b.x, b.z)
        @inbounds for q in axes(blk, 2)
            t = blk[i, q]
            blk[i, q] = blk[j, q]
            blk[j, q] = t
        end
    end
    return b
end

# Copy generator `k` into generator `j` (`j ≠ k`: distinct generators).
function _move_gen!(b::XZBlock, j::Int, k::Int)
    j == k && return b
    copyto!(_xgen(b, j), _xgen(b, k))
    copyto!(_zgen(b, j), _zgen(b, k))
    return b
end

# generator j  (+=, -=, =)  a * v, v stacked. `v` is a separate vector from the
# block, so it cannot alias generator j.
function _gen_addmul!(b::XZBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    n = _nq(b)
    addmul_mod!(_xgen(b, j), _vx(v, n), a, d)
    addmul_mod!(_zgen(b, j), _vz(v, n), a, d)
    return b
end
function _gen_submul!(b::XZBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    n = _nq(b)
    submul_mod!(_xgen(b, j), _vx(v, n), a, d)
    submul_mod!(_zgen(b, j), _vz(v, n), a, d)
    return b
end
function _gen_mulcopy!(b::XZBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    n = _nq(b)
    mulcopy_mod!(_xgen(b, j), _vx(v, n), a, d)
    mulcopy_mod!(_zgen(b, j), _vz(v, n), a, d)
    return b
end

# generator j of `b`  (+=, -=)  a * generator k of `src`. `src` may be `b`
# itself when `k ≠ j`: then dst and src are distinct generators of one matrix.
function _gen_addmul!(b::XZBlock, j::Int, src::XZBlock, k::Int, a::Int, d::Int)
    addmul_mod!(_xgen(b, j), _xgen(src, k), a, d)
    addmul_mod!(_zgen(b, j), _zgen(src, k), a, d)
    return b
end
function _gen_submul!(b::XZBlock, j::Int, src::XZBlock, k::Int, a::Int, d::Int)
    submul_mod!(_xgen(b, j), _xgen(src, k), a, d)
    submul_mod!(_zgen(b, j), _zgen(src, k), a, d)
    return b
end

function _gen_scale!(b::XZBlock, j::Int, a::Int, d::Int)
    scale_mod!(_xgen(b, j), a, d)
    scale_mod!(_zgen(b, j), a, d)
    return b
end

# stacked v[1:2n]  (+=, -=)  a * generator j. `v` is a separate vector from the block.
function _vec_addmul!(v::AbstractVector{<:_StorageEltype}, b::XZBlock, j::Int, a::Int, d::Int)
    n = _nq(b)
    addmul_mod!(_vx(v, n), _xgen(b, j), a, d)
    addmul_mod!(_vz(v, n), _zgen(b, j), a, d)
    return v
end
function _vec_submul!(v::AbstractVector{<:_StorageEltype}, b::XZBlock, j::Int, a::Int, d::Int)
    n = _nq(b)
    submul_mod!(_vx(v, n), _xgen(b, j), a, d)
    submul_mod!(_vz(v, n), _zgen(b, j), a, d)
    return v
end

# Site-restricted forms: only qudits `sites[1:nsites]` of generator j change.
# The stacked `v` is a separate vector from the block, so it cannot alias.
function _gen_addmul_sites!(b::XZBlock, j::Int, v::AbstractVector{<:_StorageEltype},
                            sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int)
    n = _nq(b)
    addmul_mod_sites!(_xgen(b, j), _zgen(b, j), _vx(v, n), _vz(v, n), sites, nsites, a, d)
    return b
end
function _gen_submul_sites!(b::XZBlock, j::Int, v::AbstractVector{<:_StorageEltype},
                            sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int)
    n = _nq(b)
    submul_mod_sites!(_xgen(b, j), _zgen(b, j), _vx(v, n), _vz(v, n), sites, nsites, a, d)
    return b
end
# `src` may be `b` itself when `k ≠ j`: distinct generators of one matrix.
function _gen_addmul_sites!(b::XZBlock, j::Int, src::XZBlock, k::Int,
                            sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int)
    addmul_mod_sites!(_xgen(b, j), _zgen(b, j), _xgen(src, k), _zgen(src, k),
                      sites, nsites, a, d)
    return b
end
function _gen_submul_sites!(b::XZBlock, j::Int, src::XZBlock, k::Int,
                            sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int)
    submul_mod_sites!(_xgen(b, j), _zgen(b, j), _xgen(src, k), _zgen(src, k),
                      sites, nsites, a, d)
    return b
end

# The x·z dot products behind phase updates, mod `d`: `dot_xz_col` pairs the X
# part of generator `i` with the Z part of generator `j`, `dot_xz_ws_vs_col` a
# stacked workspace's X part with the Z part of generator `tgt`, and
# `dot_xz_col_vs_zacc` the X part of generator `j` with a Z vector of length `n`.
# Entries are widened to `Int` before they multiply, so narrow storage cannot
# wrap the products.
@inline function dot_xz_col(b::_GMBlock, i::Int, j::Int, d::Int)
    s = 0
    x = b.x
    z = b.z
    @turbo for q in 1:size(x, 1)
        s += Int(x[q, i]) * Int(z[q, j])
    end
    return mod(s, d)
end
@inline dot_xz_col(b::_GMBlock, j::Int, d::Int) = dot_xz_col(b, j, j, d)

@inline function dot_xz_ws_vs_col(genws::AbstractVector{<:_StorageEltype}, b::_GMBlock,
                                  tgt::Int, d::Int)
    s = 0
    z = b.z
    @turbo for q in 1:size(z, 1)
        s += Int(genws[q]) * Int(z[q, tgt])
    end
    return mod(s, d)
end

@inline function dot_xz_col_vs_zacc(b::_GMBlock, j::Int, zacc::AbstractVector{<:_StorageEltype},
                                    d::Int)
    s = 0
    x = b.x
    @turbo for q in 1:size(x, 1)
        s += Int(x[q, j]) * Int(zacc[q])
    end
    return mod(s, d)
end

@inline function dot_xz_col(b::_QMBlock, i::Int, j::Int, d::Int)
    s = 0
    x = b.x
    z = b.z
    @turbo for q in 1:size(x, 2)
        s += Int(x[i, q]) * Int(z[j, q])
    end
    return mod(s, d)
end
@inline dot_xz_col(b::_QMBlock, j::Int, d::Int) = dot_xz_col(b, j, j, d)

@inline function dot_xz_ws_vs_col(genws::AbstractVector{<:_StorageEltype}, b::_QMBlock,
                                  tgt::Int, d::Int)
    s = 0
    z = b.z
    @turbo for q in 1:size(z, 2)
        s += Int(genws[q]) * Int(z[tgt, q])
    end
    return mod(s, d)
end

@inline function dot_xz_col_vs_zacc(b::_QMBlock, j::Int, zacc::AbstractVector{<:_StorageEltype},
                                    d::Int)
    s = 0
    x = b.x
    @turbo for q in 1:size(x, 2)
        s += Int(x[j, q]) * Int(zacc[q])
    end
    return mod(s, d)
end

##### in-place transpose of a block #####
#
# Switching a tableau between the layouts transposes each `n × n` block inside
# its existing parent, so the padding and line slot of the block are kept. The
# square is walked in tiles of `_transpose_tile(T)` entries: each diagonal tile
# is transposed in place and each tile below the diagonal is swapped with its
# mirror above it. Reads run down a contiguous column of the tile and writes
# step by the parent's column stride, so one tile touches at most two tiles'
# worth of columns at a time. Parent rows `n+1:end` and parent columns outside
# the view are never touched.
#
# Tiles of 32 entries for 1- and 2-byte entries and 64 for 8-byte ones had the
# smallest worst case over n = 1000, 1023-1025, 2047-2049, 4095-4097 among
# tiles of 8 to 128, the loop order with the mirror's rows inner, and a
# recursive cache-oblivious order with 16-64-entry leaves (median of three
# runs, M3 Pro, padded blocks). For `UInt8` the recursive order was at most 4%
# faster, within the run-to-run spread; for 8-byte entries 64 beat 32 by 8%
# from n = 2047 on. One `UInt8` block takes 0.18 ms at n = 1024, 0.75 ms at
# n = 2048 and 3.1 ms at n = 4096.

_transpose_tile(::Type{T}) where {T} = sizeof(T) >= 8 ? 64 : 32

function _transpose_block!(A::_BlockMatrix{T}) where {T<:_StorageEltype}
    n = size(A, 1)
    P = parent(A)
    off = first(parentindices(A)[2]) - 1
    tb = _transpose_tile(T)
    @inbounds for jb in 1:tb:n
        je = min(jb + tb - 1, n)
        for j in jb:je
            for i in jb:(j - 1)
                t = P[i, j + off]
                P[i, j + off] = P[j, i + off]
                P[j, i + off] = t
            end
        end
        for ib in (je + 1):tb:n
            ie = min(ib + tb - 1, n)
            for j in jb:je
                for i in ib:ie
                    t = P[i, j + off]
                    P[i, j + off] = P[j, i + off]
                    P[j, i + off] = t
                end
            end
        end
    end
    return A
end

##### out-of-place transpose of a block #####
#
# `dest[r+i, j] = A[j, i]` for every entry of `A`: a qudit-major block written
# generator-major into rows `r+1:r+size(A, 2)` of `dest`. Entries are below
# `d`, so the conversion to `eltype(dest)` is exact. The square is walked in
# the tiles of `_transpose_tile` for the wider of the two element types; within
# a tile, writes run down a contiguous column of `dest` and reads step by the
# column stride of `A`.
#
# On an M3 Pro (`UInt8` into `Int` and into `UInt8`, `Int` into `Int`,
# n = 1024 and 2049), writing contiguously took 0.1-1.1x the time of reading
# contiguously, which is slowest at n = 1024 with its power-of-two write
# stride. Tiles of 32 and 64 entries were within 10% of each other.
function _transpose_into!(dest::Matrix{D}, r::Int, A::AbstractMatrix{T}) where {D, T}
    nj, ni = size(A)
    tb = _transpose_tile(sizeof(D) >= sizeof(T) ? D : T)
    @inbounds for jb in 1:tb:nj
        je = min(jb + tb - 1, nj)
        for ib in 1:tb:ni
            ie = min(ib + tb - 1, ni)
            for j in jb:je
                for i in ib:ie
                    dest[r + i, j] = A[j, i] % D
                end
            end
        end
    end
    return dest
end
