##### generator-level access to a tableau block #####
#
# An `XZBlock` holds the X and Z exponents of a set of generators as two
# matrices. In the `GeneratorMajor` layout generator `j` is column `j` of both:
# `x[q, j]` and `z[q, j]` are its exponents on qudit `q`. Every operation below
# works on whole generators or single entries, and calls the `src/modular.jl`
# primitives on the contiguous per-generator vectors, so the arithmetic tiers and
# their preconditions (reduced entries, no aliasing) are those of the primitives.
#
# Methods are written for `XZBlock{GeneratorMajor}` only; another layout gets a
# `MethodError`. This file touches only `b.x`, `b.z` and plain arrays, never a
# tableau.
#
# The matrices hold one of the storage element types `E` (`_StorageEltype`).
# Single entries go in and out as `Int` (`_xget`, `_xset!` and their kin), so
# callers do their arithmetic in `Int` whatever `E` is; the generator views
# `_xgen` and `_zgen` expose the raw `E` storage.

abstract type TableauLayout end
struct GeneratorMajor <: TableauLayout end

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
# makes random two-qudit gates at d = 3 up to 1.2x slower, because an
# unpadded stride keeps the in-page offsets that the hardware prefetchers
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

@inline _nq(b::_GMBlock) = size(b.x, 1)
@inline _ncap(b::_GMBlock) = size(b.x, 2)

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

# Combined row `r` in `1:2n`, X rows first.
@inline function _rget(b::_GMBlock, r::Int, j::Int)
    n = _nq(b)
    return r <= n ? _xget(b, r, j) : _zget(b, r - n, j)
end
@inline function _rset!(b::_GMBlock, v::Int, r::Int, j::Int)
    n = _nq(b)
    r <= n ? _xset!(b, v, r, j) : _zset!(b, v, r - n, j)
    return v
end

@inline _xgen(b::_GMBlock, j::Int) = view(b.x, :, j)
@inline _zgen(b::_GMBlock, j::Int) = view(b.z, :, j)

# Stacked vectors `[x; z]` of length at least `2n`, of any storage element
# type. Operations between a vector and a block of different element types go
# to the mixed-type methods of the `src/modular.jl` primitives.
@inline _vx(v::AbstractVector{<:_StorageEltype}, n::Int) = view(v, 1:n)
@inline _vz(v::AbstractVector{<:_StorageEltype}, n::Int) = view(v, n+1:2n)

function _copy_gen!(v::AbstractVector{<:_StorageEltype}, b::_GMBlock, j::Int)
    n = _nq(b)
    copyto!(_vx(v, n), _xgen(b, j))
    copyto!(_vz(v, n), _zgen(b, j))
    return v
end

function _load_gen!(b::_GMBlock, j::Int, v::AbstractVector{<:_StorageEltype})
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

# Copy generator `k` into generator `j` (`j ≠ k`: distinct columns).
function _move_gen!(b::_GMBlock, j::Int, k::Int)
    j == k && return b
    copyto!(_xgen(b, j), _xgen(b, k))
    copyto!(_zgen(b, j), _zgen(b, k))
    return b
end

# generator j  (+=, -=, =)  a * v, v stacked. `v` is a separate vector from the
# block, so it cannot alias generator j.
function _gen_addmul!(b::_GMBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    n = _nq(b)
    addmul_mod!(_xgen(b, j), _vx(v, n), a, d)
    addmul_mod!(_zgen(b, j), _vz(v, n), a, d)
    return b
end
function _gen_submul!(b::_GMBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    n = _nq(b)
    submul_mod!(_xgen(b, j), _vx(v, n), a, d)
    submul_mod!(_zgen(b, j), _vz(v, n), a, d)
    return b
end
function _gen_mulcopy!(b::_GMBlock, j::Int, v::AbstractVector{<:_StorageEltype}, a::Int, d::Int)
    n = _nq(b)
    mulcopy_mod!(_xgen(b, j), _vx(v, n), a, d)
    mulcopy_mod!(_zgen(b, j), _vz(v, n), a, d)
    return b
end

# generator j of `b`  (+=, -=)  a * generator k of `src`. `src` may be `b`
# itself when `k ≠ j`: then dst and src are distinct columns of one matrix.
function _gen_addmul!(b::_GMBlock, j::Int, src::_GMBlock, k::Int, a::Int, d::Int)
    addmul_mod!(_xgen(b, j), _xgen(src, k), a, d)
    addmul_mod!(_zgen(b, j), _zgen(src, k), a, d)
    return b
end
function _gen_submul!(b::_GMBlock, j::Int, src::_GMBlock, k::Int, a::Int, d::Int)
    submul_mod!(_xgen(b, j), _xgen(src, k), a, d)
    submul_mod!(_zgen(b, j), _zgen(src, k), a, d)
    return b
end

function _gen_scale!(b::_GMBlock, j::Int, a::Int, d::Int)
    scale_mod!(_xgen(b, j), a, d)
    scale_mod!(_zgen(b, j), a, d)
    return b
end

# stacked v[1:2n]  (+=, -=)  a * generator j. `v` is a separate vector from the block.
function _vec_addmul!(v::AbstractVector{<:_StorageEltype}, b::_GMBlock, j::Int, a::Int, d::Int)
    n = _nq(b)
    addmul_mod!(_vx(v, n), _xgen(b, j), a, d)
    addmul_mod!(_vz(v, n), _zgen(b, j), a, d)
    return v
end
function _vec_submul!(v::AbstractVector{<:_StorageEltype}, b::_GMBlock, j::Int, a::Int, d::Int)
    n = _nq(b)
    submul_mod!(_vx(v, n), _xgen(b, j), a, d)
    submul_mod!(_vz(v, n), _zgen(b, j), a, d)
    return v
end

# Site-restricted forms: only qudits `sites[1:nsites]` of generator j change.
# The stacked `v` is a separate vector from the block, so it cannot alias.
function _gen_addmul_sites!(b::_GMBlock, j::Int, v::AbstractVector{<:_StorageEltype},
                            sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int)
    n = _nq(b)
    addmul_mod_sites!(_xgen(b, j), _zgen(b, j), _vx(v, n), _vz(v, n), sites, nsites, a, d)
    return b
end
function _gen_submul_sites!(b::_GMBlock, j::Int, v::AbstractVector{<:_StorageEltype},
                            sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int)
    n = _nq(b)
    submul_mod_sites!(_xgen(b, j), _zgen(b, j), _vx(v, n), _vz(v, n), sites, nsites, a, d)
    return b
end
# `src` may be `b` itself when `k ≠ j`: distinct columns of one matrix.
function _gen_addmul_sites!(b::_GMBlock, j::Int, src::_GMBlock, k::Int,
                            sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int)
    addmul_mod_sites!(_xgen(b, j), _zgen(b, j), _xgen(src, k), _zgen(src, k),
                      sites, nsites, a, d)
    return b
end
function _gen_submul_sites!(b::_GMBlock, j::Int, src::_GMBlock, k::Int,
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
