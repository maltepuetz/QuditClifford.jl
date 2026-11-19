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

abstract type TableauLayout end
struct GeneratorMajor <: TableauLayout end

struct XZBlock{L<:TableauLayout, M<:AbstractMatrix{Int}}
    x::M
    z::M
end
XZBlock{L}(x::M, z::M) where {L, M} = XZBlock{L, M}(x, z)

##### padded generator storage #####
#
# A tableau's generator blocks are `n × n` views into one parent matrix with
# `_padded_rows(n)` rows, each block a range of the parent's columns. Sweeps that
# read the same qudit of every generator (`apply!`'s gather and scatter, the row
# scans in `measure!` and `expect!`) step through memory by the column stride,
# and two placements decide whether those accesses spread over the cache sets:
#
# - the column stride is an odd multiple of 128 bytes, so consecutive columns
#   land on distinct cache lines modulo every power-of-two set period. A stride
#   of `n · 8` bytes is a multiple of 4 KiB at n = 512, 1024, … and puts every
#   column of a sweep into a handful of sets; strides within a few elements of
#   such a multiple (n = 511, 513, 1023, …) are nearly as bad.
# - block `i` starts at a byte offset ≡ `slots[i]` (mod 4096) from the parent's
#   base, so the X and Z entries of one generator (and the destabilizer blocks)
#   do not share a set. Separately allocated page-aligned arrays would put
#   `x[q, j]` and `z[q, j]` in the same set for every `j`.
#
# Below `_PAD_MIN_N` qudits a tableau fits in L1 and the blocks are unpadded
# and adjacent.

const _BlockMatrix = SubArray{Int, 2, Matrix{Int}, Tuple{Base.OneTo{Int}, UnitRange{Int}}, false}

const _PAD_MIN_N = 32
const _PAD_UNIT = 128 ÷ sizeof(Int)    # elements per 128 bytes

_padded_rows(n::Int) = n < _PAD_MIN_N ? n : n + mod(_PAD_UNIT - n, 2 * _PAD_UNIT)

# `length(slots)` zeroed `n × n` blocks of one fresh parent matrix.
function _padded_blocks(n::Int, slots::NTuple{N,Int}) where {N}
    L = _padded_rows(n)
    starts = zeros(Int, N)
    c = 0
    for i in 1:N
        # `L · sizeof(Int)` is an odd multiple of 128, so every slot is reached
        # within 32 columns.
        if n >= _PAD_MIN_N
            while mod(c * L * sizeof(Int), 4096) != slots[i]
                c += 1
            end
        end
        starts[i] = c
        c += n
    end
    P = zeros(Int, L, c)
    return ntuple(i -> view(P, Base.OneTo(n), starts[i]+1:starts[i]+n), Val(N))::NTuple{N,_BlockMatrix}
end

# The stabilizer X and Z blocks, and the destabilizer blocks of a
# `DestabilizerTableau`, which go to the slots between them.
_stab_blocks(n::Int) = _padded_blocks(n, (0, 2048))
_destab_blocks(n::Int) = _padded_blocks(n, (1024, 3072))

# Copies of blocks that share one parent, sharing one new parent in the same places.
function _copy_blocks(blocks::NTuple{N,_BlockMatrix}) where {N}
    P = copy(parent(blocks[1]))
    return ntuple(i -> view(P, parentindices(blocks[i])...), Val(N))
end

# Overwrite the parent of `dst` with that of `src`; both come from `_padded_blocks`
# with the same `n` and slots, so every block sits in the same place.
function _copy_parent!(dst::_BlockMatrix, src::_BlockMatrix)
    copyto!(parent(dst), parent(src))
    return dst
end

# Zero an `Int` array of any layout; the `Array{Int}` method in helper.jl is one `memset`.
_zero!(A::AbstractArray{Int}) = fill!(A, 0)

# One `memset` per column of a block.
function _zero!(A::_BlockMatrix)
    P = parent(A)
    L = size(P, 1)
    nb = size(A, 1) * sizeof(Int)
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
function _stacked_views(S::AbstractMatrix{Int}, n::Int, storephase::Bool)
    nc = size(S, 2)
    b = XZBlock{GeneratorMajor}(view(S, 1:n, :), view(S, n+1:2n, :))
    # The empty view must also exist for a matrix with no rows, so it is built
    # without the bounds check of `view` (which `@inbounds` does not reach on
    # Julia 1.10). Both branches have the same type.
    phase = storephase ? view(S, 2n+1, 1:nc) : SubArray(S, (1, 1:0))
    return b, phase
end

const _GMBlock = XZBlock{GeneratorMajor}

##### software prefetch #####
#
# A gate touches the same few qudits of every generator, one cache line per
# generator and block, at a column stride the hardware prefetchers do not
# follow once the blocks spill out of the last private cache. `apply!` then
# prefetches the lines `_PREFETCH_COLS` generators ahead, but only for tableaux
# whose blocks exceed `_PREFETCH_MIN_BYTES`: below that the lines are cache
# hits and the prefetch instructions cost 5-15% per gate. Both constants are
# measured on an M3 Pro (16 MiB L2 per cluster).

const _PREFETCH_COLS = 32
const _PREFETCH_MIN_BYTES = 16 * 2^20

# Read prefetch with high temporal locality; a hint that never faults.
@static if VERSION >= v"1.11"
    @inline _prefetch(p::Ptr{Int}) =
        ccall("llvm.prefetch.p0", llvmcall, Cvoid, (Ptr{Int}, Int32, Int32, Int32),
              p, Int32(0), Int32(3), Int32(1))
else
    @inline _prefetch(p::Ptr{Int}) = Base.llvmcall(("""
        declare void @llvm.prefetch(i8*, i32, i32, i32)
        define void @entry(i64 %a) #0 {
          %p = inttoptr i64 %a to i8*
          call void @llvm.prefetch(i8* %p, i32 0, i32 3, i32 1)
          ret void
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Cvoid, Tuple{Ptr{Int}}, p)
end

# Whether blocks of `nbytes` in all are large enough to prefetch.
@inline _prefetches(nbytes::Int) = nbytes > _PREFETCH_MIN_BYTES

# Bytes of the parent matrices behind the given blocks; blocks sharing a parent count once.
_block_bytes(b::_GMBlock) = parent(b.x) === parent(b.z) ? sizeof(parent(b.x)) :
                                                        sizeof(parent(b.x)) + sizeof(parent(b.z))

# Prefetch the lines holding qudits `targets` of generator `j` in both blocks.
# The caller keeps the block's arrays alive.
@inline function _prefetch_gen(b::XZBlock{GeneratorMajor, _BlockMatrix},
                               targets::NTuple{K,Int}, j::Int) where {K}
    x, z = b.x, b.z
    for i in 1:K
        _prefetch(pointer(x, CartesianIndex(targets[i], j)))
        _prefetch(pointer(z, CartesianIndex(targets[i], j)))
    end
    return nothing
end

@inline _nq(b::_GMBlock) = size(b.x, 1)
@inline _ncap(b::_GMBlock) = size(b.x, 2)

@inline _xget(b::_GMBlock, q::Int, j::Int) = @inbounds b.x[q, j]
@inline _zget(b::_GMBlock, q::Int, j::Int) = @inbounds b.z[q, j]
@inline _xset!(b::_GMBlock, v::Int, q::Int, j::Int) = @inbounds b.x[q, j] = v
@inline _zset!(b::_GMBlock, v::Int, q::Int, j::Int) = @inbounds b.z[q, j] = v

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

# Stacked vectors `[x; z]` of length at least `2n`.
@inline _vx(v::AbstractVector{Int}, n::Int) = view(v, 1:n)
@inline _vz(v::AbstractVector{Int}, n::Int) = view(v, n+1:2n)

function _copy_gen!(v::AbstractVector{Int}, b::_GMBlock, j::Int)
    n = _nq(b)
    copyto!(_vx(v, n), _xgen(b, j))
    copyto!(_vz(v, n), _zgen(b, j))
    return v
end

function _load_gen!(b::_GMBlock, j::Int, v::AbstractVector{Int})
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
function _gen_addmul!(b::_GMBlock, j::Int, v::AbstractVector{Int}, a::Int, d::Int)
    n = _nq(b)
    addmul_mod!(_xgen(b, j), _vx(v, n), a, d)
    addmul_mod!(_zgen(b, j), _vz(v, n), a, d)
    return b
end
function _gen_submul!(b::_GMBlock, j::Int, v::AbstractVector{Int}, a::Int, d::Int)
    n = _nq(b)
    submul_mod!(_xgen(b, j), _vx(v, n), a, d)
    submul_mod!(_zgen(b, j), _vz(v, n), a, d)
    return b
end
function _gen_mulcopy!(b::_GMBlock, j::Int, v::AbstractVector{Int}, a::Int, d::Int)
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
function _vec_addmul!(v::AbstractVector{Int}, b::_GMBlock, j::Int, a::Int, d::Int)
    n = _nq(b)
    addmul_mod!(_vx(v, n), _xgen(b, j), a, d)
    addmul_mod!(_vz(v, n), _zgen(b, j), a, d)
    return v
end
function _vec_submul!(v::AbstractVector{Int}, b::_GMBlock, j::Int, a::Int, d::Int)
    n = _nq(b)
    submul_mod!(_vx(v, n), _xgen(b, j), a, d)
    submul_mod!(_vz(v, n), _zgen(b, j), a, d)
    return v
end

# Site-restricted forms: only qudits `sites[1:nsites]` of generator j change.
# The stacked `v` is a separate vector from the block, so it cannot alias.
function _gen_addmul_sites!(b::_GMBlock, j::Int, v::AbstractVector{Int},
                            sites::AbstractVector{Int}, nsites::Int, a::Int, d::Int)
    n = _nq(b)
    addmul_mod_sites!(_xgen(b, j), _zgen(b, j), _vx(v, n), _vz(v, n), sites, nsites, a, d)
    return b
end
function _gen_submul_sites!(b::_GMBlock, j::Int, v::AbstractVector{Int},
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
@inline function dot_xz_col(b::_GMBlock, i::Int, j::Int, d::Int)
    s = 0
    x = b.x
    z = b.z
    @turbo for q in 1:size(x, 1)
        s += x[q, i] * z[q, j]
    end
    return mod(s, d)
end
@inline dot_xz_col(b::_GMBlock, j::Int, d::Int) = dot_xz_col(b, j, j, d)

@inline function dot_xz_ws_vs_col(genws::Vector{Int}, b::_GMBlock, tgt::Int, d::Int)
    s = 0
    z = b.z
    @turbo for q in 1:size(z, 1)
        s += genws[q] * z[q, tgt]
    end
    return mod(s, d)
end

@inline function dot_xz_col_vs_zacc(b::_GMBlock, j::Int, zacc::Vector{Int}, d::Int)
    s = 0
    x = b.x
    @turbo for q in 1:size(x, 1)
        s += x[q, j] * zacc[q]
    end
    return mod(s, d)
end
