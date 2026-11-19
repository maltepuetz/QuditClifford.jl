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

# Zero an `Int` array of any layout; the `Array{Int}` method in helper.jl is one `memset`.
_zero!(A::AbstractArray{Int}) = fill!(A, 0)
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
    # `@inbounds`: the empty view must also exist for a matrix with no rows.
    phase = storephase ? view(S, 2n+1, 1:nc) : @inbounds(view(S, 1, 1:0))
    return b, phase
end

const _GMBlock = XZBlock{GeneratorMajor}

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

# The `helper.jl` dot products on a block; `n` is the number of qudits of `b`.
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
