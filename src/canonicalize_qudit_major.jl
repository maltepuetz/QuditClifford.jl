##########################################
# Qudit-major canonicalization kernel    #
##########################################
#
# In the qudit-major layout the combined row `r` of the generators (X rows
# `1:n`, then Z rows) is the contiguous column of qudit `r` (or `r - n`) over
# the generators. Each pivot step of `_canonicalize_tableau!` therefore reads
# the multipliers β_j of every generator as one column and eliminates the pivot
# row with one rank-1 update per qudit column, `col_q -= β · g_c[q]`, instead
# of one strided row update per generator:
#
# 1. the pivot search, swap, scaling and x·z of the pivot generator `c` are the
#    generic steps;
# 2. β (column 1 of `coeff_workspace`) is a copy of the pivot column with
#    β_c = 0, so row `c` keeps its entries; the generators with β_j ≠ 0 are
#    listed in `c_workspace[1:k]`;
# 3. with `storephase`, the cross terms x_c·z_j (column 2) come from the
#    columns of the pivot's X support;
# 4. a `DestabilizerTableau` adds Σ_j β_j D_j into dual `c`, one sum per qudit
#    column;
# 5. the rank-1 update of every column on which the pivot is nonzero, and the
#    phases by the generic formula (`_eliminated_phase`).
#
# With few nonzero β_j (at most `m ÷ _QM_CANON_SPARSE_DIV`), steps 3-5 visit
# only the listed rows of each column, with scalar arithmetic; otherwise whole
# columns, with the vector primitives. At d = 2 a `DestabilizerTableau` keeps
# its duals bit-packed in `bit_workspace` for the whole call (one generator per
# column, as `_pack_sites_gf2!` writes them), so step 4 is a few word XORs, and
# writes them back at the end. Every path gives the generic results bit for
# bit: the same pivots, multipliers, phase formulas and reduced entries.

# Visit only the listed rows when at most `m ÷ _QM_CANON_SPARSE_DIV` generators
# besides the pivot have a nonzero entry in the pivot row. Measured on an M3 Pro
# (UInt8, n = 1024 and 2048, monitored and random states, d = 2, 3, 31, both
# tableau types), every divisor from 4 to 64 is within 6% of the best and 2 is
# up to 6x slower on random states.
const _QM_CANON_SPARSE_DIV = 8

_canonicalize!(tab::AbstractTableau, L::QuditMajor) =
    _canonicalize_qm!(tab, L, _QM_CANON_SPARSE_DIV)

# How the duals follow the column operations: not at all (`nothing`, a
# `StabilizerTableau`), bit-packed at d = 2, with sums in an unsigned
# accumulator `A`, or (when `A` would overflow even as `UInt64`) row by row.
# The row-by-row form needs (d-1)^2 m + d - 1 > typemax(UInt64), which lies
# outside the envelope `max(n, 2)(d-1)^2 <= typemax(Int)` that
# `_warn_if_dimension_unsafe` warns about; it stays as a defence for a tableau
# built past that warning, and test/layout.jl forces it.
struct _QMDualBits
    W::Matrix{UInt64}
    nw::Int     # words per generator, cld(2n, 64)
end
struct _QMDualAcc{A} end
struct _QMDualRows end

# `canonicalize!` on a qudit-major tableau. `div` sets the sparse cutoff
# (`_QM_CANON_SPARSE_DIV`); every choice gives the same result.
function _canonicalize_qm!(tab::StabilizerTableau, L::QuditMajor, div::Int)
    _canonicalize_qm!(_stab(tab, L), _phase(tab), nothing, nothing, tab, div)
    return nothing
end

function _canonicalize_qm!(tab::DestabilizerTableau, L::QuditMajor, div::Int)
    b = _stab(tab, L)
    db = _destab(tab, L)
    m, d = tab.m, tab.d
    if d == 2
        _pack_sites_gf2!(tab.bit_workspace, db, m, Base.OneTo(tab.n), tab.n)
        _canonicalize_qm!(b, _phase(tab), db, _QMDualBits(tab.bit_workspace, cld(2 * tab.n, 64)), tab, div)
        _qm_unpack_sites_gf2!(db, tab.bit_workspace, m)
        return nothing
    end
    # The narrowest accumulator that holds `m` products below `d^2` plus an
    # entry below `d`. Above `d - 1 = typemax(UInt32)` one product alone needs
    # more than 64 bits; below it the bound is under 2^127.
    bound = d - 1 > typemax(UInt32) ? typemax(UInt128) :
        UInt128(d - 1)^2 * UInt128(m) + UInt128(d - 1)
    if bound <= typemax(UInt16)
        _canonicalize_qm!(b, _phase(tab), db, _QMDualAcc{UInt16}(), tab, div)
    elseif bound <= typemax(UInt32)
        _canonicalize_qm!(b, _phase(tab), db, _QMDualAcc{UInt32}(), tab, div)
    elseif bound <= typemax(UInt64)
        _canonicalize_qm!(b, _phase(tab), db, _QMDualAcc{UInt64}(), tab, div)
    else
        _canonicalize_qm!(b, _phase(tab), db, _QMDualRows(), tab, div)
    end
    return nothing
end

# Write the bit-packed generators `1:m` of `W` (site `q` at bits 2q-2 and
# 2q-1, as `_pack_sites_gf2!` packs all sites) back into the qudit-major block
# `b`, in tiles of 64 generators by the 32 sites of one word.
function _qm_unpack_sites_gf2!(b::_QMBlock, W::AbstractMatrix{UInt64}, m::Int)
    n = _nq(b)
    X, Z = b.x, b.z
    E = eltype(X)
    @inbounds for j0 in 1:64:m
        j1 = min(j0 + 63, m)
        for w in 1:cld(2n, 64)
            k0 = 32 * (w - 1)
            for t in 1:min(32, n - k0)
                q = k0 + t
                @simd for j in j0:j1
                    v = W[w, j]
                    X[j, q] = ((v >> (2t - 2)) & one(UInt64)) % E
                    Z[j, q] = ((v >> (2t - 1)) & one(UInt64)) % E
                end
            end
        end
    end
    return b
end

# Exchange duals `c` and `j`.
_qm_dual_swap!(::Nothing, ::Nothing, ::Int, ::Int) = nothing
_qm_dual_swap!(db::_QMBlock, ::Union{_QMDualAcc, _QMDualRows}, c::Int, j::Int) =
    (_swap_gens!(db, c, j); nothing)
function _qm_dual_swap!(::_QMBlock, dual::_QMDualBits, c::Int, j::Int)
    W = dual.W
    @inbounds for w in 1:dual.nw
        W[w, c], W[w, j] = W[w, j], W[w, c]
    end
    return nothing
end

# Scale dual `c` by `a` (the inverse of the pivot generator's scaling). At
# d = 2, `a = 1`.
_qm_dual_scale!(::Nothing, ::Nothing, ::Int, ::Int, ::Int) = nothing
_qm_dual_scale!(db::_QMBlock, ::Union{_QMDualAcc, _QMDualRows}, c::Int, a::Int, d::Int) =
    (_gen_scale!(db, c, a, d); nothing)
_qm_dual_scale!(::_QMBlock, ::_QMDualBits, ::Int, ::Int, ::Int) = nothing

# D_c += Σ β_j D_j (mod d) over the generators `j` listed in `idx[1:k]`
# (β_c = 0, so `c` is never listed). `dense` says whether at least
# `m ÷ div` are listed, so that the sums run over all of `1:m` with SIMD.
_qm_dual_accumulate!(::Nothing, ::Nothing, ::Int, beta, idx, ::Int, ::Int, ::Int, ::Bool) = nothing

function _qm_dual_accumulate!(db::_QMBlock, ::_QMDualRows, c::Int, beta::AbstractVector,
                              idx::Vector{Int}, k::Int, ::Int, d::Int, ::Bool)
    @inbounds for t in 1:k
        _gen_addmul!(db, c, db, idx[t], Int(beta[idx[t]]), d)
    end
    return nothing
end

function _qm_dual_accumulate!(::_QMBlock, dual::_QMDualBits, c::Int, ::AbstractVector,
                              idx::Vector{Int}, k::Int, ::Int, ::Int, ::Bool)
    W = dual.W
    @inbounds for t in 1:k
        jj = idx[t]
        @simd for w in 1:dual.nw
            W[w, c] ⊻= W[w, jj]
        end
    end
    return nothing
end

function _qm_dual_accumulate!(db::_QMBlock, ::_QMDualAcc{A}, c::Int, beta::AbstractVector,
                              idx::Vector{Int}, k::Int, m::Int, d::Int, dense::Bool) where {A}
    for blk in (db.x, db.z)
        @inbounds for q in axes(blk, 2)
            s = zero(A)
            if dense
                @simd for j in 1:m
                    s += (beta[j] % A) * (blk[j, q] % A)
                end
            else
                for t in 1:k
                    j = idx[t]
                    s += (beta[j] % A) * (blk[j, q] % A)
                end
            end
            s == 0 && continue
            blk[c, q] = rem(s + blk[c, q] % A, d % A) % eltype(blk)
        end
    end
    return nothing
end

# The kernel on the blocks of `tab`, with its workspaces; then the x·z cache.
function _canonicalize_qm!(b::_QMBlock, phase::AbstractVector{Int}, db, dual,
                           tab::AbstractTableau, div::Int)
    _canonicalize_qm!(b, phase, db, dual, tab.m, tab.d, tab.storephase, tab.inversemod,
                      tab.coeff_workspace, tab.c_workspace, tab.pivcol_of_row, div)
    _qm_xdotz_cache!(tab.xdotz_cache, b, tab.m, tab.d)
    tab.iscanonical = true
    return nothing
end

# The elimination on the generators `1:m` of `b` (and the duals `db`, through
# `dual`), with the multipliers and cross terms in columns 1 and 2 of `ws`
# (`m × 2` at least, of the entry type of `b`), the list of nonzero
# multipliers in `idx` (length `m` at least) and the pivot table in
# `pivcol_of_row` (length `2n`). It reads and writes nothing else, so it also
# runs on a private copy of a tableau's blocks with fresh scratch.
function _canonicalize_qm!(b::_QMBlock, phase::AbstractVector{Int}, db, dual,
                           m::Int, d::Int, storephase::Bool, inv::InverseMod,
                           ws::AbstractMatrix, idx::Vector{Int},
                           pivcol_of_row::Vector{Int}, div::Int)
    E = eltype(ws)
    fill!(pivcol_of_row, 0)

    n = _nq(b)
    if storephase
        @assert length(phase) == n
        d_phase = phase_modulus(d)
        inv2 = (d == 2) ? 0 : inv(2, d)
    else
        d_phase = 0
        inv2 = 0
    end
    @assert _ncap(b) == n
    (0 <= m <= n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ n"))

    beta = view(ws, 1:m, 1)
    cross = view(ws, 1:m, 2)
    x = b.x
    z = b.z
    r = 1
    c = 1
    while r <= 2n && c <= m
        col = r <= n ? _qm_xcol(b, r, m) : _qm_zcol(b, r - n, m)
        j = c
        @inbounds while j <= m && col[j] == 0
            j += 1
        end
        if j > m
            r += 1
            continue
        end

        if j != c
            _swap_gens!(b, c, j)
            if storephase
                @inbounds phase[c], phase[j] = phase[j], phase[c]
            end
            _qm_dual_swap!(db, dual, c, j)
        end

        pivot = @inbounds Int(col[c])
        α = inv(pivot, d)
        xdotz = storephase ? dot_xz_col(b, c, d) : 0
        if storephase
            @inbounds phase[c] = _scaled_pivot_phase(phase[c], α, xdotz, d, d_phase, inv2)
        end
        _gen_scale!(b, c, α, d)
        _qm_dual_scale!(db, dual, c, mod(pivot, d), d)
        if storephase && d != 2
            xdotz = mod(mod(α * α, d) * xdotz, d)
        end

        # β over the generators, zero at the pivot, and the generators with a
        # nonzero β in idx[1:k].
        copyto!(beta, col)
        @inbounds beta[c] = zero(E)
        k = _qm_list_nonzero!(idx, beta)
        dense = k > m ÷ div

        if k > 0 && !dense
            if storephase
                @inbounds for t in 1:k
                    cross[idx[t]] = zero(E)
                end
                @inbounds for q in 1:n
                    xc = _xget(b, q, c)
                    xc == 0 && continue
                    for t in 1:k
                        jj = idx[t]
                        cross[jj] = rem(Int(cross[jj]) + xc * Int(z[jj, q]), d) % E
                    end
                end
            end
            _qm_dual_accumulate!(db, dual, c, beta, idx, k, m, d, false)
            @inbounds for q in 1:n
                xc = _xget(b, q, c)
                xc == 0 || _qm_submul_rows!(x, q, idx, k, beta, xc, d)
                zc = _zget(b, q, c)
                zc == 0 || _qm_submul_rows!(z, q, idx, k, beta, zc, d)
            end
        elseif k > 0
            if storephase
                fill!(cross, zero(E))
                @inbounds for q in 1:n
                    xc = _xget(b, q, c)
                    xc == 0 && continue
                    addmul_mod!(cross, _qm_zcol(b, q, m), xc, d)
                end
            end
            _qm_dual_accumulate!(db, dual, c, beta, idx, k, m, d, true)
            @inbounds for q in 1:n
                xc = _xget(b, q, c)
                xc == 0 || submul_mod!(_qm_xcol(b, q, m), beta, xc, d)
                zc = _zget(b, q, c)
                zc == 0 || submul_mod!(_qm_zcol(b, q, m), beta, zc, d)
            end
        end
        # The phases take the cross terms of the old generators, which the
        # updates above leave in column 2.
        if storephase
            kc = @inbounds phase[c]
            @inbounds for t in 1:k
                jj = idx[t]
                phase[jj] = _eliminated_phase(phase[jj], kc, Int(beta[jj]), Int(cross[jj]), xdotz,
                                              d, d_phase, inv2)
            end
        end

        @inbounds pivcol_of_row[r] = c
        r += 1
        c += 1
    end
    return nothing
end

# x·z of every generator `1:m` of `b` into `xdotz_cache`, accumulated column by
# column. `Int` sums wrap the same way in any order, so the reduction equals
# `dot_xz_col`'s. The loop is `@simd`, not `@turbo`: with narrow entries and
# `Int` sums, the code `@turbo` generates for 512-bit registers corrupts memory
# on Julia 1.10.
function _qm_xdotz_cache!(xdotz_cache::Vector{Int}, b::_QMBlock, m::Int, d::Int)
    fill!(xdotz_cache, 0)
    @inbounds for q in 1:_nq(b)
        xq = _qm_xcol(b, q, m)
        zq = _qm_zcol(b, q, m)
        @simd for jj in 1:m
            xdotz_cache[jj] += Int(xq[jj]) * Int(zq[jj])
        end
    end
    @inbounds for jj in 1:m
        xdotz_cache[jj] = mod(xdotz_cache[jj], d)
    end
    return xdotz_cache
end
