#############################################################
# Span decomposition helpers shared by expectation + measure #
#############################################################

"""
    coeffs_from_generators!(tab, op)

Fill `tab.c_workspace` with coefficients of `op` in the active stabilizer-generator basis.

- For `StabilizerTableau`, coefficients are read from pivot rows of the canonical tableau.
- For `DestabilizerTableau`, coefficients are computed via dual symplectic pairings.
"""
function coeffs_from_generators! end

@inline function coeffs_from_generators!(tab::StabilizerTableau, op)
    n = tab.n
    d = tab.d

    tab.iscanonical || canonicalize!(tab)

    cvec = tab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end

    # Read coefficients from pivot rows in canonical form.
    opvec = tab.res_workspace
    set_operator!(opvec, tab, op)
    piv = tab.pivcol_of_row
    @inbounds for r in 1:(2n)
        pc = piv[r]
        pc == 0 && continue
        cvec[pc] = mod(opvec[r], d)
    end

    return cvec
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op)
    n = tab.n
    m = tab.m
    d = tab.d
    destab = tab.destab

    cvec = tab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end

    # Use res_workspace as temporary xz(op) storage.
    opvec = tab.res_workspace
    set_operator!(opvec, tab, op)

    @inbounds for j in 1:m
        coeff = 0
        @turbo for q in 1:n
            coeff += destab[q, j] * opvec[n + q]
        end
        @turbo for q in 1:n
            coeff -= destab[n + q, j] * opvec[q]
        end
        cvec[j] = mod(coeff, d)
    end

    return cvec
end

@inline function _coeffs_from_destab_sparse!(
    tab::DestabilizerTableau,
    qudits::NTuple{D,Int},
    xs::NTuple{D,Int},
    zs::NTuple{D,Int},
) where D
    m = tab.m
    n = tab.n
    d = tab.d
    destab = tab.destab

    cvec = tab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end

    @inbounds for j in 1:m
        coeff = 0
        @simd for i in 1:D
            q = qudits[i]
            coeff += destab[q, j] * zs[i]
            coeff -= destab[n + q, j] * xs[i]
        end
        cvec[j] = mod(coeff, d)
    end
    return cvec
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op::SinglePauli)
    d = tab.d
    return _coeffs_from_destab_sparse!(
        tab,
        (op.qudit,),
        (mod(op.x, d),),
        (mod(op.z, d),),
    )
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op::DoublePauli)
    d = tab.d
    return _coeffs_from_destab_sparse!(
        tab,
        (op.qudit1, op.qudit2),
        (mod(op.x1, d), mod(op.x2, d)),
        (mod(op.z1, d), mod(op.z2, d)),
    )
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op::TriplePauli)
    d = tab.d
    return _coeffs_from_destab_sparse!(
        tab,
        (op.qudit1, op.qudit2, op.qudit3),
        (mod(op.x1, d), mod(op.x2, d), mod(op.x3, d)),
        (mod(op.z1, d), mod(op.z2, d), mod(op.z3, d)),
    )
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op::NPauli{D}) where D
    d = tab.d
    xs = ntuple(i -> mod(op.xs[i], d), D)
    zs = ntuple(i -> mod(op.zs[i], d), D)
    return _coeffs_from_destab_sparse!(tab, op.qudits, xs, zs)
end

"""
    residual_from_coeffs!(tab, op)

Compute `op - Σ_j c_j g_j` in XZ space and store it in `tab.res_workspace`,
where `c_j` are taken from `tab.c_workspace` and `g_j` are active stabilizer generators.
"""
@inline function residual_from_coeffs!(tab::AbstractTableau, op)
    res = tab.res_workspace
    set_operator!(res, tab, op)

    d = tab.d
    m = tab.m
    n = tab.n
    cvec = tab.c_workspace
    stab = tab.stab

    @inbounds for j in 1:m
        aj = cvec[j]
        aj == 0 && continue
        submul_mod!(view(res, 1:(2n)), view(stab, 1:(2n), j), aj, d)
    end
    return res
end

"""
    in_span_and_coeffs!(tab, op)

Fill coefficients in `tab.c_workspace`, residual in `tab.res_workspace`,
and return whether `op` is in the span of active stabilizer generators.
"""
@inline function in_span_and_coeffs!(tab::AbstractTableau, op)
    coeffs_from_generators!(tab, op)
    return _in_span_from_coeffs!(tab, op)
end

# The residual and in-span test for coefficients already in `c_workspace`.
@inline function _in_span_from_coeffs!(tab::AbstractTableau, op)
    residual_from_coeffs!(tab, op)

    n = tab.n
    res = tab.res_workspace
    @inbounds for i in 1:(2n)
        res[i] != 0 && return false
    end
    return true
end

"""
    phase_exponent_from_coeffs!(tab::AbstractTableau)

Compute the phase exponent of `Q = ∏_j g_j^{c[j]}` from `tab.c_workspace`,
where `g_j` are active stabilizer generators.
"""
@inline function phase_exponent_from_coeffs!(tab::AbstractTableau)
    n = tab.n
    m = tab.m
    d = tab.d
    stab = tab.stab
    tab.storephase || return 0

    prow = 2n + 1
    d_phase = phase_modulus(d)

    cvec = tab.c_workspace
    zacc = tab.zacc_workspace
    @turbo for q in 1:n
        zacc[q] = 0
    end
    kacc = 0

    if d == 2
        @inbounds for j in 1:m
            aj = cvec[j]
            aj == 0 && continue

            cross = 0
            @turbo for q in 1:n
                cross += stab[q, j] * zacc[q]
            end
            cross = cross & 1

            kacc = mod(kacc + stab[prow, j] + 2 * cross, d_phase)

            @inbounds @simd for q in 1:n
                zacc[q] = (zacc[q] + stab[n+q, j]) & 1
            end
        end
        return kacc
    else
        inv2 = tab.inversemod(2, d)
        xdotz_cache = tab.xdotz_cache
        # A DestabilizerTableau keeps `xdotz_cache` current through every
        # update; a StabilizerTableau fills it only when it canonicalizes.
        cached = tab isa DestabilizerTableau || tab.iscanonical

        @inbounds for j in 1:m
            aj = mod(cvec[j], d)
            aj == 0 && continue

            xdotz_j = cached ? xdotz_cache[j] : dot_xz_col(stab, n, j, d)
            kpow = mod(aj * stab[prow, j] + binom2_mod_oddprime(aj, d, inv2) * xdotz_j, d_phase)

            cross_base = dot_xz_col_vs_zacc(stab, n, j, zacc, d)
            cross = mod(aj * cross_base, d)

            kacc = mod(kacc + kpow + cross, d_phase)

            # zacc[1:n] accumulates the Z half of column j, i.e. stab rows n+1:2n.
            addmul_mod!(view(zacc, 1:n), view(stab, (n+1):(2n), j), aj, d)
        end
        return kacc
    end
end

#############################################################
# Coefficient solve on a site-major copy of the stabilizers #
#############################################################
#
# A `StabilizerTableau` reads coefficients off pivot rows, which needs the
# canonical form. Canonicalizing in place costs O(n²m) on Int entries with
# phase updates, and every random measurement or gate undoes it. When the
# tableau is not canonical, `measure!` instead solves G c = op on a copy that
# holds only the exponents mod d:
#
#   1. `_copy_generator_rows!` writes the rows of G, one per qudit and Pauli
#      half, into a workspace: the X row of site q (its exponent in every
#      generator) into column q, the Z row after it at half the column
#      height. This is the only step that knows the tableau layout; a tableau
#      that stores qudit rows already needs only a block copy.
#   2. `_solve_generator_rows!` inserts the rows one at a time into an echelon
#      basis and back-substitutes. It reads and writes rows only.
#
# Row `r` of G is the X row of site r for r <= n and the Z row of site r - n
# above, the order of `set_operator!`, so the reduced right-hand side lives in
# a vector laid out like `res_workspace`.

# Rows of G as entries mod d, one byte (3 <= d <= 255) or one Int (d > 255)
# each: X row of site q in ws[1:m, q], Z row in ws[h+1:h+m, q], with h half
# the column height. The tableau is reduced, so the narrowing is exact.
#
# A transpose, in tiles of 64 generators by 8 sites: within a tile the 64
# generator columns are read as short sequential runs, and only 8 workspace
# columns are written, which keeps them in L1 even when n is a power of two
# and the workspace's column stride maps them to a few cache sets. Measured on
# an M3 Pro, reading one cache line of each of the m columns per block of
# sites instead took twice as long at n = 2000; wider site tiles were faster
# there but 2.6x slower at n = 2048.
function _copy_generator_rows!(
    ws::AbstractMatrix{T},
    stab::AbstractMatrix{Int},
    n::Int,
    m::Int,
) where T<:Union{UInt8,Int}
    h = size(ws, 1) >> 1
    (m <= h && size(ws, 2) >= n) || throw(DimensionMismatch(
        "_copy_generator_rows!: $n sites of $m generators do not fit a $(size(ws)) workspace."))
    @inbounds for j0 in 1:64:m
        j1 = min(j0 + 63, m)
        for q0 in 1:8:n
            q1 = min(q0 + 7, n)
            for j in j0:j1
                @simd for q in q0:q1
                    ws[j, q] = stab[q, j] % T
                    ws[h + j, q] = stab[n + q, j] % T
                end
            end
        end
    end
    return ws
end

# Rows of G as bits for d = 2, generator j at bit (j - 1) & 63 of word
# (j - 1) >> 6 + 1: X row of site q in words 1:cld(m, 64) of column q, Z row
# in words h+1:h+cld(m, 64). Bits past m are zero. Tiled like the byte copy:
# the 64 generators of one word, 128 sites at a time.
function _copy_generator_rows!(
    ws::AbstractMatrix{UInt64},
    stab::AbstractMatrix{Int},
    n::Int,
    m::Int,
)
    h = size(ws, 1) >> 1
    nw = cld(m, 64)
    (nw <= h && size(ws, 2) >= n) || throw(DimensionMismatch(
        "_copy_generator_rows!: $n sites of $m generators do not fit a $(size(ws)) workspace."))
    @inbounds for w in 1:nw
        j0 = 64 * (w - 1)
        for q0 in 1:128:n
            q1 = min(q0 + 127, n)
            @simd for q in q0:q1
                ws[w, q] = zero(UInt64)
                ws[h + w, q] = zero(UInt64)
            end
            for t in 1:min(64, m - j0)
                j = j0 + t
                @simd for q in q0:q1
                    ws[w, q] |= (stab[q, j] % UInt64) << (t - 1)
                    ws[h + w, q] |= (stab[n + q, j] % UInt64) << (t - 1)
                end
            end
        end
    end
    return ws
end

# Row r of G sits in workspace column `q` from row `off + 1` on.
@inline _generator_row_slot(r::Int, n::Int, h::Int) = r <= n ? (r, 0) : (r - n, h)

# The first k in ℓ:m with ws[off + k, q] != 0, or m + 1 if there is none.
@inline function _next_nonzero_entry(ws::AbstractMatrix, off::Int, q::Int, ℓ::Int, m::Int)
    @inbounds while ℓ <= m && iszero(ws[off + ℓ, q])
        ℓ += 1
    end
    return ℓ
end

# On bytes, eight at a time: every row is scanned once from its start, and
# reduced rows are mostly zero bytes. A little-endian word's lowest set byte
# is the first in memory.
@inline function _next_nonzero_entry(ws::Matrix{UInt8}, off::Int, q::Int, ℓ::Int, m::Int)
    ℓ > m && return ℓ
    if Base.ENDIAN_BOM == 0x04030201
        checkbounds(ws, off + ℓ, q)
        checkbounds(ws, off + m, q)
        # ws[off + k, q] is byte base + k of the array, counting from 1.
        base = (q - 1) * size(ws, 1) + off
        GC.@preserve ws begin
            p = pointer(ws)
            while ℓ + 7 <= m
                x = unsafe_load(Ptr{UInt64}(p + base + ℓ - 1))
                iszero(x) || return ℓ + (trailing_zeros(x) >> 3)
                ℓ += 8
            end
        end
    end
    @inbounds while ℓ <= m && iszero(ws[off + ℓ, q])
        ℓ += 1
    end
    return ℓ
end

# Solve G c = op for the rows that `_copy_generator_rows!` left in `ws`.
#
# `opv[1:2n]` holds op's exponents mod d in the row order above. Rows are
# inserted site by site (X, then Z): each is reduced by the pivot rows whose
# lead (first nonzero generator index) it hits, in increasing lead order, and
# a row that survives becomes the pivot of its first nonzero entry. `opv[r]`
# follows the same row operations. Pivot rows are zero before their lead, so
# a reduction clears that entry without touching earlier ones. Once all m
# leads are taken the remaining rows add nothing, and the coefficients follow
# by back-substitution, lead m first, into `c[1:m]`.
#
# `lead[ℓ]` is the row holding the pivot of generator ℓ. Independent
# generators give m pivots; a lead without one (generators that were not
# independent) gets coefficient 0. Whether G c actually equals op is left to
# `residual_from_coeffs!`: when op is outside the span, c is some vector and
# the residual is nonzero. Returns the number of pivots.
function _solve_generator_rows!(
    ws::AbstractMatrix{T},
    opv::AbstractVector{Int},
    lead::AbstractVector{Int},
    c::AbstractVector{Int},
    n::Int,
    m::Int,
    d::Int,
    inversemod::InverseMod,
) where T<:Union{UInt8,Int}
    h = size(ws, 1) >> 1
    @inbounds for ℓ in 1:m
        lead[ℓ] = 0
    end
    rank = 0
    @inbounds for q in 1:n, half in 0:1
        rank == m && break
        r = q + half * n
        off = half * h
        o = opv[r]
        ℓ = 1
        while true
            ℓ = _next_nonzero_entry(ws, off, q, ℓ, m)
            ℓ > m && break
            p = lead[ℓ]
            if p == 0
                lead[ℓ] = r
                rank += 1
                break
            end
            pq, poff = _generator_row_slot(p, n, h)
            β = mod(Int(ws[off + ℓ, q]) * inversemod(Int(ws[poff + ℓ, pq]), d), d)
            # Rows of different sites sit in different columns, and the X and
            # Z rows of one site in disjoint halves of its column (m <= h).
            _submul_col!(view(ws, (off + ℓ):(off + m), q), view(ws, (poff + ℓ):(poff + m), pq), β, d)
            o = mod(o - β * opv[p], d)
            ℓ += 1
        end
        opv[r] = o
    end

    @inbounds for ℓ in m:-1:1
        p = lead[ℓ]
        if p == 0
            c[ℓ] = 0
            continue
        end
        pq, poff = _generator_row_slot(p, n, h)
        s = 0
        @simd for k in (ℓ + 1):m
            s += Int(ws[poff + k, pq]) * c[k]
        end
        rhs = mod(opv[p] - s, d)
        c[ℓ] = mod(rhs * inversemod(Int(ws[poff + ℓ, pq]), d), d)
    end
    return rank
end

# Bit k of a set of coefficients packed into the integers of `cw`, as many
# bits per entry as its element type has: two entries per 64-bit word where
# `Int` is `Int32`.
@inline function _set_coeff_bit!(cw::AbstractVector{T}, k::Int) where T<:Signed
    b = 8 * sizeof(T)
    @inbounds cw[(k - 1) ÷ b + 1] |= one(T) << ((k - 1) % b)
    return cw
end

# 64-bit word `w` (bits 64w-63:64w) of the set in `cw`.
@inline function _coeff_word(cw::AbstractVector{T}, w::Int) where T<:Signed
    per = 8 ÷ sizeof(T)
    x = zero(UInt64)
    @inbounds for i in 1:per
        x |= UInt64(cw[per * (w - 1) + i] % unsigned(T)) << (8 * sizeof(T) * (i - 1))
    end
    return x
end

# The GF(2) version on packed rows. Every nonzero entry is 1, so a reduction
# is an XOR of the words from the lead's word on, and back-substitution is the
# parity of a pivot row ANDed with the coefficients found so far, which are
# packed into `cw` as they appear (`_set_coeff_bit!`). `cw` needs
# `cld(m, 64) * 8 ÷ sizeof(eltype(cw))` entries.
function _solve_generator_rows!(
    ws::AbstractMatrix{UInt64},
    opv::AbstractVector{Int},
    lead::AbstractVector{Int},
    c::AbstractVector{Int},
    cw::AbstractVector{<:Signed},
    n::Int,
    m::Int,
)
    h = size(ws, 1) >> 1
    nw = cld(m, 64)
    @inbounds for ℓ in 1:m
        lead[ℓ] = 0
    end
    rank = 0
    @inbounds for q in 1:n, half in 0:1
        rank == m && break
        r = q + half * n
        off = half * h
        o = opv[r]
        w = 1
        while w <= nw
            x = ws[off + w, q]
            if iszero(x)
                w += 1
                continue
            end
            ℓ = 64 * (w - 1) + trailing_zeros(x) + 1
            p = lead[ℓ]
            if p == 0
                lead[ℓ] = r
                rank += 1
                break
            end
            pq, poff = _generator_row_slot(p, n, h)
            @simd for l in w:nw
                ws[off + l, q] = xor(ws[off + l, q], ws[poff + l, pq])
            end
            o = xor(o, opv[p])
        end
        opv[r] = o
    end

    @inbounds for i in 1:(nw * (8 ÷ sizeof(eltype(cw))))
        cw[i] = 0
    end
    @inbounds for ℓ in m:-1:1
        p = lead[ℓ]
        if p == 0
            c[ℓ] = 0
            continue
        end
        pq, poff = _generator_row_slot(p, n, h)
        w0 = (ℓ - 1) >> 6 + 1
        acc = zero(UInt64)
        for w in w0:nw
            acc = xor(acc, ws[poff + w, pq] & _coeff_word(cw, w))
        end
        bit = (opv[p] + count_ones(acc)) & 1
        c[ℓ] = bit
        bit == 1 && _set_coeff_bit!(cw, ℓ)
    end
    return rank
end

"""
    _span_coeffs_keep_basis!(tab, op)

`coeffs_from_generators!` without changing the generator basis: a
`StabilizerTableau` that is not canonical solves for the coefficients on a
copy in its rank workspace for `d` (see `src/rank_workspaces.jl`) instead of
canonicalizing in place. The coefficients of an in-span `op` are unique, so
they agree with the canonicalizing path's; `tab.stab` and `iscanonical` are
left as they were. Writes `c_workspace`, `res_workspace`, `zacc_workspace`
and, at `d = 2`, `generator_workspace`, whose `2n + storephase` entries hold
the `cld(m, 64) * 8 ÷ sizeof(Int) <= 2n` coefficient bit words.
"""
@inline _span_coeffs_keep_basis!(tab::DestabilizerTableau, op) = coeffs_from_generators!(tab, op)

function _span_coeffs_keep_basis!(tab::StabilizerTableau, op)
    tab.iscanonical && return coeffs_from_generators!(tab, op)
    n = tab.n
    m = tab.m
    d = tab.d

    cvec = tab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end
    opv = tab.res_workspace
    set_operator!(opv, tab, op)
    lead = tab.zacc_workspace

    if d == 2
        ws = tab.bit_workspace
        _copy_generator_rows!(ws, tab.stab, n, m)
        _solve_generator_rows!(ws, opv, lead, cvec, tab.generator_workspace, n, m)
    elseif _uses_narrow_rank(d)
        ws = tab.narrow_workspace
        _copy_generator_rows!(ws, tab.stab, n, m)
        _solve_generator_rows!(ws, opv, lead, cvec, n, m, d, tab.inversemod)
    else
        ws = tab.workspace
        _copy_generator_rows!(ws, tab.stab, n, m)
        _solve_generator_rows!(ws, opv, lead, cvec, n, m, d, tab.inversemod)
    end
    return cvec
end
