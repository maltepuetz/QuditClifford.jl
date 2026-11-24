########################################
# Packed qubit span decomposition      #
########################################
#
# The commuting branches of `measure!` (fixed outcome, appended generator) and
# `expect!`/`expect_int!` on packed storage. The pipeline is the one of
# src/span_decomposition.jl: coefficients `c` of the operand in the active
# generators, the residual `op ⊻ ⊕_j c_j g_j`, and, when the residual vanishes,
# the phase exponent of `∏_j g_j^{c_j}`, multiplied in increasing `j`.
#
# At d = 2, with `g_j = i^{k_j} X^{x_j} Z^{z_j}` and
# `(X^x Z^z)(X^x' Z^z') = (-1)^{z·x'} X^{x+x'} Z^{z+z'}`, that exponent is
#
#     k = Σ_j c_j k_j + 2 Σ_{i<j} c_i c_j (z_i·x_j)   (mod 4),
#
# for Hermitian and non-Hermitian generators alike. The first sum is
# `popcount(c & P0) + 2·popcount(c & P1)` on the phase planes. The second is a
# parity, the one `phase_exponent_from_coeffs!` accumulates generator by
# generator.
#
# `word_workspace` during a call:
#
# - column 1: the coefficients as a mask over generators `1:m`, bits past `m`
#   zero. A `DestabilizerTableau` reads them off its duals (generator-major one
#   commutator per dual, qudit-major the XOR of the dual columns on the
#   operand's support), a canonical `StabilizerTableau` off its pivot rows, and
#   a `StabilizerTableau` that is not canonical packs the solution of
#   `_span_coeffs_keep_basis!` (`measure!`) or is canonicalized first
#   (`expect!`).
# - columns 3 (X) and 4 (Z): the operand packed by qudit before the kernel, the
#   residual after it. The operand is packed before the coefficients are
#   found, because the keep-basis solve overwrites `res_workspace`.
# - column 2: generator-major, the XOR of the Z halves of the generators
#   multiplied so far.
#
# Generator-major, every generator of the mask takes one pass over its words:
# the parity of `x_j & column 2`, and the XOR of its halves into columns 3 and
# 2. Qudit-major, every qudit column `q` takes one pass over the mask words
# with `a = z[:, q] & c` and `b = x[:, q] & c`: the residual bits are the
# parities of `b` and `a`, and the pairs `i < j` with `a_i = b_j = 1` are
# counted mod 2 as `b & (strict prefix parity of a)`, the prefix parity
# carried from word to word (the last carry is the parity of `a`). When few
# mask words are nonzero only those are visited, listed in `zacc_workspace`
# (`_bit_span_sparse_cut`).
#
# A pure state (`m = n`) needs no residual: its generators span every Pauli
# that commutes with them, which `measure!` has established by then and
# `_expect_int!` tests first, so the coefficients are exact. Qudit-major, where
# the residual costs a pass over every qudit column, only the phase is
# computed then (`_qm_bit_mask_product_phase!`): the phase of the one
# generator when the mask has a single bit, else the same pair count without
# the residual. This is the reasoning by which the generic path settles a pure
# state without phases from the commutation check alone; a tableau whose
# `m = n` generators are not independent and commuting gets no error from it.
# Generator-major the residual comes with the phase, so it is always taken.
#
# The append hook of a `DestabilizerTableau` then reuses the columns: the
# residual gives the qudit of the unit vector `e`, the new dual
# `v = e ⊻ ⊕_{j: ⟨e, S_j⟩ = 1} D_j` is built in columns 3 (X) and 4 (Z),
# qudit-major with the pairings in column 1, and `_orthogonalize_duals!` uses
# column 2 qudit-major. Its support goes to `zacc_workspace` qudit-major.
#
# A packed qudit-major `canonicalize!` overwrites all four columns, so
# `_expect_int!` canonicalizes a `StabilizerTableau` before anything is put
# there.

# The qudit-major kernel visits only the nonzero mask words when at most
# `_bit_span_sparse_cut` of the `nwm` words of generators `1:m` are nonzero.
# Per visited word the listed loop takes about 1.4 times the dense one with
# phases (the dense one pairs adjacent words for the compiler to vectorize)
# and 2.2 times without (there the dense loop is a SIMD reduction); measured
# at n = 1024 and 4096 on an M3 Pro.
@inline _bit_span_sparse_cut(nwm::Int, phases::Bool) = phases ? (2 * nwm) ÷ 3 : nwm ÷ 4

# The operand packed by qudit into columns 3 (X) and 4 (Z) of `ws`.
@inline function _bit_pack_operand!(ws::Matrix{UInt64}, op::NPauli{D}, n::Int) where {D}
    @inbounds for w in 1:_nwords(n)
        ws[w, 3] = UInt64(0)
        ws[w, 4] = UInt64(0)
    end
    @inbounds for i in 1:D
        q = op.qudits[i]
        w, o = _word_index(q), _bit_offset(q)
        ws[w, 3] |= ((op.xs[i] % UInt64) & UInt64(1)) << o
        ws[w, 4] |= ((op.zs[i] % UInt64) & UInt64(1)) << o
    end
    return nothing
end
@inline _bit_pack_operand!(ws::Matrix{UInt64}, op::Vector{Int}, n::Int) =
    _gm_bit_pack_operand!(ws, op, n)

########################################
# Coefficient masks                    #
########################################

# Set bit `j` of column 1 of `ws`.
@inline function _bit_mask_set!(ws::Matrix{UInt64}, j::Int)
    @inbounds ws[_word_index(j), 1] |= UInt64(1) << _bit_offset(j)
    return nothing
end

@inline function _bit_mask_zero!(ws::Matrix{UInt64}, m::Int)
    @inbounds for w in 1:_nwords(m)
        ws[w, 1] = UInt64(0)
    end
    return nothing
end

# The coefficients of the operand into column 1 of word_workspace.
#
# Generator-major duals: c_j is the commutator parity of dual `j` with the
# operand; a dense operand is read from columns 3-4.
function _bit_coeff_mask!(tab::_BitDestab, L::GeneratorMajor, op::_BitOperand)
    ws = tab.word_workspace
    destab = _destab(tab, L)
    m = tab.m
    @inbounds for w in 1:_nwords(m)
        j0 = (w - 1) << _WORD_SHIFT
        M = UInt64(0)
        for t in 1:min(_WORD_BITS, m - j0)
            M |= (_gm_bit_comm(destab, j0 + t, op, ws) % UInt64) << (t - 1)
        end
        ws[w, 1] = M
    end
    return nothing
end

# Qudit-major duals: the commutator mask of the duals with the operand.
function _bit_coeff_mask!(tab::_BitDestab, L::QuditMajor, op::_BitOperand)
    _qm_bit_mask!(tab.word_workspace, _destab(tab, L), op, tab.m)
    return nothing
end

# A canonical `StabilizerTableau`: the coefficient of the generator with its
# pivot in row `r` is the operand's exponent in that row.
function _bit_coeff_mask!(tab::_BitStab, ::TableauLayout, op::NPauli{D}) where {D}
    ws = tab.word_workspace
    n = tab.n
    piv = tab.pivcol_of_row
    _bit_mask_zero!(ws, tab.m)
    @inbounds for i in 1:D
        q = op.qudits[i]
        if isodd(op.xs[i])
            pc = piv[q]
            pc == 0 || _bit_mask_set!(ws, pc)
        end
        if isodd(op.zs[i])
            pc = piv[n + q]
            pc == 0 || _bit_mask_set!(ws, pc)
        end
    end
    return nothing
end
function _bit_coeff_mask!(tab::_BitStab, ::TableauLayout, op::Vector{Int})
    ws = tab.word_workspace
    piv = tab.pivcol_of_row
    _bit_mask_zero!(ws, tab.m)
    @inbounds for r in 1:(2 * tab.n)
        pc = piv[r]
        (pc == 0 || iseven(op[r])) && continue
        _bit_mask_set!(ws, pc)
    end
    return nothing
end

# The coefficients `cvec[1:m]`, each 0 or 1, into column 1 of `ws`.
function _bit_coeffs_to_mask!(ws::Matrix{UInt64}, cvec::Vector{Int}, m::Int)
    @inbounds for w in 1:_nwords(m)
        j0 = (w - 1) << _WORD_SHIFT
        M = UInt64(0)
        for t in 1:min(_WORD_BITS, m - j0)
            M |= ((cvec[j0 + t] % UInt64) & UInt64(1)) << (t - 1)
        end
        ws[w, 1] = M
    end
    return nothing
end

########################################
# Residual and phase                   #
########################################

# `Σ_j c_j k_j` mod 4 for the mask in column 1 of `ws` and the phase planes `P`.
@inline function _bit_mask_phase(ws::Matrix{UInt64}, P::Matrix{UInt64}, m::Int)
    s0 = 0
    s1 = 0
    @inbounds for w in 1:_nwords(m)
        M = ws[w, 1]
        s0 += count_ones(M & P[w, 1])
        s1 += count_ones(M & P[w, 2])
    end
    return (s0 + 2 * s1) & 3
end

@inline _bit_parity(x::UInt64) = (count_ones(x) % UInt64) & UInt64(1)

# Bit `k` of the result is the parity of bits `0:k` of `a`.
@inline function _bit_prefix_parity(a::UInt64)
    a ⊻= a << 1
    a ⊻= a << 2
    a ⊻= a << 4
    a ⊻= a << 8
    a ⊻= a << 16
    a ⊻= a << 32
    return a
end

# Whether the residual in columns 3-4 of `ws` vanishes on `n` qudits.
@inline function _bit_residual_zero(ws::Matrix{UInt64}, n::Int)
    bad = UInt64(0)
    @inbounds for w in 1:_nwords(n)
        bad |= ws[w, 3] | ws[w, 4]
    end
    return bad == 0
end

# With the coefficient mask in column 1 of `ws` and the packed operand in
# columns 3-4: leave the residual in columns 3-4 and return whether it
# vanishes and the phase exponent of the product (0 without planes `P`).
function _gm_bit_residual_phase!(ws::Matrix{UInt64}, b::_GMBits,
                                 P::Union{Nothing,Matrix{UInt64}}, m::Int)
    x, z = b.x, b.z
    nw = size(x, 1)
    cr = UInt64(0)
    if P !== nothing
        @inbounds for w in 1:nw
            ws[w, 2] = UInt64(0)
        end
    end
    @inbounds for wm in 1:_nwords(m)
        M = ws[wm, 1]
        j0 = (wm - 1) << _WORD_SHIFT
        while M != 0
            j = j0 + trailing_zeros(M) + 1
            M &= M - UInt64(1)
            if P === nothing
                @simd ivdep for w in 1:nw
                    ws[w, 3] ⊻= x[w, j]
                    ws[w, 4] ⊻= z[w, j]
                end
            else
                @simd ivdep for w in 1:nw
                    xj = x[w, j]
                    cr ⊻= xj & ws[w, 2]
                    ws[w, 3] ⊻= xj
                    ws[w, 2] ⊻= z[w, j]
                end
            end
        end
    end
    P === nothing && return _bit_residual_zero(ws, b.n), 0
    @inbounds for w in 1:nw
        ws[w, 4] ⊻= ws[w, 2]
    end
    kacc = (_bit_mask_phase(ws, P, m) + 2 * (count_ones(cr) & 1)) & 3
    return _bit_residual_zero(ws, b.n), kacc
end

# The pair count of one mask word: `a` and `bq` are the masked Z and X words
# of a qudit column, `carry` is all ones when the Z bits of the earlier words
# have odd parity. Returns a word whose parity is the number of pairs `i < j`
# with `a_i = bq_j = 1`, `i` in this or an earlier word and `j` in this one,
# and the inclusive prefix parities of `a`, whose top bit is the parity of `a`.
@inline function _bit_span_pairs(a::UInt64, bq::UInt64, carry::UInt64)
    p = _bit_prefix_parity(a)
    return bq & ((p << 1) ⊻ carry), p
end

# All ones when the top bit of `p` is set.
@inline _bit_top_mask(p::UInt64) = ((p % Int64) >> (_WORD_BITS - 1)) % UInt64

# With at most this many listed mask words, a qudit column is first tested for
# any mask generator acting on it at all, and skipped if none does: the
# generators of a monitored circuit are local, and a fixed single-qubit
# outcome there selects one or two of them.
const _BIT_SPAN_PROBE_WORDS = 4

# Whether no generator of the listed mask words `list[1:nz]` acts on qudit
# `q`; only asked for `nz <= _BIT_SPAN_PROBE_WORDS`, else `false`.
@inline function _qm_bit_column_untouched(x, z, ws::Matrix{UInt64}, q::Int,
                                          list::Vector{Int}, nz::Int)
    nz <= _BIT_SPAN_PROBE_WORDS || return false
    any = UInt64(0)
    @inbounds for k in 1:nz
        w = list[k]
        any |= (x[w, q] | z[w, q]) & ws[w, 1]
    end
    return any == 0
end

# One qudit column `q` against the mask words `1:nwm`: the XOR of the masked X
# words, the parity of the masked Z words (the last carry, 0 or 1), and a word
# whose parity is the number of pairs `i < j` of mask generators with
# `z_i[q] = x_j[q] = 1`. Two words per pass:
# their prefix parities do not depend on each other, so the compiler computes
# them in one vector (1.3-1.4 times faster than word by word, M3 Pro).
@inline function _qm_bit_span_column(x, z, ws::Matrix{UInt64}, q::Int, nwm::Int)
    rx = UInt64(0)
    cr = UInt64(0)
    carry = UInt64(0)
    w = 1
    @inbounds while w < nwm
        M0 = ws[w, 1]
        M1 = ws[w + 1, 1]
        a0 = z[w, q] & M0
        a1 = z[w + 1, q] & M1
        b0 = x[w, q] & M0
        b1 = x[w + 1, q] & M1
        rx ⊻= b0 ⊻ b1
        s0 = _bit_top_mask(_bit_prefix_parity(a0))
        c0, _ = _bit_span_pairs(a0, b0, carry)
        c1, p1 = _bit_span_pairs(a1, b1, carry ⊻ s0)
        cr ⊻= c0 ⊻ c1
        carry ⊻= s0 ⊻ _bit_top_mask(p1)
        w += 2
    end
    @inbounds if w == nwm
        M = ws[w, 1]
        a = z[w, q] & M
        bq = x[w, q] & M
        rx ⊻= bq
        c, p = _bit_span_pairs(a, bq, carry)
        cr ⊻= c
        carry ⊻= _bit_top_mask(p)
    end
    return rx, carry & UInt64(1), cr
end

# The same over the listed words `list[1:nz]`, in increasing order.
@inline function _qm_bit_span_column(x, z, ws::Matrix{UInt64}, q::Int,
                                     list::Vector{Int}, nz::Int)
    if nz == 1
        # One mask word: no carry, and nothing to do where it selects no bit.
        w = @inbounds list[1]
        M = @inbounds ws[w, 1]
        a = @inbounds z[w, q] & M
        bq = @inbounds x[w, q] & M
        (a | bq) == 0 && return UInt64(0), UInt64(0), UInt64(0)
        c, p = _bit_span_pairs(a, bq, UInt64(0))
        return bq, _bit_top_mask(p) & UInt64(1), c
    end
    _qm_bit_column_untouched(x, z, ws, q, list, nz) && return UInt64(0), UInt64(0), UInt64(0)
    rx = UInt64(0)
    cr = UInt64(0)
    carry = UInt64(0)
    k = 1
    @inbounds while k < nz
        w0 = list[k]
        w1 = list[k + 1]
        M0 = ws[w0, 1]
        M1 = ws[w1, 1]
        a0 = z[w0, q] & M0
        a1 = z[w1, q] & M1
        b0 = x[w0, q] & M0
        b1 = x[w1, q] & M1
        rx ⊻= b0 ⊻ b1
        s0 = _bit_top_mask(_bit_prefix_parity(a0))
        c0, _ = _bit_span_pairs(a0, b0, carry)
        c1, p1 = _bit_span_pairs(a1, b1, carry ⊻ s0)
        cr ⊻= c0 ⊻ c1
        carry ⊻= s0 ⊻ _bit_top_mask(p1)
        k += 2
    end
    @inbounds if k == nz
        w = list[k]
        M = ws[w, 1]
        a = z[w, q] & M
        bq = x[w, q] & M
        rx ⊻= bq
        c, p = _bit_span_pairs(a, bq, carry)
        cr ⊻= c
        carry ⊻= _bit_top_mask(p)
    end
    return rx, carry & UInt64(1), cr
end

# Without the pair count.
@inline function _qm_bit_residual_column(x, z, ws::Matrix{UInt64}, q::Int, nwm::Int)
    rx = UInt64(0)
    rz = UInt64(0)
    @inbounds @simd for w in 1:nwm
        M = ws[w, 1]
        rx ⊻= x[w, q] & M
        rz ⊻= z[w, q] & M
    end
    return rx, rz
end
@inline function _qm_bit_residual_column(x, z, ws::Matrix{UInt64}, q::Int,
                                         list::Vector{Int}, nz::Int)
    _qm_bit_column_untouched(x, z, ws, q, list, nz) && return UInt64(0), UInt64(0)
    rx = UInt64(0)
    rz = UInt64(0)
    @inbounds for k in 1:nz
        w = list[k]
        M = ws[w, 1]
        rx ⊻= x[w, q] & M
        rz ⊻= z[w, q] & M
    end
    return rx, rz
end

# The qudit-major form of `_gm_bit_residual_phase!`. `list` receives the
# nonzero mask words, and only those are visited when there are at most `cut`
# of them (`_bit_span_sparse_cut`; `nwm` for always, -1 for never). Every
# choice gives the same result.
function _qm_bit_residual_phase!(ws::Matrix{UInt64}, b::_QMBits,
                                 P::Union{Nothing,Matrix{UInt64}}, m::Int,
                                 list::Vector{Int}, cut::Int)
    x, z = b.x, b.z
    n = b.n
    nwm = _nwords(m)
    nz = 0
    @inbounds for w in 1:nwm
        if ws[w, 1] != 0
            nz += 1
            list[nz] = w
        end
    end
    sparse = nz <= cut
    cr = UInt64(0)
    bad = UInt64(0)
    @inbounds for wq in 1:_nwords(n)
        q0 = (wq - 1) << _WORD_SHIFT
        rxw = UInt64(0)
        rzw = UInt64(0)
        for o in 0:(min(_WORD_BITS, n - q0) - 1)
            q = q0 + o + 1
            if P === nothing
                rx, rz = sparse ? _qm_bit_residual_column(x, z, ws, q, list, nz) :
                                  _qm_bit_residual_column(x, z, ws, q, nwm)
                rxw |= _bit_parity(rx) << o
                rzw |= _bit_parity(rz) << o
            else
                rx, pz, c = sparse ? _qm_bit_span_column(x, z, ws, q, list, nz) :
                                     _qm_bit_span_column(x, z, ws, q, nwm)
                cr ⊻= c
                rxw |= _bit_parity(rx) << o
                rzw |= pz << o
            end
        end
        rxw ⊻= ws[wq, 3]
        rzw ⊻= ws[wq, 4]
        ws[wq, 3] = rxw
        ws[wq, 4] = rzw
        bad |= rxw | rzw
    end
    kacc = P === nothing ? 0 : (_bit_mask_phase(ws, P, m) + 2 * (count_ones(cr) & 1)) & 3
    return bad == 0, kacc
end

@inline _bit_residual_phase!(tab::_BitTableau, L::GeneratorMajor) =
    _gm_bit_residual_phase!(tab.word_workspace, _stab(tab, L), _bit_phase_planes(tab), tab.m)
@inline _bit_residual_phase!(tab::_BitTableau, L::QuditMajor) =
    _qm_bit_residual_phase!(tab.word_workspace, _stab(tab, L), _bit_phase_planes(tab), tab.m,
                            tab.zacc_workspace, _bit_span_sparse_cut(_nwords(tab.m), tab.storephase))

########################################
# Phase alone, for a pure state        #
########################################

# The parity of `Σ_{i<j} z_i·x_j` over the generators of the mask in column 1
# of `ws`: the pair count of every qudit column; `list` and `cut` as in
# `_qm_bit_residual_phase!`.
function _qm_bit_pair_parity!(ws::Matrix{UInt64}, b::_QMBits, m::Int, list::Vector{Int},
                              cut::Int)
    x, z = b.x, b.z
    nwm = _nwords(m)
    nz = 0
    @inbounds for w in 1:nwm
        if ws[w, 1] != 0
            nz += 1
            list[nz] = w
        end
    end
    sparse = nz <= cut
    cr = UInt64(0)
    for q in 1:b.n
        _, _, c = sparse ? _qm_bit_span_column(x, z, ws, q, list, nz) :
                           _qm_bit_span_column(x, z, ws, q, nwm)
        cr ⊻= c
    end
    return count_ones(cr) & 1
end

# The phase exponent of the product of the generators of the mask in column 1
# of word_workspace, for a qudit-major tableau with phases. A single generator
# is its own phase.
function _qm_bit_mask_product_phase!(tab::_BitTableau, L::QuditMajor)
    ws = tab.word_workspace
    m = tab.m
    count = 0
    @inbounds for w in 1:_nwords(m)
        count += count_ones(ws[w, 1])
    end
    lin = _bit_mask_phase(ws, tab.phase.planes, m)
    count <= 1 && return lin
    pairs = _qm_bit_pair_parity!(ws, _stab(tab, L), m, tab.zacc_workspace,
                                 _bit_span_sparse_cut(_nwords(m), true))
    return (lin + 2 * pairs) & 3
end

# Whether a pure state takes its phase without the residual. Qudit-major the
# residual is a pass over every qudit column; generator-major it comes with
# the phase at the cost of one more word column, so it stays.
@inline _bit_pure_phase_only(::GeneratorMajor) = false
@inline _bit_pure_phase_only(::QuditMajor) = true

########################################
# measure!: commuting branches         #
########################################

@inline _bit_span_needs_solve(tab::_BitStab) = !tab.iscanonical
@inline _bit_span_needs_solve(::_BitDestab) = false

# `_measure_commuting!` on packed storage, with `opc` the commutation operand
# of `op`. A dense `opc` is `res_workspace`, valid on entry.
function _bit_measure_commuting!(rng::Random.AbstractRNG, tab::_BitTableau, L::TableauLayout,
                                 op::AbstractPauli, opc::_BitOperand, kop::Int,
                                 outcome::Union{Int,Nothing})
    n = tab.n
    m = tab.m
    # A pure state's n independent generators span every Pauli that commutes
    # with them, so the outcome is fixed, and `0` by convention without phases.
    m == n && !tab.storephase && return 0

    ws = tab.word_workspace
    _bit_pack_operand!(ws, opc, n)
    if _bit_span_needs_solve(tab)
        # The generator basis stays as it is; the solve overwrites res_workspace.
        _span_coeffs_keep_basis!(tab, L, op)
        _bit_coeffs_to_mask!(ws, tab.c_workspace, m)
    else
        _bit_coeff_mask!(tab, L, opc)
    end
    # A pure state's generators span `op`, which commutes with them, so its
    # coefficients are exact without a residual.
    if m == n && _bit_pure_phase_only(L)
        return _deterministic_outcome(_qm_bit_mask_product_phase!(tab, L), kop, 2)
    end
    in_span, kacc = _bit_residual_phase!(tab, L)

    if in_span
        tab.storephase || return 0
        return _deterministic_outcome(kacc, kop, 2)
    end
    (m < n) || throw(ArgumentError("Cannot append generator: tableau at full capacity (m==n)."))
    t = _random_outcome(rng, outcome, 2)

    newcol = m + 1
    tab.m = newcol
    set_operator!(tab, L, newcol, op)
    if tab.storephase
        _phase(tab)[newcol] = _measured_generator_phase(t, kop, 2)
    end
    _bit_after_append!(tab, L, newcol, opc)

    tab.iscanonical = false
    return t
end

# The first qudit where the residual in columns 3-4 of `ws` is nonzero (0 if
# none), and whether its X bit is set there.
@inline function _bit_residual_site(ws::Matrix{UInt64}, n::Int)
    @inbounds for w in 1:_nwords(n)
        r = ws[w, 3] | ws[w, 4]
        r == 0 && continue
        o = trailing_zeros(r)
        return ((w - 1) << _WORD_SHIFT) + o + 1, (ws[w, 3] >> o) & UInt64(1) != 0
    end
    return 0, false
end

# The pairing parity `⟨v, op⟩` of the vector in columns 3 (X) and 4 (Z) of `ws`
# with the operand.
@inline function _bit_pair_operand(ws::Matrix{UInt64}, op::NPauli{D}, ::Int) where {D}
    c = UInt64(0)
    @inbounds for i in 1:D
        q = op.qudits[i]
        c ⊻= (_ws_bit(ws, q, 3) & (op.zs[i] % UInt64)) ⊻ (_ws_bit(ws, q, 4) & (op.xs[i] % UInt64))
    end
    return (c & UInt64(1)) % Int
end
@inline function _bit_pair_operand(ws::Matrix{UInt64}, op::Vector{Int}, n::Int)
    acc = UInt64(0)
    @inbounds for w in 1:_nwords(n)
        acc ⊻= (ws[w, 3] & _pack_word(op, n, w, n)) ⊻ (ws[w, 4] & _pack_word(op, 0, w, n))
    end
    return count_ones(acc) & 1
end

@inline _bit_after_append!(::_BitStab, ::TableauLayout, ::Int, ::_BitOperand) = nothing

# The dual of the appended generator `newcol`, which holds the operand, and
# the old duals made orthogonal to it. The residual is in columns 3-4 of
# word_workspace. The new dual is `v = e ⊻ ⊕_j t_j D_j` over the old duals,
# with `e` the unit vector on the residual's first qudit `q` (Z_q where the
# residual has an X there, else X_q) and `t_j = ⟨e, S_j⟩`, one bit of
# generator `j`.
function _bit_after_append!(tab::_BitDestab, L::GeneratorMajor, newcol::Int, op::_BitOperand)
    ws = tab.word_workspace
    n = tab.n
    m = newcol - 1
    stab = _stab(tab, L)
    destab = _destab(tab, L)
    dx, dz = destab.x, destab.z
    nw = size(dx, 1)

    q, zq = _bit_residual_site(ws, n)
    q == 0 && throw(ArgumentError("Failed to construct destabilizer: residual is zero."))
    @inbounds for w in 1:nw
        ws[w, 3] = UInt64(0)
        ws[w, 4] = UInt64(0)
    end
    @inbounds ws[_word_index(q), zq ? 4 : 3] = UInt64(1) << _bit_offset(q)
    @inbounds for j in 1:m
        t = zq ? _xget(stab, q, j) : _zget(stab, q, j)
        t == 0 && continue
        @simd ivdep for w in 1:nw
            ws[w, 3] ⊻= dx[w, j]
            ws[w, 4] ⊻= dz[w, j]
        end
    end
    _bit_pair_operand(ws, op, n) == 0 &&
        throw(ArgumentError("Failed to construct destabilizer: β = 0."))
    @inbounds @simd ivdep for w in 1:nw
        dx[w, newcol] = ws[w, 3]
        dz[w, newcol] = ws[w, 4]
    end

    supp = tab.support_workspace
    nsupp = _bit_operand_support!(supp, op)
    _orthogonalize_duals!(tab, L, newcol, m, supp, nsupp, tab.zacc_workspace, 0, false)
    tab.xdotz_cache[newcol] = _bit_operand_xz(op)
    return nothing
end

# Qudit-major, the pairings `t` over the old generators are the words of
# column `q` of the generators (column 1 of word_workspace), and each bit of
# `v` is the parity of a dual column under that mask.
function _bit_after_append!(tab::_BitDestab, L::QuditMajor, newcol::Int, op::_BitOperand)
    ws = tab.word_workspace
    n = tab.n
    m = newcol - 1
    stab = _stab(tab, L)
    destab = _destab(tab, L)
    dx, dz = destab.x, destab.z
    nwm = _nwords(m)

    q, zq = _bit_residual_site(ws, n)
    q == 0 && throw(ArgumentError("Failed to construct destabilizer: residual is zero."))
    T = zq ? stab.x : stab.z
    @inbounds for w in 1:nwm
        ws[w, 1] = T[w, q]
    end
    nwm == 0 || (@inbounds ws[nwm, 1] &= _tail_mask(m))
    @inbounds for wq in 1:_nwords(n)
        k0 = (wq - 1) << _WORD_SHIFT
        vxw = UInt64(0)
        vzw = UInt64(0)
        for o in 0:(min(_WORD_BITS, n - k0) - 1)
            vx, vz = _qm_bit_residual_column(dx, dz, ws, k0 + o + 1, nwm)
            vxw |= _bit_parity(vx) << o
            vzw |= _bit_parity(vz) << o
        end
        ws[wq, 3] = vxw
        ws[wq, 4] = vzw
    end
    @inbounds ws[_word_index(q), zq ? 4 : 3] ⊻= UInt64(1) << _bit_offset(q)
    _bit_pair_operand(ws, op, n) == 0 &&
        throw(ArgumentError("Failed to construct destabilizer: β = 0."))

    sites = tab.zacc_workspace
    ns = 0
    wn, on = _word_index(newcol), _bit_offset(newcol)
    clr = ~(UInt64(1) << on)
    @inbounds for k in 1:n
        vx = _ws_bit(ws, k, 3)
        vz = _ws_bit(ws, k, 4)
        dx[wn, k] = (dx[wn, k] & clr) | (vx << on)
        dz[wn, k] = (dz[wn, k] & clr) | (vz << on)
        if (vx | vz) != 0
            ns += 1
            sites[ns] = k
        end
    end

    supp = tab.support_workspace
    nsupp = _bit_operand_support!(supp, op)
    _orthogonalize_duals!(tab, L, newcol, m, supp, nsupp, sites, ns, true)
    tab.xdotz_cache[newcol] = _bit_operand_xz(op)
    return nothing
end

########################################
# expect_int!                          #
########################################

# Whether the operand commutes with every active generator.
@inline function _bit_commutes_with_generators(tab::_BitTableau, L::GeneratorMajor,
                                               op::_BitOperand)
    ws = tab.word_workspace
    _gm_bit_pack_operand!(ws, op, tab.n)
    return _gm_bit_first(_stab(tab, L), op, ws, 1, tab.m) == 0
end
@inline _bit_commutes_with_generators(tab::_BitTableau, L::QuditMajor, op::_BitOperand) =
    _qm_bit_mask!(tab.word_workspace, _stab(tab, L), op, tab.m) == 0

@inline _bit_canonical_for_span!(tab::_BitStab, L::TableauLayout) =
    (tab.iscanonical || _canonicalize!(tab, L); nothing)
@inline _bit_canonical_for_span!(::_BitDestab, ::TableauLayout) = nothing

function _expect_int!(tab::_BitTableau, L::TableauLayout, op::AbstractPauli)
    _check_pauli_support(tab, op)
    pure = tab.m == tab.n
    if pure && !tab.storephase
        return _bit_commutes_with_generators(tab, L, _commutation_operand!(tab, op)) ? 0 : -1
    end

    # Canonicalizing uses word_workspace, so it comes before the operand.
    _bit_canonical_for_span!(tab, L)
    opc = _commutation_operand!(tab, op)
    ws = tab.word_workspace
    if pure && _bit_pure_phase_only(L)
        # The span of a pure state is what commutes with its generators, and
        # the coefficients of such an operand are exact without a residual.
        _bit_commutes_with_generators(tab, L, opc) || return -1
        _bit_coeff_mask!(tab, L, opc)
        return mod(mod(op.phase, 4) - _qm_bit_mask_product_phase!(tab, L), 4)
    end
    _bit_pack_operand!(ws, opc, tab.n)
    _bit_coeff_mask!(tab, L, opc)
    in_span, kacc = _bit_residual_phase!(tab, L)
    in_span || return -1
    tab.storephase || return 0
    return mod(mod(op.phase, 4) - kacc, 4)
end
