########################################
# Packed qubit measurement kernels     #
########################################
#
# The random branch of `measure!` and `dephase!` on packed storage. At d = 2
# every commutator is a parity, every anticommuting generator absorbs the
# pivot generator `g_p` once (`g_i ⊻= g_p`), and its phase gains
# `kws + 2·(x_p·z_i)` mod 4, `kws` the phase of `g_p`. A `DestabilizerTableau`
# patches its cache by `x_p·z_i ⊻ x_i·z_p ⊻ x_p·z_p`, which equals the parity of
# x·z recomputed after the update. These are the formulas of
# `_absorb_pivot_scalars!` at `a = 1`, so the results equal the unpacked ones
# bit for bit.
#
# - Generator-major, the pivot is copied into columns 1 (X) and 2 (Z) of
#   `word_workspace`, and each later anticommuting generator takes one pass
#   over its words: the cross-term parities and the XOR of the pivot words.
#   A dense operand is packed into columns 3 (X) and 4 (Z), so a commutator is
#   the `count_ones` parity of word ANDs, and the measured generator, its x·z
#   parity and its support are read from there.
# - Qudit-major, column 1 of `word_workspace` holds the commutator mask over
#   generators, the XOR of the operand's support columns. The pivot is its
#   lowest set bit. The pivot generator is gathered along qudits into columns
#   3 (X) and 4 (Z), with its support listed in `zacc_workspace`, and every
#   anticommuting generator is updated 64 at a time: the phase planes from the
#   mask and the parity words `⊻_{q: x_p[q] = 1} z[:, q]` (column 2), the cache
#   plane likewise, and the XOR of the mask into every pivot-support column.
#   When few mask words are nonzero, only those words are visited, listed in
#   `c_workspace`.
#
# Exactly one outcome is drawn per random measurement, through the same
# `_random_outcome` call as the unpacked kernels, and none when the operand
# commutes with every generator; the commuting branches are in
# src/span_decomposition_bits.jl, which lists what they keep in
# `word_workspace`.

# The commutation operand of `_commutation_operand!`: reduced sparse exponents,
# or a dense exponent vector `[x; z]` in `res_workspace`.
const _BitOperand = Union{NPauli, Vector{Int}}

# Add `l + 2h` mod 4, one bit of each per generator, to word `w` of the phase
# planes `P`.
@inline function _bit_phase_add!(P::Matrix{UInt64}, w::Int, l::UInt64, h::UInt64)
    @inbounds begin
        p0 = P[w, 1]
        P[w, 1] = p0 ⊻ l
        P[w, 2] ⊻= h ⊻ (p0 & l)
    end
    return nothing
end

@inline _bit_phase_planes(tab::_BitTableau) = tab.storephase ? tab.phase.planes : nothing

# The bit of qudit `k` in word column `c` of `ws`, as a `UInt64` 0 or 1.
@inline _ws_bit(ws::Matrix{UInt64}, k::Int, c::Int) =
    @inbounds (ws[_word_index(k), c] >> _bit_offset(k)) & UInt64(1)

# The parity of the operand's x·z.
@inline function _bit_operand_xz(op::NPauli{D}) where {D}
    s = 0
    @inbounds for i in 1:D
        s ⊻= (op.xs[i] & op.zs[i]) & 1
    end
    return s
end
@inline function _bit_operand_xz(op::Vector{Int})
    n = length(op) ÷ 2
    s = 0
    @inbounds for q in 1:n
        s ⊻= (op[q] & op[n + q]) & 1
    end
    return s
end

# The qudits where the operand has an odd exponent, into `supp`; returns how
# many. This is the support of a generator overwritten by the operand.
@inline function _bit_operand_support!(supp::Vector{Int}, op::NPauli{D}) where {D}
    k = 0
    @inbounds for i in 1:D
        if isodd(op.xs[i]) || isodd(op.zs[i])
            k += 1
            supp[k] = op.qudits[i]
        end
    end
    return k
end
@inline function _bit_operand_support!(supp::Vector{Int}, op::Vector{Int})
    n = length(op) ÷ 2
    k = 0
    @inbounds for q in 1:n
        if isodd(op[q]) || isodd(op[n + q])
            k += 1
            supp[k] = q
        end
    end
    return k
end

# Set the odd exponents of the operand in generator `j` of `b`, which is zero.
@inline function _bit_set_operand!(b::BitBlock, j::Int, op::NPauli{D}) where {D}
    @inbounds for i in 1:D
        q = op.qudits[i]
        isodd(op.xs[i]) && _xflip!(b, UInt64(1), q, j)
        isodd(op.zs[i]) && _zflip!(b, UInt64(1), q, j)
    end
    return b
end
@inline _bit_set_operand!(b::BitBlock, j::Int, op::Vector{Int}) =
    _set_bit_exponents!(b, j, op, length(op) ÷ 2)

# The phase of the measured generator and, for a `DestabilizerTableau`, its
# cache entry `xz`, the parity of the operand's x·z.
@inline function _bit_measured_scalars!(tab::_BitTableau, pivot::Int, t::Int, kop::Int, xz::Int)
    tab.storephase && (_phase(tab)[pivot] = _measured_generator_phase(t, kop, 2))
    tab isa DestabilizerTableau && (tab.xdotz_cache[pivot] = xz)
    return nothing
end

########################################
# Generator-major                      #
########################################

# A dense operand packed into columns 3 (X) and 4 (Z) of `ws`; a sparse one is
# read as it is.
@inline _gm_bit_pack_operand!(::Matrix{UInt64}, ::NPauli, ::Int) = nothing
@inline function _gm_bit_pack_operand!(ws::Matrix{UInt64}, op::Vector{Int}, n::Int)
    @inbounds for w in 1:_nwords(n)
        ws[w, 3] = _pack_word(op, 0, w, n)
        ws[w, 4] = _pack_word(op, n, w, n)
    end
    return nothing
end

# Overwrite generator `pivot` with the operand and return the parity of its
# x·z. A dense operand is copied from its packed form in columns 3-4 of `ws`.
@inline function _gm_bit_set_pivot!(b::_GMBits, pivot::Int, op::NPauli, ::Matrix{UInt64})
    _zero_gen!(b, pivot)
    _bit_set_operand!(b, pivot, op)
    return _bit_operand_xz(op)
end
@inline function _gm_bit_set_pivot!(b::_GMBits, pivot::Int, ::Vector{Int}, ws::Matrix{UInt64})
    x, z = b.x, b.z
    s = UInt64(0)
    @inbounds @simd for w in axes(x, 1)
        xw = ws[w, 3]
        zw = ws[w, 4]
        x[w, pivot] = xw
        z[w, pivot] = zw
        s ⊻= xw & zw
    end
    return count_ones(s) & 1
end

# The qudits of the operand with an odd exponent, into `supp`; returns how
# many. A dense operand is read from its packed form in columns 3-4 of `ws`.
@inline _gm_bit_operand_support!(supp::Vector{Int}, op::NPauli, ::Matrix{UInt64}, ::Int) =
    _bit_operand_support!(supp, op)
@inline function _gm_bit_operand_support!(supp::Vector{Int}, ::Vector{Int}, ws::Matrix{UInt64}, n::Int)
    k = 0
    @inbounds for w in 1:_nwords(n)
        u = ws[w, 3] | ws[w, 4]
        while u != 0
            k += 1
            supp[k] = (w - 1) << _WORD_SHIFT + trailing_zeros(u) + 1
            u &= u - UInt64(1)
        end
    end
    return k
end

# The commutator parity of generator `j` with the operand, 0 or 1.
@inline function _gm_bit_comm(b::_GMBits, j::Int, op::NPauli{D}, ::Matrix{UInt64}) where {D}
    c = 0
    @inbounds for i in 1:D
        q = op.qudits[i]
        isodd(op.zs[i]) && (c ⊻= _xget(b, q, j))
        isodd(op.xs[i]) && (c ⊻= _zget(b, q, j))
    end
    return c
end
@inline function _gm_bit_comm(b::_GMBits, j::Int, ::Vector{Int}, ws::Matrix{UInt64})
    x, z = b.x, b.z
    acc = UInt64(0)
    @inbounds @simd for w in axes(x, 1)
        acc ⊻= (x[w, j] & ws[w, 4]) ⊻ (z[w, j] & ws[w, 3])
    end
    return count_ones(acc) & 1
end

# The first generator in `lo:m` that anticommutes with the operand, or 0.
@inline function _gm_bit_first(b::_GMBits, op::_BitOperand, ws::Matrix{UInt64}, lo::Int, m::Int)
    for j in lo:m
        _gm_bit_comm(b, j, op, ws) == 0 || return j
    end
    return 0
end

# Copy generator `pivot` into columns 1 (X) and 2 (Z) of `ws`, then let every
# later generator of `1:m` that anticommutes with the operand absorb it, with
# the phase planes `P` and the cache plane `C` (`nothing` when absent).
# Generator `pivot` itself is left as it is.
function _gm_bit_absorb!(b::_GMBits, P::Union{Nothing,Matrix{UInt64}},
                         C::Union{Nothing,Matrix{UInt64}}, ws::Matrix{UInt64},
                         op::_BitOperand, pivot::Int, m::Int, kws::Int)
    x, z = b.x, b.z
    nw = size(x, 1)
    xzp = UInt64(0)
    @inbounds @simd for w in 1:nw
        px = x[w, pivot]
        pz = z[w, pivot]
        ws[w, 1] = px
        ws[w, 2] = pz
        xzp ⊻= px & pz
    end
    xzp = (count_ones(xzp) % UInt64) & UInt64(1)
    kl = (kws % UInt64) & UInt64(1)
    kh = ((kws >> 1) % UInt64) & UInt64(1)
    for i in (pivot + 1):m
        _gm_bit_comm(b, i, op, ws) == 0 && continue
        cb = UInt64(0)
        cz = UInt64(0)
        @inbounds @simd for w in 1:nw
            xi = x[w, i]
            zi = z[w, i]
            px = ws[w, 1]
            pz = ws[w, 2]
            cb ⊻= px & zi
            cz ⊻= xi & pz
            x[w, i] = xi ⊻ px
            z[w, i] = zi ⊻ pz
        end
        wi, o = _word_index(i), _bit_offset(i)
        pb = (count_ones(cb) % UInt64) & UInt64(1)
        P === nothing || _bit_phase_add!(P, wi, kl << o, (kh ⊻ pb) << o)
        if C !== nothing
            zb = (count_ones(cz) % UInt64) & UInt64(1)
            @inbounds C[wi, 1] ⊻= (pb ⊻ zb ⊻ xzp) << o
        end
    end
    return nothing
end

function _measure!(rng::Random.AbstractRNG, tab::_BitTableau, L::GeneratorMajor,
                   op::AbstractPauli, outcome::Union{Int,Nothing}, phase_policy::Int)
    kop = _measurement_phase(tab, op, outcome, phase_policy)
    m = tab.m
    # A dense operand lives in res_workspace, which nothing below writes before
    # the commuting branch.
    opc = _commutation_operand!(tab, op)
    stab = _stab(tab, L)
    ws = tab.word_workspace
    _gm_bit_pack_operand!(ws, opc, tab.n)
    pivot = _gm_bit_first(stab, opc, ws, 1, m)
    pivot == 0 && return _bit_measure_commuting!(rng, tab, L, op, opc, kop, outcome)

    # The outcome is random; draw it before anything in tab changes.
    t = _random_outcome(rng, outcome, 2)
    kws = tab.storephase ? _phase(tab)[pivot] : 0
    _gm_bit_absorb!(stab, _bit_phase_planes(tab), _bit_cache(tab), ws, opc, pivot, m, kws)
    xz = _gm_bit_set_pivot!(stab, pivot, opc, ws)
    _bit_measured_scalars!(tab, pivot, t, kop, xz)
    _after_bit_noncommuting_measurement!(tab, L, pivot, m, opc, 0)
    tab.iscanonical = false
    return t
end

function _dephase!(tab::_BitTableau, L::GeneratorMajor, op::AbstractPauli)
    _check_pauli_support(tab, op)
    m = tab.m
    opc = _commutation_operand!(tab, op)
    stab = _stab(tab, L)
    ws = tab.word_workspace
    _gm_bit_pack_operand!(ws, opc, tab.n)
    pivot = _gm_bit_first(stab, opc, ws, 1, m)
    pivot == 0 && return tab
    kws = tab.storephase ? _phase(tab)[pivot] : 0
    _gm_bit_absorb!(stab, _bit_phase_planes(tab), _bit_cache(tab), ws, opc, pivot, m, kws)
    _drop_generator!(tab, L, pivot)
    tab.iscanonical = false
    return tab
end

# The destabilizers after the random branch: the old pivot generator (columns
# 1-2 of word_workspace) becomes the pivot dual, and every other dual is made
# orthogonal to the measured generator. `np` is not read.
function _after_bit_noncommuting_measurement!(tab::_BitDestab, L::GeneratorMajor, pivot::Int,
                                              m::Int, op::_BitOperand, ::Int)
    destab = _destab(tab, L)
    dx, dz = destab.x, destab.z
    ws = tab.word_workspace
    @inbounds @simd for w in axes(dx, 1)
        dx[w, pivot] = ws[w, 1]
        dz[w, pivot] = ws[w, 2]
    end
    supp = tab.support_workspace
    nsupp = _gm_bit_operand_support!(supp, op, ws, tab.n)
    _orthogonalize_duals!(tab, L, pivot, m, supp, nsupp, tab.zacc_workspace, 0, false)
    return nothing
end

# D_j ⊻= D_col for every dual `j ≠ col` of `1:m` whose pairing with generator
# `col` is odd. A pairing is read from the bits on `supp[1:nsupp]` when that
# is shorter than a column, else as the parity of word ANDs. `sites` is not
# read: each update is a word XOR of whole columns.
function _orthogonalize_duals!(tab::_BitDestab, L::GeneratorMajor, col::Int, m::Int,
                               supp::Vector{Int}, nsupp::Int, ::Vector{Int}, ::Int, ::Bool)
    stab = _stab(tab, L)
    destab = _destab(tab, L)
    sx, sz = stab.x, stab.z
    dx, dz = destab.x, destab.z
    nw = size(dx, 1)
    bybits = nsupp <= nw
    @inbounds for j in 1:m
        j == col && continue
        if bybits
            tb = 0
            for k in 1:nsupp
                q = supp[k]
                tb ⊻= (_zget(stab, q, col) & _xget(destab, q, j)) ⊻
                      (_xget(stab, q, col) & _zget(destab, q, j))
            end
        else
            acc = UInt64(0)
            @simd for w in 1:nw
                acc ⊻= (dx[w, j] & sz[w, col]) ⊻ (dz[w, j] & sx[w, col])
            end
            tb = count_ones(acc) & 1
        end
        tb == 0 && continue
        @simd for w in 1:nw
            dx[w, j] ⊻= dx[w, col]
            dz[w, j] ⊻= dz[w, col]
        end
    end
    return nothing
end

@inline _after_bit_noncommuting_measurement!(::_BitStab, ::TableauLayout, ::Int, ::Int,
                                             ::_BitOperand, ::Int) = nothing

########################################
# Qudit-major                          #
########################################

# Visit only the nonzero mask words when at most `(nw - 1) ÷ _BIT_QM_SPARSE_DIV`
# of the `nw` words of generators `1:m` are nonzero, i.e. when
# `4·(nonzero words) < nw`. Random-branch Z measurements, n = 1024 and 4096,
# both tableau types (Apple silicon): on brickwork states of depth 2-32 this
# cutoff is up to 1.10-1.16 times faster than always visiting every word
# (depth 32) and within 5% of always visiting only the nonzero ones; on random
# states it takes the dense path, while always-sparse is 1.8-4.3 times slower.
const _BIT_QM_SPARSE_DIV = 4

# The commutator mask of generators `1:m` with the operand into column 1 of
# `ws`, and the pivot, its lowest set bit (0 when the mask is zero).
function _qm_bit_mask!(ws::Matrix{UInt64}, b::_QMBits, op::NPauli{D}, m::Int) where {D}
    x, z = b.x, b.z
    nwm = _nwords(m)
    @inbounds for w in 1:nwm
        ws[w, 1] = UInt64(0)
    end
    @inbounds for i in 1:D
        q = op.qudits[i]
        if isodd(op.zs[i])
            @simd ivdep for w in 1:nwm
                ws[w, 1] ⊻= x[w, q]
            end
        end
        if isodd(op.xs[i])
            @simd ivdep for w in 1:nwm
                ws[w, 1] ⊻= z[w, q]
            end
        end
    end
    return _qm_bit_pivot(ws, m)
end
function _qm_bit_mask!(ws::Matrix{UInt64}, b::_QMBits, op::Vector{Int}, m::Int)
    x, z = b.x, b.z
    n = b.n
    nwm = _nwords(m)
    @inbounds for w in 1:nwm
        ws[w, 1] = UInt64(0)
    end
    @inbounds for q in 1:n
        if isodd(op[n + q])
            @simd ivdep for w in 1:nwm
                ws[w, 1] ⊻= x[w, q]
            end
        end
        if isodd(op[q])
            @simd ivdep for w in 1:nwm
                ws[w, 1] ⊻= z[w, q]
            end
        end
    end
    return _qm_bit_pivot(ws, m)
end

@inline function _qm_bit_pivot(ws::Matrix{UInt64}, m::Int)
    nwm = _nwords(m)
    nwm == 0 && return 0
    @inbounds ws[nwm, 1] &= _tail_mask(m)
    @inbounds for w in 1:nwm
        M = ws[w, 1]
        M == 0 || return ((w - 1) << _WORD_SHIFT) + trailing_zeros(M) + 1
    end
    return 0
end

# Every generator of `1:m` other than `pivot` whose bit is set in the mask
# (column 1 of word_workspace) absorbs generator `pivot`, which is left as it
# is. The pivot is gathered along qudits into columns 3 (X) and 4 (Z), and its
# support into zacc_workspace; returns the size of that support. `div` sets
# the sparse cutoff (`_BIT_QM_SPARSE_DIV`, 0 for always sparse); every choice
# gives the same result.
function _qm_bit_absorb!(tab::_BitTableau, b::_QMBits, pivot::Int, m::Int, div::Int)
    ws = tab.word_workspace
    sites = tab.zacc_workspace
    n = b.n
    x, z = b.x, b.z
    nwm = _nwords(m)
    wp, bp = _word_index(pivot), _bit_offset(pivot)
    @inbounds ws[wp, 1] &= ~(UInt64(1) << bp)

    @inbounds for w in 1:_nwords(n)
        ws[w, 3] = UInt64(0)
        ws[w, 4] = UInt64(0)
    end
    np = 0
    @inbounds for k in 1:n
        xb = (x[wp, k] >> bp) & UInt64(1)
        zb = (z[wp, k] >> bp) & UInt64(1)
        if (xb | zb) != 0
            np += 1
            sites[np] = k
            wk, ok = _word_index(k), _bit_offset(k)
            ws[wk, 3] |= xb << ok
            ws[wk, 4] |= zb << ok
        end
    end

    P = _bit_phase_planes(tab)
    C = _bit_cache(tab)
    kws = tab.storephase ? _phase(tab)[pivot] : 0
    m0 = _bit_word(kws)
    m1 = _bit_word(kws >> 1)
    xzm = UInt64(0)
    if C !== nothing
        s = UInt64(0)
        @inbounds for w in 1:_nwords(n)
            s ⊻= ws[w, 3] & ws[w, 4]
        end
        xzm = _bit_word(count_ones(s))
    end

    mw = tab.c_workspace
    nz = 0
    @inbounds for w in 1:nwm
        if ws[w, 1] != 0
            nz += 1
            mw[nz] = w
        end
    end
    nz == 0 && return np

    if div == 0 || nz <= (nwm - 1) ÷ div
        @inbounds for kk in 1:nz
            w = mw[kk]
            M = ws[w, 1]
            pr = UInt64(0)
            pz = UInt64(0)
            for s in 1:np
                k = sites[s]
                _ws_bit(ws, k, 3) == 0 || (pr ⊻= z[w, k])
                _ws_bit(ws, k, 4) == 0 || (pz ⊻= x[w, k])
            end
            P === nothing || _bit_phase_add!(P, w, M & m0, M & (m1 ⊻ pr))
            C === nothing || (C[w, 1] ⊻= M & (pr ⊻ pz ⊻ xzm))
        end
        @inbounds for s in 1:np
            k = sites[s]
            if _ws_bit(ws, k, 3) != 0
                for kk in 1:nz
                    w = mw[kk]
                    x[w, k] ⊻= ws[w, 1]
                end
            end
            if _ws_bit(ws, k, 4) != 0
                for kk in 1:nz
                    w = mw[kk]
                    z[w, k] ⊻= ws[w, 1]
                end
            end
        end
        return np
    end

    # The parity words `⊻_{q: x_p[q] = 1} z[:, q]` in column 2, read before the
    # rank-1 update below changes the columns.
    if P !== nothing || C !== nothing
        @inbounds for w in 1:nwm
            ws[w, 2] = UInt64(0)
        end
        @inbounds for s in 1:np
            k = sites[s]
            _ws_bit(ws, k, 3) == 0 && continue
            @simd ivdep for w in 1:nwm
                ws[w, 2] ⊻= z[w, k]
            end
        end
    end
    if C !== nothing
        @inbounds for s in 1:np
            k = sites[s]
            _ws_bit(ws, k, 4) == 0 && continue
            @simd ivdep for w in 1:nwm
                C[w, 1] ⊻= ws[w, 1] & x[w, k]
            end
        end
        @inbounds @simd ivdep for w in 1:nwm
            C[w, 1] ⊻= ws[w, 1] & (ws[w, 2] ⊻ xzm)
        end
    end
    if P !== nothing
        @inbounds @simd ivdep for w in 1:nwm
            M = ws[w, 1]
            _bit_phase_add!(P, w, M & m0, M & (m1 ⊻ ws[w, 2]))
        end
    end
    @inbounds for s in 1:np
        k = sites[s]
        if _ws_bit(ws, k, 3) != 0
            @simd ivdep for w in 1:nwm
                x[w, k] ⊻= ws[w, 1]
            end
        end
        if _ws_bit(ws, k, 4) != 0
            @simd ivdep for w in 1:nwm
                z[w, k] ⊻= ws[w, 1]
            end
        end
    end
    return np
end

function _measure!(rng::Random.AbstractRNG, tab::_BitTableau, L::QuditMajor,
                   op::AbstractPauli, outcome::Union{Int,Nothing}, phase_policy::Int)
    return _measure_bits_qm!(rng, tab, L, op, outcome, phase_policy, _BIT_QM_SPARSE_DIV)
end

# `measure!` on a packed qudit-major tableau. `div` sets the sparse cutoff of
# `_qm_bit_absorb!`; every choice gives the same result.
function _measure_bits_qm!(rng::Random.AbstractRNG, tab::_BitTableau, L::QuditMajor,
                           op::AbstractPauli, outcome::Union{Int,Nothing}, phase_policy::Int,
                           div::Int)
    kop = _measurement_phase(tab, op, outcome, phase_policy)
    m = tab.m
    # A dense operand lives in res_workspace, which nothing below writes before
    # the commuting branch.
    opc = _commutation_operand!(tab, op)
    stab = _stab(tab, L)
    pivot = _qm_bit_mask!(tab.word_workspace, stab, opc, m)
    pivot == 0 && return _bit_measure_commuting!(rng, tab, L, op, opc, kop, outcome)

    # The outcome is random; draw it before anything in tab changes.
    t = _random_outcome(rng, outcome, 2)
    np = _qm_bit_absorb!(tab, stab, pivot, m, div)

    # The pivot generator vanishes outside its support, so clearing it there
    # zeroes it before the operand is written.
    x, z = stab.x, stab.z
    sites = tab.zacc_workspace
    wp, clr = _word_index(pivot), ~(UInt64(1) << _bit_offset(pivot))
    @inbounds for s in 1:np
        k = sites[s]
        x[wp, k] &= clr
        z[wp, k] &= clr
    end
    _bit_set_operand!(stab, pivot, opc)
    _bit_measured_scalars!(tab, pivot, t, kop, _bit_operand_xz(opc))
    _after_bit_noncommuting_measurement!(tab, L, pivot, m, opc, np)
    tab.iscanonical = false
    return t
end

function _dephase!(tab::_BitTableau, L::QuditMajor, op::AbstractPauli)
    _check_pauli_support(tab, op)
    m = tab.m
    opc = _commutation_operand!(tab, op)
    stab = _stab(tab, L)
    pivot = _qm_bit_mask!(tab.word_workspace, stab, opc, m)
    pivot == 0 && return tab
    _qm_bit_absorb!(tab, stab, pivot, m, _BIT_QM_SPARSE_DIV)
    _drop_generator!(tab, L, pivot)
    tab.iscanonical = false
    return tab
end

# The destabilizers after the random branch: the old pivot generator (columns
# 3-4 of word_workspace, support `zacc_workspace[1:np]`) becomes the pivot
# dual, and every other dual is made orthogonal to the measured generator.
function _after_bit_noncommuting_measurement!(tab::_BitDestab, L::QuditMajor, pivot::Int, m::Int,
                                              op::_BitOperand, np::Int)
    destab = _destab(tab, L)
    ws = tab.word_workspace
    sites = tab.zacc_workspace
    _zero_gen!(destab, pivot)
    @inbounds for s in 1:np
        k = sites[s]
        _xflip!(destab, _ws_bit(ws, k, 3), k, pivot)
        _zflip!(destab, _ws_bit(ws, k, 4), k, pivot)
    end
    supp = tab.support_workspace
    nsupp = _bit_operand_support!(supp, op)
    _orthogonalize_duals!(tab, L, pivot, m, supp, nsupp, sites, np, true)
    return nothing
end

# D_j ⊻= D_col for every dual `j ≠ col` of `1:m` whose pairing with generator
# `col` is odd, 64 duals at a time: the pairing mask over duals (column 2 of
# word_workspace) is the XOR of the dual columns on the support `supp[1:nsupp]`
# of generator `col`, and it is XORed into every column of `sites[1:nsites]`,
# the support of dual `col`, where that dual is 1.
function _orthogonalize_duals!(tab::_BitDestab, L::QuditMajor, col::Int, m::Int,
                               supp::Vector{Int}, nsupp::Int, sites::Vector{Int}, nsites::Int,
                               ::Bool)
    stab = _stab(tab, L)
    destab = _destab(tab, L)
    dx, dz = destab.x, destab.z
    ws = tab.word_workspace
    nwm = _nwords(m)
    nwm == 0 && return nothing
    @inbounds for w in 1:nwm
        ws[w, 2] = UInt64(0)
    end
    @inbounds for s in 1:nsupp
        q = supp[s]
        if _zget(stab, q, col) != 0
            @simd ivdep for w in 1:nwm
                ws[w, 2] ⊻= dx[w, q]
            end
        end
        if _xget(stab, q, col) != 0
            @simd ivdep for w in 1:nwm
                ws[w, 2] ⊻= dz[w, q]
            end
        end
    end
    @inbounds ws[nwm, 2] &= _tail_mask(m)
    col <= m && (@inbounds ws[_word_index(col), 2] &= ~(UInt64(1) << _bit_offset(col)))
    @inbounds for s in 1:nsites
        q = sites[s]
        if _xget(destab, q, col) != 0
            @simd ivdep for w in 1:nwm
                dx[w, q] ⊻= ws[w, 2]
            end
        end
        if _zget(destab, q, col) != 0
            @simd ivdep for w in 1:nwm
                dz[w, q] ⊻= ws[w, 2]
            end
        end
    end
    return nothing
end
