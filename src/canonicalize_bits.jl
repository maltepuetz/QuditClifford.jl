########################################
# Packed qubit canonicalization        #
########################################
#
# `canonicalize!` on packed storage, with the pivot order and results of
# `_canonicalize_tableau!` at d = 2: every pivot is 1, eliminating generator
# `j` is `g_j ⊻= g_c`, and its phase gains `k_c + 2·(x_c·z_j)` mod 4
# (`_eliminated_phase`), with `z_j` read before the update. A
# `DestabilizerTableau` adds every eliminated dual into the pivot dual,
# `D_c ⊻= D_j`. The x·z cache is the parity of each generator's x·z after the
# last pivot.
#
# - Qudit-major, combined row `r` is the word column of one qudit over the
#   generators. Its lowest set bit at or after `c` is the pivot; after the swap
#   the column, without bit `c`, is the mask β of the generators to eliminate
#   (column 1 of `word_workspace`). The pivot generator is gathered along
#   qudits into columns 3 (X) and 4 (Z), its support listed in
#   `zacc_workspace`. Then, 64 generators per word: the phase planes from β and
#   the parity words `⊻_{q: x_c[q] = 1} z[:, q]` (column 2), the rank-1 update
#   that XORs β into every pivot-support column, and for a
#   `DestabilizerTableau` bit `c` of every dual column flipped by the parity
#   of `β & D[:, q]`. When few β words are nonzero, only those words are
#   visited, listed in `c_workspace`.
# - Generator-major, row `r` is one bit of each generator's word column. Each
#   eliminated generator takes one pass over its words, which XORs the pivot
#   in and accumulates the parity of `x_c & z_j`; the phase planes change once
#   after the pass. Duals are XORed word by word. Against the generic
#   algorithm on the same packed blocks, which reads the pivot row bit by bit
#   and takes separate passes for the cross term and the update, this took
#   0.63-0.80x the time (M3 Pro, n = 2048 and 4096, random and monitored
#   states, both tableau types).

# Visit only the nonzero β words when at most `(nw - 1) ÷ _BIT_QM_CANON_SPARSE_DIV`
# of the `nw` words of generators `1:m` are nonzero.
const _BIT_QM_CANON_SPARSE_DIV = 4

_canonicalize!(tab::_BitStab, L::QuditMajor) =
    _canonicalize_bits_qm!(tab, L, _BIT_QM_CANON_SPARSE_DIV)
_canonicalize!(tab::_BitDestab, L::QuditMajor) =
    _canonicalize_bits_qm!(tab, L, _BIT_QM_CANON_SPARSE_DIV)
_canonicalize!(tab::_BitStab, L::GeneratorMajor) = _canonicalize_gm_bits!(tab, L)
_canonicalize!(tab::_BitDestab, L::GeneratorMajor) = _canonicalize_gm_bits!(tab, L)

@inline _bit_canon_destab(::_BitStab, ::TableauLayout) = nothing
@inline _bit_canon_destab(tab::_BitDestab, L::TableauLayout) = _destab(tab, L)

# Exchange bits `i` and `j` of every plane of `P`.
@inline _bit_plane_swap!(::Nothing, ::Int, ::Int) = nothing
@inline function _bit_plane_swap!(P::Matrix{UInt64}, i::Int, j::Int)
    wi, oi = _word_index(i), _bit_offset(i)
    wj, oj = _word_index(j), _bit_offset(j)
    @inbounds for b in axes(P, 2)
        t = ((P[wi, b] >> oi) ⊻ (P[wj, b] >> oj)) & UInt64(1)
        P[wi, b] ⊻= t << oi
        P[wj, b] ⊻= t << oj
    end
    return nothing
end

# Bit `j` of plane `b` of `P`, as a `UInt64` 0 or 1; 0 without planes.
@inline _bit_plane_bit(::Nothing, ::Int, ::Int) = UInt64(0)
@inline _bit_plane_bit(P::Matrix{UInt64}, j::Int, b::Int) =
    @inbounds (P[_word_index(j), b] >> _bit_offset(j)) & UInt64(1)

########################################
# Qudit-major                          #
########################################

# `canonicalize!` on a packed qudit-major tableau. `div` sets the sparse
# cutoff (`_BIT_QM_CANON_SPARSE_DIV`, 0 for always sparse); every choice gives
# the same result.
function _canonicalize_bits_qm!(tab::_BitTableau, L::QuditMajor, div::Int)
    (0 <= tab.m <= tab.n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ tab.n"))
    _canonicalize_bits_qm!(_stab(tab, L), _bit_phase_planes(tab), _bit_canon_destab(tab, L),
                           tab, div)
    _bit_cache_qm!(tab.xdotz_cache.planes, _stab(tab, L), tab.m)
    tab.iscanonical = true
    return nothing
end

# The first generator in `c:m` whose bit is set in word column `q` of `A`, or 0.
@inline function _qm_bit_first_from(A::_BlockMatrix{UInt64}, q::Int, c::Int, m::Int)
    nwm = _nwords(m)
    w = _word_index(c)
    @inbounds v = A[w, q] & (typemax(UInt64) << _bit_offset(c))
    while true
        w == nwm && (v &= _tail_mask(m))
        v == 0 || return ((w - 1) << _WORD_SHIFT) + trailing_zeros(v) + 1
        w == nwm && return 0
        w += 1
        @inbounds v = A[w, q]
    end
end

function _canonicalize_bits_qm!(b::_QMBits, P::Union{Nothing,Matrix{UInt64}}, db,
                                tab::_BitTableau, div::Int)
    ws = tab.word_workspace
    mw = tab.c_workspace
    pivcol_of_row = tab.pivcol_of_row
    fill!(pivcol_of_row, 0)
    m = tab.m
    n = b.n
    nwm = _nwords(m)
    x, z = b.x, b.z
    r = 1
    c = 1
    while r <= 2n && c <= m
        A = r <= n ? x : z
        qr = r <= n ? r : r - n
        j = _qm_bit_first_from(A, qr, c, m)
        if j == 0
            r += 1
            continue
        end
        if j != c
            _swap_gens!(b, c, j)
            _bit_plane_swap!(P, c, j)
            db === nothing || _swap_gens!(db, c, j)
        end

        # β over generators 1:m without the pivot, and its nonzero words.
        wc, oc = _word_index(c), _bit_offset(c)
        nz = 0
        @inbounds for w in 1:nwm
            v = A[w, qr]
            w == wc && (v &= ~(UInt64(1) << oc))
            w == nwm && (v &= _tail_mask(m))
            ws[w, 1] = v
            if v != 0
                nz += 1
                mw[nz] = w
            end
        end
        if nz > 0
            if div == 0 || nz <= (nwm - 1) ÷ div
                _qm_bit_eliminate!(b, P, db, tab, c, view(mw, 1:nz))
            else
                _qm_bit_eliminate!(b, P, db, tab, c, Base.OneTo(nwm))
            end
        end
        @inbounds pivcol_of_row[r] = c
        r += 1
        c += 1
    end
    return nothing
end

# Eliminate pivot generator `c` from every generator in the mask β (column 1
# of `word_workspace`), visiting the β words `words`, which hold every nonzero
# one.
function _qm_bit_eliminate!(b::_QMBits, P::Union{Nothing,Matrix{UInt64}}, db,
                            tab::_BitTableau, c::Int, words::AbstractVector{Int})
    ws = tab.word_workspace
    sites = tab.zacc_workspace
    n = b.n
    x, z = b.x, b.z
    wc, oc = _word_index(c), _bit_offset(c)

    # The support of the pivot generator, with its X and Z bits in columns 3
    # and 4.
    @inbounds for w in 1:_nwords(n)
        ws[w, 3] = UInt64(0)
        ws[w, 4] = UInt64(0)
    end
    np = 0
    @inbounds for k in 1:n
        xb = (x[wc, k] >> oc) & UInt64(1)
        zb = (z[wc, k] >> oc) & UInt64(1)
        if (xb | zb) != 0
            np += 1
            sites[np] = k
            wk, ok = _word_index(k), _bit_offset(k)
            ws[wk, 3] |= xb << ok
            ws[wk, 4] |= zb << ok
        end
    end

    # The phases, from the parity words `⊻_{q: x_c[q] = 1} z[:, q]` of the
    # columns before the rank-1 update.
    if P !== nothing
        m0 = _bit_word((_bit_plane_bit(P, c, 1) % Int))
        m1 = _bit_word((_bit_plane_bit(P, c, 2) % Int))
        @inbounds @simd for w in words
            ws[w, 2] = UInt64(0)
        end
        @inbounds for s in 1:np
            k = sites[s]
            _ws_bit(ws, k, 3) == 0 && continue
            @simd ivdep for w in words
                ws[w, 2] ⊻= z[w, k]
            end
        end
        @inbounds @simd ivdep for w in words
            M = ws[w, 1]
            _bit_phase_add!(P, w, M & m0, M & (m1 ⊻ ws[w, 2]))
        end
    end

    # D_c ⊻= Σ β_j D_j, one parity per dual column.
    if db !== nothing
        dx, dz = db.x, db.z
        @inbounds for q in 1:n
            ax = UInt64(0)
            az = UInt64(0)
            @simd for w in words
                M = ws[w, 1]
                ax ⊻= M & dx[w, q]
                az ⊻= M & dz[w, q]
            end
            dx[wc, q] ⊻= ((count_ones(ax) % UInt64) & UInt64(1)) << oc
            dz[wc, q] ⊻= ((count_ones(az) % UInt64) & UInt64(1)) << oc
        end
    end

    # The rank-1 update of every column on the pivot's support.
    @inbounds for s in 1:np
        k = sites[s]
        if _ws_bit(ws, k, 3) != 0
            @simd ivdep for w in words
                x[w, k] ⊻= ws[w, 1]
            end
        end
        if _ws_bit(ws, k, 4) != 0
            @simd ivdep for w in words
                z[w, k] ⊻= ws[w, 1]
            end
        end
    end
    return nothing
end

# The x·z parity of every generator into the cache plane `C`, accumulated
# over the qudit columns; generators past `m` are zero.
function _bit_cache_qm!(C::Matrix{UInt64}, b::_QMBits, m::Int)
    x, z = b.x, b.z
    nwm = _nwords(m)
    fill!(C, UInt64(0))
    @inbounds for q in 1:b.n
        @simd for w in 1:nwm
            C[w, 1] ⊻= x[w, q] & z[w, q]
        end
    end
    nwm > 0 && (@inbounds C[nwm, 1] &= _tail_mask(m))
    return C
end

########################################
# Generator-major                      #
########################################

# `canonicalize!` on a packed generator-major tableau, one fused word pass per
# eliminated generator.
function _canonicalize_gm_bits!(tab::_BitTableau, L::GeneratorMajor)
    (0 <= tab.m <= tab.n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ tab.n"))
    _canonicalize_gm_bits!(_stab(tab, L), _bit_phase_planes(tab), _bit_canon_destab(tab, L),
                           tab.m, tab.pivcol_of_row)
    _bit_cache_gm!(tab.xdotz_cache.planes, _stab(tab, L), tab.m)
    tab.iscanonical = true
    return nothing
end

function _canonicalize_gm_bits!(b::_GMBits, P::Union{Nothing,Matrix{UInt64}}, db,
                                m::Int, pivcol_of_row::Vector{Int})
    fill!(pivcol_of_row, 0)
    n = b.n
    x, z = b.x, b.z
    nw = size(x, 1)
    r = 1
    c = 1
    while r <= 2n && c <= m
        A = r <= n ? x : z
        qr = r <= n ? r : r - n
        wr, orr = _word_index(qr), _bit_offset(qr)
        j = c
        @inbounds while j <= m && (A[wr, j] >> orr) & UInt64(1) == 0
            j += 1
        end
        if j > m
            r += 1
            continue
        end
        if j != c
            _swap_gens!(b, c, j)
            _bit_plane_swap!(P, c, j)
            db === nothing || _swap_gens!(db, c, j)
        end
        kl = _bit_plane_bit(P, c, 1)
        kh = _bit_plane_bit(P, c, 2)
        for jj in 1:m
            jj == c && continue
            @inbounds (A[wr, jj] >> orr) & UInt64(1) == 0 && continue
            if P === nothing
                @inbounds @simd for w in 1:nw
                    x[w, jj] ⊻= x[w, c]
                    z[w, jj] ⊻= z[w, c]
                end
            else
                cb = UInt64(0)
                @inbounds @simd for w in 1:nw
                    px = x[w, c]
                    zj = z[w, jj]
                    cb ⊻= px & zj
                    x[w, jj] ⊻= px
                    z[w, jj] = zj ⊻ z[w, c]
                end
                o = _bit_offset(jj)
                pb = (count_ones(cb) % UInt64) & UInt64(1)
                _bit_phase_add!(P, _word_index(jj), kl << o, (kh ⊻ pb) << o)
            end
            if db !== nothing
                dx, dz = db.x, db.z
                @inbounds @simd for w in 1:nw
                    dx[w, c] ⊻= dx[w, jj]
                    dz[w, c] ⊻= dz[w, jj]
                end
            end
        end
        @inbounds pivcol_of_row[r] = c
        r += 1
        c += 1
    end
    return nothing
end

# The x·z parity of every generator into the cache plane `C`; generators past
# `m` are zero.
function _bit_cache_gm!(C::Matrix{UInt64}, b::_GMBits, m::Int)
    x, z = b.x, b.z
    @inbounds for W in axes(C, 1)
        word = UInt64(0)
        j0 = (W - 1) << _WORD_SHIFT
        for k in 1:min(_WORD_BITS, m - j0)
            j = j0 + k
            s = UInt64(0)
            @simd for w in axes(x, 1)
                s ⊻= x[w, j] & z[w, j]
            end
            word |= ((count_ones(s) % UInt64) & UInt64(1)) << (k - 1)
        end
        C[W, 1] = word
    end
    return C
end
