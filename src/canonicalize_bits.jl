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
# Row `r` is one bit of each generator's word column, generator-major. Each
# eliminated generator takes one pass over its words, which XORs the pivot in
# and accumulates the parity of `x_c & z_j`; the phase planes change once
# after the pass. Duals are XORed word by word. The generic algorithm on the
# same packed blocks reads the pivot row bit by bit and takes separate passes
# for the cross term and the update, so it is slower on random and monitored
# states alike.
#
# A qudit-major tableau is switched to generator-major in place, canonicalized
# by the same kernel and switched back. A packed switch is two or four block
# transposes, cheap next to the elimination. Eliminating along the qudit
# columns instead has to gather each pivot generator across qudits, and a
# `DestabilizerTableau` takes a parity over every dual column per pivot, so
# such a kernel is slower than the switched route, most of all on monitored
# states.

_canonicalize!(tab::_BitStab, ::QuditMajor) = _canonicalize_bits_switched!(tab)
_canonicalize!(tab::_BitDestab, ::QuditMajor) = _canonicalize_bits_switched!(tab)
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

# `canonicalize!` on a packed qudit-major tableau: switched to generator-major,
# canonicalized by the kernel, and switched back. The switches allocate
# nothing and leave the phase and cache planes alone.
function _canonicalize_bits_switched!(tab::_BitTableau)
    (0 <= tab.m <= tab.n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ tab.n"))
    generator_major!(tab)
    _canonicalize_gm_bits!(tab, GeneratorMajor())
    qudit_major!(tab)
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
