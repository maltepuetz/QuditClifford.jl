########################################
# Packed qubit gate kernels            #
########################################
#
# A gate on packed storage runs bit-sliced: bit `k` of every word is one
# generator, so one pass of `_bit_gate_core` over `2K` words applies the gate
# to 64 generators at once. The gate's data become words that are all ones or
# zero (`_BitPrep`), so every product of the `d = 2` arithmetic is an `&` and
# every sum mod 2 a `⊻`:
#
# - the new coordinates are `o_i = ⊻_k F[k][i] & v_k`, the formula of `_matvec`;
# - the phase delta is `_phase_qubit_closed`, `a·v + 2 Σ_{i<j} Q_ij v_i v_j`
#   mod 4, kept as two planes `(l, h)` (`delta = l + 2h`). Adding `a_i v_i`
#   adds its low bit to `l` and its high bit, plus the carry out of `l`, to
#   `h`; the cross terms land in `h` only;
# - the `xdotz_cache` delta of a `DestabilizerTableau` is the change of the
#   parity of `x·z` over the targets.
#
# The phase and the cache are bit planes indexed by generator in both layouts,
# so each takes one word update per 64 generators: `p0' = p0 ⊻ l` and
# `p1' = p1 ⊻ h ⊻ (p0 & l)` for the phase, `c' = c ⊻ delta` for the cache.
#
# Qudit-major, word `w` of the `2K` target columns holds the coordinates of
# generators `64(w-1)+1 … 64w`, so one SIMD loop over the words streams the
# gate. Generator-major, each batch of 64 generators gathers its target bits
# into one word per coordinate, runs the same core, and flips the bits that
# changed.
#
# Every output word is an `&` with an input word, so generators whose target
# bits are all zero (capacity generators, the bits past `n`) stay zero, and so
# do their phase and cache bits.

# The gate data of a `PreparedClifford`, as words that are all ones (an odd
# entry) or zero. `F[i][k]` is entry `k` of row `i`, so output `i` is
# `⊻_k F[i][k] & v_k`; `a0`/`a1` are bits 0 and 1 of the phase coefficients
# mod 4, and `Q` the cross terms of `_qubit_cross_bits`.
struct _BitPrep{K,S}
    targets::NTuple{K,Int}
    F::NTuple{S,NTuple{S,UInt64}}
    a0::NTuple{S,UInt64}
    a1::NTuple{S,UInt64}
    Q::NTuple{S,NTuple{S,UInt64}}
end

@inline _bit_word(x::Int) = isodd(x) ? typemax(UInt64) : UInt64(0)

# Built from the gate matrix on every tier: `Q` is derived here rather than read
# from `prep.Q`, which only the Barrett tier fills.
@inline function _bit_prep(prep::PreparedClifford{K,S}) where {K,S}
    F = prep.F
    a = prep.a
    Q = _qubit_cross_bits(F, K)
    Fw = ntuple(Val(S)) do i
        @inline
        ntuple(k -> (@inline; @inbounds _bit_word(F[k][i])), Val(S))
    end
    a0 = ntuple(i -> (@inline; @inbounds _bit_word(mod(a[i], 4))), Val(S))
    a1 = ntuple(i -> (@inline; @inbounds _bit_word(mod(a[i], 4) >> 1)), Val(S))
    Qw = ntuple(Val(S)) do i
        @inline
        ntuple(j -> (@inline; @inbounds _bit_word(Q[i][j])), Val(S))
    end
    return _BitPrep{K,S}(prep.targets, Fw, a0, a1, Qw)
end

# The gate on 64 generators: the new target words, and the phase delta planes
# `(l, h)`, `delta = l + 2h` mod 4 per bit.
@inline function _bit_gate_core(v::NTuple{S,UInt64}, bp::_BitPrep{K,S}) where {K,S}
    l = UInt64(0)
    h = UInt64(0)
    @inbounds for i in 1:S
        b0 = v[i] & bp.a0[i]
        b1 = v[i] & bp.a1[i]
        c = l & b0
        l ⊻= b0
        h ⊻= b1 ⊻ c
        for j in (i + 1):S
            h ⊻= bp.Q[i][j] & v[i] & v[j]
        end
    end
    o = ntuple(Val(S)) do i
        @inline
        s = UInt64(0)
        @inbounds for k in 1:S
            s ⊻= bp.F[i][k] & v[k]
        end
        s
    end
    return o, l, h
end

# The parity of `x·z` over the targets, one bit per generator.
@inline function _bit_xz_parity(v::NTuple{S,UInt64}, ::Val{K}) where {S,K}
    s = UInt64(0)
    @inbounds for q in 1:K
        s ⊻= v[q] & v[K + q]
    end
    return s
end

# Word `w` of the phase planes `P` (`nothing`: no phase) and the cache plane
# `C` (`nothing`: no cache), after a core pass on the words `v` → `o`.
@inline function _bit_planes_update!(P::Union{Nothing,Matrix{UInt64}},
                                     C::Union{Nothing,Matrix{UInt64}}, w::Int,
                                     v::NTuple{S,UInt64}, o::NTuple{S,UInt64},
                                     l::UInt64, h::UInt64, ::Val{K}) where {S,K}
    @inbounds if P !== nothing
        p0 = P[w, 1]
        P[w, 1] = p0 ⊻ l
        P[w, 2] ⊻= h ⊻ (p0 & l)
    end
    @inbounds if C !== nothing
        C[w, 1] ⊻= _bit_xz_parity(v, Val(K)) ⊻ _bit_xz_parity(o, Val(K))
    end
    return nothing
end

# The gate on the `m` active generators of a qudit-major packed block: one SIMD
# loop over the words of the `2K` target columns.
@inline function _bit_gate!(b::_QMBits, P::Union{Nothing,Matrix{UInt64}},
                            C::Union{Nothing,Matrix{UInt64}}, m::Int,
                            bp::_BitPrep{K,S}) where {K,S}
    t = bp.targets
    x, z = b.x, b.z
    cols = ntuple(Val(S)) do i
        @inline
        @inbounds i <= K ? view(x, :, t[i]) : view(z, :, t[i - K])
    end
    @inbounds @simd ivdep for w in 1:_nwords(m)
        v = ntuple(i -> (@inline; @inbounds cols[i][w]), Val(S))
        o, l, h = _bit_gate_core(v, bp)
        ntuple(Val(S)) do i
            @inline
            @inbounds cols[i][w] = o[i]
            nothing
        end
        _bit_planes_update!(P, C, w, v, o, l, h, Val(K))
    end
    return nothing
end

# `v` with bit `k` of each word set from the target bit of generator `j`.
@inline function _bit_gather(v::NTuple{S,UInt64}, mats::NTuple{S}, rows::NTuple{S,Int},
                             offs::NTuple{S,Int}, j::Int, k::Int) where {S}
    return ntuple(Val(S)) do i
        @inline
        @inbounds v[i] | (((mats[i][rows[i], j] >> offs[i]) & UInt64(1)) << k)
    end
end

# The gate on the `m` active generators of a generator-major packed block, 64
# generators per batch. Batch `w` covers generators `64(w-1)+1 … min(64w, m)`:
# their target bits are gathered into word bits `0 … cnt-1`, and only those
# columns are written back, so generators past `m` are never touched.
@inline function _bit_gate!(b::_GMBits, P::Union{Nothing,Matrix{UInt64}},
                            C::Union{Nothing,Matrix{UInt64}}, m::Int,
                            bp::_BitPrep{K,S}) where {K,S}
    t = bp.targets
    x, z = b.x, b.z
    mats = ntuple(i -> (@inline; i <= K ? x : z), Val(S))
    rows = ntuple(i -> (@inline; @inbounds _word_index(t[i <= K ? i : i - K])), Val(S))
    offs = ntuple(i -> (@inline; @inbounds _bit_offset(t[i <= K ? i : i - K])), Val(S))
    @inbounds for w in 1:_nwords(m)
        j0 = (w - 1) << _WORD_SHIFT
        cnt = min(_WORD_BITS, m - j0)
        v = ntuple(_ -> UInt64(0), Val(S))
        for k in 0:(cnt - 1)
            v = _bit_gather(v, mats, rows, offs, j0 + k + 1, k)
        end
        o, l, h = _bit_gate_core(v, bp)
        flips = map(⊻, o, v)
        for k in 0:(cnt - 1)
            j = j0 + k + 1
            ntuple(Val(S)) do i
                @inline
                @inbounds mats[i][rows[i], j] ⊻= ((flips[i] >> k) & UInt64(1)) << offs[i]
                nothing
            end
        end
        _bit_planes_update!(P, C, w, v, o, l, h, Val(K))
    end
    return nothing
end

@inline _bit_cache(::_BitStab) = nothing
@inline _bit_cache(tab::_BitDestab) = tab.xdotz_cache.planes

# The stabilizer block with its phase and cache, then the destabilizer block of
# a `DestabilizerTableau`. Not inlined into `_apply_columns!`: compiled into
# the same body as `_bit_prep`, a gate took about 1.4x as long (two-qubit
# gates, qudit-major `StabilizerTableau`, n = 2048, M3 Pro).
@noinline function _bit_apply!(tab::_BitTableau, L::TableauLayout, bp::_BitPrep)
    stab = _stab(tab, L)
    cache = _bit_cache(tab)
    if tab.storephase
        _bit_gate!(stab, tab.phase.planes, cache, tab.m, bp)
    else
        _bit_gate!(stab, nothing, cache, tab.m, bp)
    end
    _after_clifford!(tab, L, bp)
    tab.iscanonical = false
    return tab
end

# The destabilizer pass: the same gate on the destabilizer block, with neither
# phase nor cache.
@inline function _after_clifford!(tab::_BitDestab, L::TableauLayout, bp::_BitPrep)
    _bit_gate!(_destab(tab, L), nothing, nothing, tab.m, bp)
    return nothing
end

@noinline _apply_columns!(tab::_BitTableau, prep::PreparedClifford{K,S},
                          L::QuditMajor) where {K,S} = _bit_apply!(tab, L, _bit_prep(prep))

@noinline _apply_columns!(tab::_BitTableau, prep::PreparedClifford{K,S},
                          L::GeneratorMajor) where {K,S} = _bit_apply!(tab, L, _bit_prep(prep))
