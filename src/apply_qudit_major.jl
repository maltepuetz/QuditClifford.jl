########################################
# Qudit-major gate kernel              #
########################################
#
# In the qudit-major layout the X (Z) exponent of qudit `t` over all
# generators is one contiguous column, so a gate on `K` qudits reads and writes
# `2K` columns, and one SIMD loop over the generators does the whole gate. The
# loop runs in a lane type `T ∈ (Int16, Int32, Int64)` chosen by
# `_clifford_lane_bytes`: entries are widened to `T`, every product and sum is
# formed in `T` and reduced by `_barrett_reduce` under the constant `(M, s)` of
# `_clifford_barrett(T, S, d)`, so every unreduced value stays at most
# `X = 2S·d·(d-1)`, and results are stored back as `E`. The formulas are those
# of `_matvec`, `_phase_odd_barrett`, `_phase_qubit_closed` and
# `_col_xdotz_prepared` on the Barrett tier, so the results are identical to
# the per-generator path.
#
# The closures carry their own `@inbounds`: the loop's does not reach into them.
#
# The `Int` phase and `xdotz_cache` are not touched inside that loop: an `Int`
# load and store there forces `Int`-width vectors on the whole loop body. Pass 1
# stores each generator's phase delta and cache delta, reduced, as `E` into
# columns 1 and 2 of `coeff_workspace`; pass 2 adds them into the `Int` vectors
# with one conditional subtract each.

@inline _lane_tuple(::Type{T}, t::NTuple{S,Int}) where {T,S} = ntuple(i -> t[i] % T, Val(S))

# One gate on the `m` active generators of the qudit-major block `b`. `phase`
# is the phase vector, or `nothing` for no phase work; `cache` is the
# `xdotz_cache`, or `nothing` when there is none. `ws` is the tableau's
# `coeff_workspace`. Requires `prep.bm != 0` and `(M, s) =
# _clifford_barrett(T, S, prep.d)` with `M != 0`; at `d = 2` the lane is not
# used for reductions and any `T` holds every value (`<= 3S`).
@inline function _qm_tuple_kernel!(b::_QMBlock, phase::Union{Nothing,Vector{Int}},
                                   cache::Union{Nothing,Vector{Int}},
                                   ws::Matrix{E}, m::Int,
                                   prep::PreparedClifford{K,S}, ::Type{T}, M::T,
                                   s::Int) where {E,K,S,T<:Signed}
    t = prep.targets
    xc = ntuple(i -> view(b.x, :, t[i]), Val(K))
    zc = ntuple(i -> view(b.z, :, t[i]), Val(K))
    wp = view(ws, :, 1)
    wc = view(ws, :, 2)
    if prep.d == 2
        _qm_pass_qubit!(xc, zc, wp, wc, m, prep, T, phase !== nothing, cache !== nothing)
    else
        _qm_pass_odd!(xc, zc, wp, wc, m, prep, T, M, s, phase !== nothing, cache !== nothing)
    end
    # The deltas are reduced (`< p` and `< d`) and so are the stored values, so
    # each sum is below twice the modulus and one subtract reduces it.
    if phase !== nothing
        p = prep.p
        @inbounds @simd for j in 1:m
            x = phase[j] + Int(wp[j])
            phase[j] = x >= p ? x - p : x
        end
    end
    if cache !== nothing
        d = prep.d
        @inbounds @simd for j in 1:m
            x = cache[j] + Int(wc[j])
            cache[j] = x >= d ? x - d : x
        end
    end
    return nothing
end

# Sums over a gate's coordinates are written over tuples, not as loops: an
# inner loop that is still a loop when the vectorizer runs (two accumulators
# over `1:K` were enough) leaves the generator loop scalar.
# The arguments are nonempty tuples of one lane type, of equal length.
@inline _lane_sum(t::Tuple) = +(t...)
@inline _lane_dot(u::Tuple, v::Tuple) =
    _lane_sum(ntuple(i -> @inbounds(u[i] * v[i]), Val(length(u))))
@inline _lane_and(u::Tuple, v::Tuple) =
    ntuple(i -> @inbounds(u[i] & v[i]), Val(length(u)))
@inline _lane_xor(t::Tuple) = ⊻(t...)

@inline _xpart(v::NTuple{S}, ::Val{K}) where {S,K} = ntuple(q -> @inbounds(v[q]), Val(K))
@inline _zpart(v::NTuple{S}, ::Val{K}) where {S,K} = ntuple(q -> @inbounds(v[K + q]), Val(K))

# Row `i` of the gate matrix: `F[k][i]` for every column `k`, in `T`.
@inline _lane_rows(::Type{T}, F::NTuple{S,NTuple{S,Int}}) where {T,S} =
    ntuple(i -> ntuple(k -> @inbounds(F[k][i]) % T, Val(S)), Val(S))

@inline function _qm_gather(xc::NTuple{K}, zc::NTuple{K}, j::Int, ::Type{T},
                            ::Val{S}) where {K,S,T}
    return ntuple(Val(S)) do i
        @inline
        @inbounds i <= K ? xc[i][j] % T : zc[i - K][j] % T
    end
end

@inline function _qm_scatter!(xc::NTuple{K}, zc::NTuple{K}, j::Int, vout::NTuple{S},
                              ::Type{E}) where {K,S,E}
    ntuple(Val(K)) do q
        @inline
        @inbounds xc[q][j] = vout[q] % E
        @inbounds zc[q][j] = vout[K + q] % E
        nothing
    end
    return nothing
end

# Pass 1 at odd `d`. `dophase` and `docache` are constants after inlining, so
# each combination compiles to its own loop.
@inline function _qm_pass_odd!(xc::NTuple{K}, zc::NTuple{K}, wp::AbstractVector{E},
                               wc::AbstractVector{E}, m::Int,
                               prep::PreparedClifford{K,S}, ::Type{T}, M::T, s::Int,
                               dophase::Bool, docache::Bool) where {E,K,S,T<:Signed}
    di = prep.d
    d = di % T
    Fr = _lane_rows(T, prep.F)
    a = _lane_tuple(T, prep.a)
    D = _lane_tuple(T, prep.D)
    inv2 = prep.inv2 % T
    # Multiples of `d` no smaller than `x·z + D·v <= (K+S)(d-1)^2` and
    # `x·z <= K(d-1)^2`; both are at most `X`, so they fit in `T`.
    C = ((K + S) * (di - 1) * di) % T
    Cc = (K * (di - 1) * di) % T
    @inbounds @simd ivdep for j in 1:m
        v = _qm_gather(xc, zc, j, T, Val(S))
        # Each row sum is at most S(d-1)^2 <= X.
        vout = ntuple(i -> _barrett_reduce(_lane_dot(Fr[i], v), d, M, s), Val(S))
        xz = _lane_dot(_xpart(v, Val(K)), _zpart(v, Val(K)))
        xzout = _lane_dot(_xpart(vout, Val(K)), _zpart(vout, Val(K)))
        if dophase
            av = _lane_dot(a, v)
            Dv = _lane_dot(D, v)
            # x_out·z_out + C - x·z - D·v <= K(d-1)^2 + (K+S)(d-1)d <= X, and
            # a·v + inv2·t <= S(d-1)^2 + (d-1)^2 <= X.
            tt = _barrett_reduce(xzout + C - xz - Dv, d, M, s)
            wp[j] = _barrett_reduce(av + inv2 * tt, d, M, s) % E
        end
        if docache
            # x_out·z_out + Cc - x·z <= K(d-1)^2 + K(d-1)d <= X.
            wc[j] = _barrett_reduce(xzout + Cc - xz, d, M, s) % E
        end
        _qm_scatter!(xc, zc, j, vout, E)
    end
    return nothing
end

# Pass 1 at `d = 2`: entries are bits, so products are `&` and sums mod 2 are
# `⊻`. The phase delta is `_phase_qubit_closed`: `a·v <= 3S` and the cross
# term is a bit, so the sum before `& 3` is at most `3S + 2`. The cache delta
# is the change of the parity `x·z mod 2`.
@inline function _qm_pass_qubit!(xc::NTuple{K}, zc::NTuple{K}, wp::AbstractVector{E},
                                 wc::AbstractVector{E}, m::Int,
                                 prep::PreparedClifford{K,S}, ::Type{T},
                                 dophase::Bool, docache::Bool) where {E,K,S,T<:Signed}
    Fr = _lane_rows(T, prep.F)
    a = _lane_tuple(T, prep.a)
    Q = ntuple(i -> _lane_tuple(T, prep.Q[i]), Val(S))
    @inbounds @simd ivdep for j in 1:m
        v = _qm_gather(xc, zc, j, T, Val(S))
        vout = ntuple(i -> _lane_xor(_lane_and(Fr[i], v)), Val(S))
        if dophase
            # Q[i][k] is zero for k <= i, so row i of Q against v sums the
            # pairs i < k.
            cross = _lane_xor(ntuple(i -> v[i] & _lane_xor(_lane_and(Q[i], v)), Val(S)))
            wp[j] = ((_lane_dot(a, v) + T(2) * cross) & T(3)) % E
        end
        if docache
            xz = _lane_xor(_lane_and(_xpart(v, Val(K)), _zpart(v, Val(K))))
            xzout = _lane_xor(_lane_and(_xpart(vout, Val(K)), _zpart(vout, Val(K))))
            wc[j] = (xz ⊻ xzout) % E
        end
        _qm_scatter!(xc, zc, j, vout, E)
    end
    return nothing
end

# The whole gate on a qudit-major tableau in lane `T`: the stabilizer block with
# its phase and cache, then the destabilizer block of a `DestabilizerTableau`.
@inline function _qm_apply!(tab::AbstractTableau, prep::PreparedClifford{K,S},
                            L::QuditMajor, ::Type{T}) where {K,S,T<:Signed}
    M, s = _clifford_barrett(T, S, prep.d)
    stab = _stab(tab, L)
    cache = _qm_cache(tab)
    if prep.storephase
        _qm_tuple_kernel!(stab, _phase(tab), cache, tab.coeff_workspace, tab.m, prep, T, M, s)
    else
        _qm_tuple_kernel!(stab, nothing, cache, tab.coeff_workspace, tab.m, prep, T, M, s)
    end
    _after_clifford!(tab, L, prep, T, M, s)
    return nothing
end

@inline _qm_cache(::AbstractTableau) = nothing
@inline _qm_cache(tab::DestabilizerTableau) = tab.xdotz_cache
