########################################
# Image dot products                   #
########################################

# x(col) · z(col) mod d, for a column of F or a gathered coordinate tuple.
@inline function _col_xdotz(col::NTuple{S,Int}, K::Int, d::Int, fast::Bool) where {S}
    if fast
        s = 0
        @inbounds for q in 1:K
            s += col[q] * col[K + q]
        end
        return mod(s, d)
    end
    s = 0
    @inbounds for q in 1:K
        s = add_mod(s, mul_mod(col[q], col[K + q], d), d)
    end
    return s
end

# D[i] = x(F[:,i]) · z(F[:,i]) mod d. This is the ONLY derived phase data the
# design stores: it replaces v5's folded coefficient vector and quadratic
# matrix, dropping preparation from O(k^3) to O(k^2).
@inline _image_xdotz(F::NTuple{S,NTuple{S,Int}}, K::Int, d::Int, fast::Bool) where {S} =
    ntuple(i -> _col_xdotz(F[i], K, d, fast), Val(S))

@inline function _dot_mod(u::NTuple{S,Int}, v::NTuple{S,Int}, M::Int, fast::Bool) where {S}
    if fast
        s = 0
        @inbounds for i in 1:S
            s += u[i] * v[i]
        end
        return mod(s, M)
    end
    s = 0
    @inbounds for i in 1:S
        s = add_mod(s, mul_mod(u[i], v[i], M), M)
    end
    return s
end

########################################
# Phase evaluation                     #
########################################

# Odd primes (spec 3.2):
#     φ_U(v) = a·v + inv2 [ x_out·z_out − x·z − D·v ]   (mod d)
#
# Every dot is at most 2k terms and no triple product is formed, so the tier
# guard in `clifford_fast_dots` covers the whole expression. Given `Fv`, which
# the action has already computed, this is O(k).
#
# Precondition: `v` and `vout` are canonical, every entry already reduced
# mod `d` (Task 4 is what supplies `vout`). The final `mod`s only fix up the
# returned phase; they do not rescue an out-of-range input upstream of them.
# In particular the fast tier's accumulators (`_col_xdotz`, `_dot_mod`) sum
# entries assumed `< d`, which is exactly the bound `clifford_fast_dots`
# sizes its overflow guard against -- an unreduced entry can silently
# overflow there with no error.
@inline function _phase_odd(
    v::NTuple{S,Int},
    vout::NTuple{S,Int},
    a::NTuple{S,Int},
    D::NTuple{S,Int},
    K::Int,
    d::Int,
    inv2::Int,
    fast::Bool,
) where {S}
    av    = _dot_mod(a, v, d, fast)
    Dv    = _dot_mod(D, v, d, fast)
    xz    = _col_xdotz(v, K, d, fast)
    xzout = _col_xdotz(vout, K, d, fast)
    t = sub_mod(sub_mod(xzout, xz, d), Dv, d)
    return add_mod(av, mul_mod(inv2, t, d), d)
end

# Qubits (spec 3.3): multiply the generator images in order, carrying a running
# Z prefix. Canonical coefficients are bits, so binomial(v_i, 2) vanishes and
# the whole quadratic part is the cross terms.
#
# The prefix is a bitmask rather than a vector: at d = 2 every coordinate is a
# bit, so no heap scratch is needed and `apply!` stays allocation-free. This
# is a named-gate specialization (K ≤ 2), with an enforced K ≤ 64 bound.
# P2 adds a separate vector-prefix evaluator for arbitrary dense supports;
# the bitmask bound must not become a public Clifford arity restriction.
#
# Checked once per preparation, never per column: `_prepare` calls this when it
# resolves the qubit regime, so the evaluator below may assume it. A P2 dense
# operator past the bound must fail loudly rather than silently drop its high
# coordinates.
@inline function _check_qubit_bitmask_bound(K::Int)
    0 <= K <= 64 || throw(ArgumentError(
        "The bitmask qubit phase evaluator requires 0 ≤ K ≤ 64, got $K; " *
        "a larger support needs a vector prefix."))
    return nothing
end

# Precondition: `_check_qubit_bitmask_bound(K)` has already passed, the
# tuples are sized `S = 2K`, `v` is canonical (every `v[i] ∈ {0,1}`), and
# every entry of `F` is reduced mod 2 (`F[i][q] ∈ {0,1}`). Do not call this
# with an unvalidated `K` or a non-canonical `v`/`F`: the `v[i] == 0 &&
# continue` test and the `F[i][q] == 1` bit tests both assume canonical
# input and give a silent wrong answer otherwise -- e.g. `v[i] == 2` fails
# the `== 0` test, so generator `i` is (wrongly) treated as active, exactly
# as `v[i] == 1` would be, instead of erroring.
@inline function _phase_qubit(
    v::NTuple{S,Int},
    F::NTuple{S,NTuple{S,Int}},
    a::NTuple{S,Int},
    K::Int,
) where {S}
    phase = 0
    zpref = UInt64(0)
    @inbounds for i in 1:S
        v[i] == 0 && continue
        phase = add_mod(phase, a[i], 4)
        xmask = UInt64(0)
        for q in 1:K
            F[i][q] == 1 && (xmask |= (UInt64(1) << (q - 1)))
        end
        # Reducing the prefix mod 2 is valid because it is multiplied by 2 in
        # phase modulus 4.
        phase = add_mod(phase, 2 * (count_ones(zpref & xmask) & 1), 4)
        for q in 1:K
            F[i][K + q] == 1 && (zpref ⊻= (UInt64(1) << (q - 1)))
        end
    end
    return phase
end
