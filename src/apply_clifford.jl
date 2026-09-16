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

########################################
# Preparation                          #
########################################

# Resolved once per `apply!`, never per column: the gate's raw data, the image
# dot products, the arithmetic tier, and the phase regime. Immutable and
# built from tuples, so a named-gate application allocates nothing.
struct PreparedClifford{K,S}
    targets::NTuple{K,Int}
    F::NTuple{S,NTuple{S,Int}}
    a::NTuple{S,Int}
    D::NTuple{S,Int}
    d::Int
    p::Int
    inv2::Int
    fast::Bool
    storephase::Bool
end

# `force_safe` is an internal testing hook; it can only disable the fast tier.
@inline function _prepare(
    g::AbstractClifford,
    d::Int,
    inversemod::InverseMod,
    storephase::Bool;
    force_safe::Bool=false,
)
    targets, F, a = _clifford_data(g, d, inversemod)
    K = length(targets)
    S = 2K
    fast = !force_safe && clifford_fast_dots(S, d)
    # Only the odd-prime evaluator needs D and inv2; skip both otherwise. A
    # `PrecomputedInvMod` must never be called with x >= d, hence the d != 2
    # guard rather than a bare `inversemod(2, d)`.
    needs_odd = storephase && d != 2
    # The qubit evaluator's bitmask bound is a precondition of the regime, so
    # it is established here rather than re-tested on every generator column.
    storephase && d == 2 && _check_qubit_bitmask_bound(K)
    inv2 = needs_odd ? inversemod(2, d) : 0
    D = needs_odd ? _image_xdotz(F, K, d, fast) : ntuple(_ -> 0, Val(S))
    return PreparedClifford{K,S}(targets, F, a, D, d, phase_modulus(d),
                                 inv2, fast, storephase)
end

@inline function _validate_targets(targets::NTuple{K,Int}, n::Int) where {K}
    @inbounds for i in 1:K
        t = targets[i]
        (1 <= t <= n) || throw(ArgumentError(
            "Clifford target $t is outside the register 1:$n."))
    end
    return nothing
end

########################################
# Gather / matvec / scatter            #
########################################

# Target coordinates in all-X-then-all-Z order, matching the generator order.
@inline function _gather(A::Matrix{Int}, targets::NTuple{K,Int}, n::Int, j::Int) where {K}
    return ntuple(Val(2K)) do i
        @inbounds i <= K ? A[targets[i], j] : A[n + targets[i - K], j]
    end
end

@inline function _scatter!(
    A::Matrix{Int},
    targets::NTuple{K,Int},
    n::Int,
    j::Int,
    vout::NTuple{S,Int},
) where {K,S}
    @inbounds for i in 1:K
        A[targets[i], j] = vout[i]
        A[n + targets[i], j] = vout[K + i]
    end
    return nothing
end

# (Fv)_i = Σ_j F[i,j] v_j, and F[j] is column j, so F[i,j] is F[j][i].
@inline function _matvec(prep::PreparedClifford{K,S}, v::NTuple{S,Int}) where {K,S}
    d = prep.d
    F = prep.F
    fast = prep.fast
    return ntuple(Val(S)) do i
        if fast
            s = 0
            @inbounds for j in 1:S
                s += F[j][i] * v[j]
            end
            mod(s, d)
        else
            s = 0
            @inbounds for j in 1:S
                s = add_mod(s, mul_mod(F[j][i], v[j], d), d)
            end
            s
        end
    end
end

@inline function _phase(
    prep::PreparedClifford{K,S},
    v::NTuple{S,Int},
    vout::NTuple{S,Int},
) where {K,S}
    if prep.d == 2
        return _phase_qubit(v, prep.F, prep.a, K)
    else
        return _phase_odd(v, vout, prep.a, prep.D, K, prep.d, prep.inv2, prep.fast)
    end
end

########################################
# Type-specific hooks                  #
########################################

# Specialised for `DestabilizerTableau` in src/apply_clifford.jl below; a plain
# `StabilizerTableau` needs neither, because clearing `iscanonical` is the
# existing contract for a stale cache.
@inline _before_clifford_scatter!(::AbstractTableau, prep, j::Int, v, vout) = nothing
@inline _after_clifford!(::AbstractTableau, prep) = nothing

########################################
# apply!                               #
########################################

"""
    apply!(tab::AbstractTableau, g::AbstractClifford)

Apply a Clifford unitary to `tab` in place by conjugation, returning `tab`.

Each stabilizer generator `P` becomes `U P U†`. Only the gate's target rows and
the active columns `1:m` change; `m`, the zeroed unused capacity, and the
generator contract are all preserved. `tab.iscanonical` is cleared.

# Arguments
- `tab::AbstractTableau`: tableau to update (`StabilizerTableau` or `DestabilizerTableau`).
- `g::AbstractClifford`: a named gate such as [`Fourier`](@ref) or [`SUM`](@ref).

# Throws
`ArgumentError` if a target lies outside `1:tab.n`, or if a gate parameter is
invalid at `tab.d` (for example a [`Multiplier`](@ref) coefficient congruent to
zero). Validation happens before any mutation.

# Examples
```julia
tab = DestabilizerTableau(3, 2; state=:product, basis=:Z)
apply!(tab, Fourier(1))
apply!(tab, SUM(1, 2))
```

# Notes
For a [`DestabilizerTableau`](@ref) the dual basis is transformed by the same
symplectic map, which preserves `⟨D_j, S_k⟩ = δ_jk` without re-orthogonalization,
and `xdotz_cache` is patched by a local delta.

With `storephase=false` no phase work is done and only the exponents move.

See also [`conjugate`](@ref), [`measure!`](@ref).
"""
function apply!(tab::AbstractTableau, g::AbstractClifford)
    _validate_targets(_clifford_targets(g), tab.n)
    prep = _prepare(g, tab.d, tab.inversemod, tab.storephase)
    return _apply_prepared!(tab, prep)
end

function _apply_prepared!(tab::AbstractTableau, prep::PreparedClifford{K,S}) where {K,S}
    K == 0 && return tab
    n = tab.n
    stab = tab.stab
    phase_row = 2n + 1
    @inbounds for j in 1:tab.m
        v = _gather(stab, prep.targets, n, j)
        vout = _matvec(prep, v)
        if prep.storephase
            δ = _phase(prep, v, vout)
            stab[phase_row, j] = add_mod(stab[phase_row, j], δ, prep.p)
        end
        # The cache delta needs the OLD target entries, so it runs before the
        # scatter overwrites them.
        _before_clifford_scatter!(tab, prep, j, v, vout)
        _scatter!(stab, prep.targets, n, j, vout)
    end
    _after_clifford!(tab, prep)
    tab.iscanonical = false
    return tab
end
