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

# D[i] = x(F[:,i]) · z(F[:,i]) mod d. This is the ONLY derived phase data
# stored, and it is what holds preparation to O(k^2): both evaluators
# reconstruct the quadratic part of the phase from `D` and the gathered
# column, so no per-gate coefficient vector or quadratic matrix is built.
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

# Odd primes:
#     φ_U(v) = a·v + inv2 [ x_out·z_out − x·z − D·v ]   (mod d)
#
# Every dot is at most 2k terms and no triple product is formed, so the tier
# guard in `clifford_fast_dots` covers the whole expression. Given `Fv`, which
# the action has already computed, this is O(k).
#
# Precondition: `v` and `vout` are canonical, every entry already reduced
# mod `d` (the caller's matvec is what supplies `vout`). The final `mod`s
# only fix up the returned phase; they do not rescue an out-of-range input
# upstream of them.
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

# Qubits: multiply the generator images in order, carrying a running Z prefix.
# Canonical coefficients are bits, so binomial(v_i, 2) vanishes and the whole
# quadratic part is the cross terms.
#
# The prefix is a bitmask rather than a vector: at d = 2 every coordinate is a
# bit, so no heap scratch is needed and `apply!` stays allocation-free. This is
# a named-gate specialization (K ≤ 2), with an enforced K ≤ 64 bound.
# `_phase_qubit_dense` is the vector-prefix evaluator used for arbitrary dense
# supports; the bitmask bound is therefore internal to this path and is not a
# public Clifford arity restriction. Note `UInt64(1) << 64 == 0` in Julia, so a
# bitmask would silently drop coordinate 65 rather than erroring -- which is why
# the bound is checked once in `_prepare`.
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

# Backing-agnostic tag. `PreparedClifford` is the compile-time tuple backing;
# `PreparedDenseClifford` is the runtime-k dense one. The pipeline below is
# written once against this tag and specialized by Julia per backing.
abstract type AbstractPreparedClifford end

# Resolved once per `apply!`, never per column: the gate's raw data, the image
# dot products, the arithmetic tier, and the phase regime. Immutable and
# built from tuples, so a named-gate application allocates nothing.
struct PreparedClifford{K,S} <: AbstractPreparedClifford
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
        for j in 1:(i - 1)
            targets[j] == t && throw(ArgumentError(
                "Clifford targets must be distinct; $t appears more than once " *
                "in $targets."))
        end
    end
    return nothing
end

# Distinctness is O(k^2) and is a CONSTRUCTOR invariant for a stored operator,
# so consumption re-derives only the O(k) bounds for backings that carry that
# proof. Directly built tuple preparations keep the full defense.
@inline _validate_clifford_targets(g::AbstractClifford, n::Int) =
    _validate_targets(_clifford_targets(g), n)

@inline _validate_clifford_targets(prep::PreparedClifford, n::Int) =
    _validate_targets(prep.targets, n)

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

########################################
# Prepared-operation protocol          #
########################################
#
# Seven operations carry the backing difference; everything above them is
# written once. The tuple methods delegate to the tuple primitives (`_gather`,
# `_scatter!`, `_matvec`, `_col_xdotz`), which keep their own direct tests. The
# bang marks that a backing MAY mutate borrowed scratch -- the tuple backing
# does not need to.

@inline _gather_prepared!(prep::PreparedClifford{K,S}, A::Matrix{Int}, n::Int,
                          j::Int) where {K,S} = _gather(A, prep.targets, n, j)

@inline _matvec_prepared!(prep::PreparedClifford{K,S},
                          v::NTuple{S,Int}) where {K,S} = _matvec(prep, v)

@inline _scatter_prepared!(prep::PreparedClifford{K,S}, A::Matrix{Int}, n::Int,
                           j::Int, vout::NTuple{S,Int}) where {K,S} =
    _scatter!(A, prep.targets, n, j, vout)

@inline _col_xdotz_prepared(prep::PreparedClifford{K,S},
                            v::NTuple{S,Int}) where {K,S} =
    _col_xdotz(v, K, prep.d, prep.fast)

@inline function _gather_vec_prepared!(prep::PreparedClifford{K,S},
                                       xz::Vector{Int}, n::Int) where {K,S}
    t = prep.targets
    return ntuple(Val(S)) do i
        @inbounds i <= K ? xz[t[i]] : xz[n + t[i - K]]
    end
end

@inline function _scatter_vec_prepared!(prep::PreparedClifford{K,S},
                                        xz::Vector{Int}, n::Int,
                                        vout::NTuple{S,Int}) where {K,S}
    t = prep.targets
    @inbounds for i in 1:K
        xz[t[i]] = vout[i]
        xz[n + t[i]] = vout[K + i]
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
# Dense backing                        #
########################################
#
# Owns NOTHING. Its semantic arrays and its scratch may come from different
# owners: `apply!` borrows both from the operator, while the allocating APIs
# pair the operand's semantic arrays with result-owned or call-local scratch.
#
# Two aliasing invariants, both free on the tuple backing and both violable here:
#   * `v !== vout` -- the old gather stays live through phase evaluation and the
#     pre-scatter `xdotz_cache` delta;
#   * `_phase` never mutates `v` or `vout`; the qubit evaluator writes `zpref`.
struct PreparedDenseClifford <: AbstractPreparedClifford
    targets::Vector{Int}
    F::Matrix{Int}
    a::Vector{Int}
    D::Vector{Int}
    v::Vector{Int}
    vout::Vector{Int}
    zpref::Vector{Int}
    k::Int
    d::Int
    p::Int
    inv2::Int
    fast::Bool
    storephase::Bool
end

@inline function _gather_prepared!(prep::PreparedDenseClifford, A::Matrix{Int},
                                   n::Int, j::Int)
    t = prep.targets; k = prep.k; v = prep.v
    @inbounds for i in 1:k
        v[i] = A[t[i], j]
        v[k + i] = A[n + t[i], j]
    end
    return v
end

@inline function _scatter_prepared!(prep::PreparedDenseClifford, A::Matrix{Int},
                                    n::Int, j::Int, vout::Vector{Int})
    t = prep.targets; k = prep.k
    @inbounds for i in 1:k
        A[t[i], j] = vout[i]
        A[n + t[i], j] = vout[k + i]
    end
    return nothing
end

@inline function _gather_vec_prepared!(prep::PreparedDenseClifford,
                                       xz::Vector{Int}, n::Int)
    t = prep.targets; k = prep.k; v = prep.v
    @inbounds for i in 1:k
        v[i] = xz[t[i]]
        v[k + i] = xz[n + t[i]]
    end
    return v
end

@inline function _scatter_vec_prepared!(prep::PreparedDenseClifford,
                                        xz::Vector{Int}, n::Int,
                                        vout::Vector{Int})
    t = prep.targets; k = prep.k
    @inbounds for i in 1:k
        xz[t[i]] = vout[i]
        xz[n + t[i]] = vout[k + i]
    end
    return nothing
end

# Size (S = 2k) up to which the dense matvec keeps the row-wise dot product.
# Row-wise sums stay in a register and win for small supports; above this size
# the stride-S row reads lose to a column pass over contiguous columns, which
# also skips zero inputs (a product-state or single-site input costs O(S), not
# O(S^2)). Measured crossover for inputs without zeros is S ≈ 32; inputs with
# zeros favour the column pass earlier. Correctness does not depend on the value.
const _DENSE_MATVEC_ROWWISE_MAX_S = 24

# (Fv)_i = sum_j F[i,j] v[j]. Column j is the image of generator j. Below
# `_DENSE_MATVEC_ROWWISE_MAX_S` this is a row-wise dot product; above it, the
# accumulation is column-wise instead (see the constant above) and skips a
# zero v[j] outright, so out[i] sums only a SUBSET of the row form's S
# nonnegative products, in a different order -- never more of them, and each
# one it does sum is the same product the row form would also add in, so
# `clifford_fast_dots(S, d)` still bounds the unreduced fast-tier accumulator
# exactly as it does for the row form.
# Precondition: v and F canonical (every entry already reduced mod d), and
# v !== prep.vout.
@inline function _matvec_prepared!(prep::PreparedDenseClifford, v::Vector{Int})
    F = prep.F; d = prep.d; S = 2 * prep.k; out = prep.vout
    if S <= _DENSE_MATVEC_ROWWISE_MAX_S
        if prep.fast
            @inbounds for i in 1:S
                s = 0
                for j in 1:S
                    s += F[i, j] * v[j]
                end
                out[i] = mod(s, d)
            end
        else
            @inbounds for i in 1:S
                s = 0
                for j in 1:S
                    s = add_mod(s, mul_mod(F[i, j], v[j], d), d)
                end
                out[i] = s
            end
        end
    else
        @inbounds for i in 1:S
            out[i] = 0
        end
        if prep.fast
            @inbounds for j in 1:S
                x = v[j]
                x == 0 && continue
                @simd for i in 1:S
                    out[i] += F[i, j] * x
                end
            end
            @inbounds for i in 1:S
                out[i] = mod(out[i], d)
            end
        else
            @inbounds for j in 1:S
                x = v[j]
                x == 0 && continue
                for i in 1:S
                    out[i] = add_mod(out[i], mul_mod(F[i, j], x, d), d)
                end
            end
        end
    end
    return out
end

@inline function _col_xdotz_dense(col::Vector{Int}, k::Int, d::Int, fast::Bool)
    if fast
        s = 0
        @inbounds for q in 1:k
            s += col[q] * col[k + q]
        end
        return mod(s, d)
    end
    s = 0
    @inbounds for q in 1:k
        s = add_mod(s, mul_mod(col[q], col[k + q], d), d)
    end
    return s
end

@inline _col_xdotz_prepared(prep::PreparedDenseClifford, v::Vector{Int}) =
    _col_xdotz_dense(v, prep.k, prep.d, prep.fast)

@inline function _dot_mod_dense(u::Vector{Int}, v::Vector{Int}, S::Int, M::Int,
                                fast::Bool)
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

# The closed-form odd-prime phase: φ = a·v + inv2 [ x_out·z_out − x·z − D·v ].
# Precondition: same as the tuple form `_phase_odd` -- v and vout canonical,
# every entry already reduced mod d.
@inline function _phase_odd_dense(v::Vector{Int}, vout::Vector{Int},
                                  a::Vector{Int}, D::Vector{Int}, k::Int,
                                  d::Int, inv2::Int, fast::Bool)
    S = 2k
    av    = _dot_mod_dense(a, v, S, d, fast)
    Dv    = _dot_mod_dense(D, v, S, d, fast)
    xz    = _col_xdotz_dense(v, k, d, fast)
    xzout = _col_xdotz_dense(vout, k, d, fast)
    t = sub_mod(sub_mod(xzout, xz, d), Dv, d)
    return add_mod(av, mul_mod(inv2, t, d), d)
end

# The vector-prefix qubit phase evaluator used for arbitrary dense supports,
# carrying the running Z prefix in a Vector rather than the named path's
# UInt64 bitmask. That bitmask is why `_phase_qubit` is bounded to K <= 64;
# using a vector here keeps the bound an internal property of the named path
# and never a public Clifford arity limit. `UInt64(1) << 64 == 0` in Julia, so
# a bitmask would silently drop coordinate 65 rather than erroring.
# Precondition: same as the tuple form `_phase_qubit` -- v canonical (every
# v[i] ∈ {0,1}) and every entry of F reduced mod 2.
@inline function _phase_qubit_dense(v::Vector{Int}, F::Matrix{Int},
                                    a::Vector{Int}, k::Int, zpref::Vector{Int})
    S = 2k
    @inbounds for q in 1:k
        zpref[q] = 0
    end
    phase = 0
    @inbounds for i in 1:S
        v[i] == 0 && continue
        phase = add_mod(phase, a[i], 4)
        # Under the precondition every F entry and zpref entry is 0 or 1, so
        # `F[q,i] & zpref[q]` is exactly the bit product and `c & 1` is the
        # parity x(F[:,i]) · zpref mod 2: the ordered-product cross term of
        # this image against the Z parts of the images already multiplied in.
        # c <= k, so the accumulator cannot overflow.
        c = 0
        @simd for q in 1:k
            c += F[q, i] & zpref[q]
        end
        # Reducing the prefix mod 2 is valid because it is multiplied by 2 in
        # phase modulus 4.
        phase = add_mod(phase, 2 * (c & 1), 4)
        @simd for q in 1:k
            zpref[q] ⊻= F[k + q, i]
        end
    end
    return phase
end

@inline function _phase(prep::PreparedDenseClifford, v::Vector{Int},
                        vout::Vector{Int})
    if prep.d == 2
        return _phase_qubit_dense(v, prep.F, prep.a, prep.k, prep.zpref)
    else
        return _phase_odd_dense(v, vout, prep.a, prep.D, prep.k, prep.d,
                                prep.inv2, prep.fast)
    end
end

# Bounds only: distinctness is proven by the stored operator's constructor.
@inline function _validate_clifford_targets(prep::PreparedDenseClifford, n::Int)
    @inbounds for i in 1:prep.k
        t = prep.targets[i]
        (1 <= t <= n) || throw(ArgumentError(
            "Clifford target $t is outside the register 1:$n."))
    end
    return nothing
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
generator contract are all preserved. Nonempty support clears
`tab.iscanonical`; empty support returns the tableau unchanged, including its
canonical flag.

# Arguments
- `tab::AbstractTableau`: tableau to update (`StabilizerTableau` or `DestabilizerTableau`).
- `g::AbstractClifford`: one of the named gates [`Fourier`](@ref),
  [`Phase`](@ref), [`Multiplier`](@ref), [`PauliGate`](@ref), [`SUM`](@ref),
  [`CPhase`](@ref) or [`SWAP`](@ref), or a stored [`CliffordOperator`](@ref).
  [`AbstractClifford`](@ref) tabulates the named gates' actions.

# Throws
`ArgumentError` if a target lies outside `1:tab.n`, if a gate parameter is
invalid at `tab.d` (for example a [`Multiplier`](@ref) coefficient congruent to
zero), or if a stored [`CliffordOperator`](@ref)'s dimension does not match
`tab.d`. Validation happens before any mutation.

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

See also [`AbstractClifford`](@ref), [`conjugate`](@ref), [`measure!`](@ref).
"""
function apply!(tab::AbstractTableau, g::AbstractClifford)
    _validate_clifford_targets(g, tab.n)
    prep = _prepare(g, tab.d, tab.inversemod, tab.storephase)
    return _apply_prepared!(tab, prep)
end

# A `PreparedClifford` carries its own dimension, phase regime and target list,
# so this entry point cannot assume `apply!` built it for this tableau. All
# three are re-checked here in O(1)/O(k^2) at function entry, never per column:
# a mismatched `d` would silently reduce exponents and phases against the wrong
# modulus, a mismatched `storephase` would write past the end of `stab`, and an
# out-of-range target would index out of bounds -- the column loop below runs
# under `@inbounds`, so none of the three would raise on its own.
# The dense backing re-checks target BOUNDS but trusts its constructor for
# distinctness, so a hand-built `PreparedDenseClifford` with duplicate targets
# is outside this boundary's guarantee. That is the accepted cost of keeping
# stored application off an O(k^2) prelude when `m` is small or zero.
function _apply_prepared!(tab::AbstractTableau, prep::AbstractPreparedClifford)
    prep.d == tab.d || throw(ArgumentError(
        "PreparedClifford was built for d=$(prep.d), but the tableau has " *
        "d=$(tab.d); rebuild the PreparedClifford for this tableau before " *
        "applying it."))
    prep.storephase == tab.storephase || throw(ArgumentError(
        "PreparedClifford was built with storephase=$(prep.storephase), but " *
        "the tableau has storephase=$(tab.storephase); rebuild the " *
        "PreparedClifford for this tableau before applying it."))
    length(prep.targets) == 0 && return tab
    _validate_clifford_targets(prep, tab.n)
    n = tab.n
    stab = tab.stab
    phase_row = 2n + 1
    @inbounds for j in 1:tab.m
        v = _gather_prepared!(prep, stab, n, j)
        vout = _matvec_prepared!(prep, v)
        if prep.storephase
            δ = _phase(prep, v, vout)
            stab[phase_row, j] = add_mod(stab[phase_row, j], δ, prep.p)
        end
        # The cache delta needs the OLD target entries, so it runs before the
        # scatter overwrites them.
        _before_clifford_scatter!(tab, prep, j, v, vout)
        _scatter_prepared!(prep, stab, n, j, vout)
    end
    _after_clifford!(tab, prep)
    tab.iscanonical = false
    return tab
end

########################################
# DestabilizerTableau hooks            #
########################################

@inline function _before_clifford_scatter!(tab::DestabilizerTableau,
                                           prep::AbstractPreparedClifford,
                                           j::Int, v, vout)
    d = prep.d
    old = _col_xdotz_prepared(prep, v)
    new = _col_xdotz_prepared(prep, vout)
    @inbounds tab.xdotz_cache[j] =
        add_mod(sub_mod(tab.xdotz_cache[j], old, d), new, d)
    return nothing
end

# The dual basis gets the SAME symplectic map and no phase work. This pass
# deliberately REUSES the main loop's `v`/`vout` buffers on the dense backing:
# it runs strictly after that loop, nothing from it stays live, and this is not
# an instance of the allocating-API scratch rule. Do not give it own buffers.
function _after_clifford!(tab::DestabilizerTableau, prep::AbstractPreparedClifford)
    n = tab.n
    destab = tab.destab
    @inbounds for j in 1:tab.m
        v = _gather_prepared!(prep, destab, n, j)
        vout = _matvec_prepared!(prep, v)
        _scatter_prepared!(prep, destab, n, j, vout)
    end
    return nothing
end

########################################
# Pauli conjugation                    #
########################################

# All support/shape checks are read-only and precede preparation/allocation.
# Register-size validation in _conjugate guarantees that 2n is representable.
function _validate_pauli(op::GeneralPauli, n::Int)
    len = length(op.xz)
    len == 2n || throw(ArgumentError(
        "GeneralPauli has length $len, expected 2n = $(2n)."))
    return nothing
end

function _validate_pauli(op::FewQuditPauli, n::Int)
    qudits, _, _, _ = _few_qudit_pauli_data(op)
    @inbounds for i in eachindex(qudits)
        q = qudits[i]
        (1 <= q <= n) || throw(ArgumentError(
            "Pauli qudit index $q is outside the register 1:$n."))
        for r in 1:(i - 1)
            qudits[r] == q && throw(ArgumentError(
                "Pauli has a repeated qudit index $q; that has no product interpretation here."))
        end
    end
    return nothing
end

# Internal precondition: dimension, register size, and support were validated.
# Normalize into independent storage; the caller's object is never mutated.
function _dense_pauli(op::GeneralPauli, n::Int, d::Int, p::Int)
    xz = Vector{Int}(undef, 2n)
    @inbounds for i in 1:(2n)
        xz[i] = mod(op.xz[i], d)
    end
    return GeneralPauli(xz, mod(op.phase, p))
end

function _dense_pauli(op::FewQuditPauli, n::Int, d::Int, p::Int)
    qudits, xs, zs, phase = _few_qudit_pauli_data(op)
    xz = zeros(Int, 2n)
    @inbounds for i in eachindex(qudits)
        q = qudits[i]
        xz[q] = mod(xs[i], d)
        xz[n + q] = mod(zs[i], d)
    end
    return GeneralPauli(xz, mod(phase, p))
end

# Named gates carry no dimension; a stored operator supplies its own. The
# second argument type is IDENTICAL on every method of this hook -- see the
# dispatch note on `_prepare` in `src/clifford_operator.jl`.
function _resolve_clifford_dimension(::AbstractClifford, d::Union{Int,Nothing})
    d === nothing && throw(ArgumentError(
        "conjugate with a named gate needs an explicit dimension: pass d = <prime>."))
    dd = d::Int
    Primes.isprime(dd) || throw(ArgumentError("Qudit dimension d must be a prime number."))
    return dd
end

# Standalone preparation never builds a d-1 lookup table. For a named gate this
# also completes modulus-dependent validation (e.g. a zero-residue Multiplier)
# BEFORE the caller allocates any dense result.
_prepare_for_conjugation(g::AbstractClifford, d::Int) =
    _prepare(g, d, JustInTimeInvMod(), true)

# Transformation only: every check and allocation has already happened.
function _conjugate_prepared!(out::GeneralPauli, prep::AbstractPreparedClifford,
                              n::Int)
    xz = out.xz
    v = _gather_vec_prepared!(prep, xz, n)
    vout = _matvec_prepared!(prep, v)
    δ = _phase(prep, v, vout)
    _scatter_vec_prepared!(prep, xz, n, vout)
    out.phase = add_mod(out.phase, δ, prep.p)
    return out
end

function _conjugate(g::AbstractClifford, op::AbstractPauli, n::Int,
                    d::Union{Int,Nothing})
    dd = _resolve_clifford_dimension(g, d)
    n >= 0 || throw(ArgumentError("Register size n must be nonnegative, got $n."))
    n <= typemax(Int) ÷ 2 || throw(ArgumentError("Register size 2n must fit in Int."))
    _validate_pauli(op, n)
    _validate_clifford_targets(g, n)
    p = phase_modulus(dd)
    prep = _prepare_for_conjugation(g, dd)
    out = _dense_pauli(op, n, dd, p)
    return _conjugate_prepared!(out, prep, n)
end

"""
    conjugate(g::AbstractClifford, op::GeneralPauli; d=nothing)
    conjugate(g::AbstractClifford, op::AbstractPauli, n::Int; d=nothing)

Return `U op U†` as a new [`GeneralPauli`](@ref), where `U` is the Clifford `g`.

This is the active conjugation, matching [`apply!`](@ref): applying `g` to a
state and conjugating an observable by `g` transform expectation values
consistently.

# Arguments
- `g::AbstractClifford`: the Clifford unitary — one of [`Fourier`](@ref),
  [`Phase`](@ref), [`Multiplier`](@ref), [`PauliGate`](@ref), [`SUM`](@ref),
  [`CPhase`](@ref) or [`SWAP`](@ref), or a stored [`CliffordOperator`](@ref).
  The named gates are tabulated in [`AbstractClifford`](@ref).
- `op`: the Pauli to conjugate. A dense [`GeneralPauli`](@ref) carries its own
  register size; a sparse `FewQuditPauli` does not, so `n` must be given.
- `n::Int`: register size, required for sparse Paulis and checked against a
  dense one.

# Keyword Arguments
- `d`: Prime qudit dimension, required for named gates. A stored operator
  supplies its own dimension; an explicit `d` must match it.

# Returns
A new `GeneralPauli` on the full `n`-qudit register, fully normalized. The
input is not mutated, and non-target coordinates are preserved.

# Throws
`ArgumentError` for a missing `d` with a named gate, a non-prime `d`, a `d`
that does not match a stored [`CliffordOperator`](@ref)'s own dimension, a
malformed dense length, a sparse index outside `1:n` or repeated, or a gate
target outside `1:n`.

# Examples
```julia
conjugate(Fourier(1), SinglePauli(1, 1, 0), 2; d = 3)   # X₁ ↦ Z₁
conjugate(SUM(1, 2), GeneralPauli([1, 0, 0, 0], 0); d = 2)
```

# Notes
Heisenberg evolution of an observable under state evolution by `U` is `U† P U`;
build it as `conjugate(inv(U), op)` from a [`CliffordOperator`](@ref).

See also [`AbstractClifford`](@ref), [`apply!`](@ref).
"""
function conjugate(g::AbstractClifford, op::GeneralPauli;
                   d::Union{Int,Nothing}=nothing)
    len = length(op.xz)
    iseven(len) || throw(ArgumentError(
        "GeneralPauli xz must have even length, got $len."))
    return _conjugate(g, op, len ÷ 2, d)
end

function conjugate(g::AbstractClifford, op::AbstractPauli, n::Int;
                   d::Union{Int,Nothing}=nothing)
    return _conjugate(g, op, n, d)
end
