##### uniformly random Clifford operators and stabilizer states #####
#
# One sampler serves both public entry points. `_sample_pairs!` builds a
# uniformly random symplectic matrix one symplectic pair at a time: `u` is a
# uniform nonzero vector of the space the earlier pairs leave, and `w` a
# uniform partner with <u, w> = 1. The columns not yet paired hold a basis of
# that remaining space in place, so the sampler needs four length-2k vectors
# and no second matrix. `random_clifford!` runs every step inside a stored
# operator, borrowing its scratch until the phases are drawn; `random_state!`
# stops after m steps in call-local scratch, then writes the first m Z images
# (stabilizers) and X images (destabilizers) into the tableau.
#
# The pairing is the package's <u, v> = x(u)·z(v) − z(u)·x(v), so X images fill
# columns 1:k and Z images columns k+1:2k, as in a stored operator's `F`.
#
# Every random residue comes from `_draw`, so tests can script, count or
# interrupt the stream with AbstractRNG subtypes of their own.

@inline _draw(rng::Random.AbstractRNG, d::Int) = rand(rng, 0:(d - 1))

# Local index j of the current basis at step i, with r = k - i + 1: the first r
# locals are X-half columns i:k, the rest Z-half columns k+i:2k.
@inline _basis_column(i::Int, j::Int, r::Int, k::Int) =
    j <= r ? i + j - 1 : k + i + j - r - 1

# <v, F[:, c]> mod d for a length-2k vector v, in the tier `fast` selects.
@inline function _pair_vec_col(v::Vector{Int}, F::Matrix{Int}, c::Int, k::Int,
                               d::Int, fast::Bool)
    if fast
        s = 0
        @inbounds for q in 1:k
            s += v[q] * F[k + q, c] - v[k + q] * F[q, c]
        end
        return mod(s, d)
    end
    s = 0
    @inbounds for q in 1:k
        s = add_mod(s, mul_mod(v[q], F[k + q, c], d), d)
        s = sub_mod(s, mul_mod(v[k + q], F[q, c], d), d)
    end
    return s
end

# v += c F[:, col] (mod d).
@inline function _addmul_col!(v::Vector{Int}, F::Matrix{Int}, col::Int, c::Int,
                              d::Int, fast::Bool)
    if fast
        # `v` is sampler scratch and the source a column of F, so the two
        # cannot alias; the fast-tier guard bounds every product formed.
        addmul_mod!(v, view(F, :, col), c, d)
    else
        @inbounds for q in eachindex(v)
            v[q] = add_mod(v[q], mul_mod(c, F[q, col], d), d)
        end
    end
    return v
end

# F[:, col] ← x + <w, x> u − <u, x> w for x = F[:, col]. This equals
# x − <x, w> u + <x, u> w, the projection onto the symplectic complement of
# span(u, w), given <u, w> = 1. Both pairings read x before either update.
@inline function _project_col!(F::Matrix{Int}, col::Int, u::Vector{Int},
                               w::Vector{Int}, k::Int, d::Int, fast::Bool)
    pw = _pair_vec_col(w, F, col, k, d, fast)
    pu = _pair_vec_col(u, F, col, k, d, fast)
    if fast
        x = view(F, :, col)
        # `x` is a column of F and `u`, `w` are sampler scratch: no aliasing.
        addmul_mod!(x, u, pw, d)
        submul_mod!(x, w, pu, d)
    else
        @inbounds for q in 1:(2k)
            F[q, col] = sub_mod(add_mod(F[q, col], mul_mod(pw, u[q], d), d),
                                mul_mod(pu, w[q], d), d)
        end
    end
    return F
end

@inline function _swap_columns!(F::Matrix{Int}, a::Int, b::Int)
    @inbounds for q in axes(F, 1)
        F[q, a], F[q, b] = F[q, b], F[q, a]
    end
    return F
end

# α[1:R] uniform over the nonzero coordinate vectors: the whole block is drawn
# again while it is zero, which keeps every nonzero block equally likely.
@inline function _draw_nonzero!(rng::Random.AbstractRNG, α::Vector{Int}, R::Int,
                                d::Int)
    while true
        nonzero = false
        for j in 1:R
            α[j] = _draw(rng, d)
            nonzero |= α[j] != 0
        end
        nonzero && return α
    end
end

# Run `steps` steps of the pair sampler on the 2k × 2k matrix F, whose columns
# must be a canonical basis of Z_d^(2k) (both callers pass the identity).
# Afterwards u_i = F[:, i] and w_i = F[:, k + i] for i ≤ steps satisfy
# <u_i, w_j> = δ_ij and <u_i, u_j> = <w_i, w_j> = 0, and the remaining columns
# are a basis of the symplectic complement of those pairs. `u`, `w`, `α`, `β`
# are distinct length-2k scratch vectors, written before they are read.
# `fast` is `clifford_fast_dots(2k, d)`, or `false` to force the safe tier.
function _sample_pairs!(rng::Random.AbstractRNG, F::Matrix{Int}, k::Int,
                        steps::Int, d::Int, fast::Bool, u::Vector{Int},
                        w::Vector{Int}, α::Vector{Int}, β::Vector{Int})
    S = 2k
    for i in 1:steps
        r = k - i + 1
        R = 2r
        # u = Σ α_j B_j over the current basis B, uniform on its nonzero span.
        _draw_nonzero!(rng, α, R, d)
        fill!(u, 0)
        for j in 1:R
            _addmul_col!(u, F, _basis_column(i, j, r, k), α[j], d, fast)
        end
        # w = Σ β_j B_j with <u, w> = 1. β first holds c_j = <u, B_j>; each
        # free coordinate replaces its c_j when drawn, and β[l] is solved for.
        l = 0
        for j in 1:R
            β[j] = _pair_vec_col(u, F, _basis_column(i, j, r, k), k, d, fast)
            l == 0 && β[j] != 0 && (l = j)
        end
        cl = β[l]
        acc = 0
        for j in 1:R
            j == l && continue
            cj = β[j]
            β[j] = _draw(rng, d)
            acc = add_mod(acc, mul_mod(cj, β[j], d), d)
        end
        β[l] = mul_mod(Base.invmod(cl, d), sub_mod(1, acc, d), d)
        fill!(w, 0)
        for j in 1:R
            _addmul_col!(w, F, _basis_column(i, j, r, k), β[j], d, fast)
        end
        # Drop B_s and B_t, where the coordinate minor at (s, t) is nonzero:
        # projecting the other 2r - 2 basis vectors then gives a basis of the
        # remaining space. Two swaps move B_s and B_t into columns i and k + i,
        # in either order; both are overwritten below.
        s = 1
        while α[s] == 0
            s += 1
        end
        t = 1
        while sub_mod(mul_mod(α[s], β[t], d), mul_mod(α[t], β[s], d), d) == 0
            t += 1
        end
        a = _basis_column(i, s, r, k)
        b = _basis_column(i, t, r, k)
        b == i && ((a, b) = (b, a))
        a != i && _swap_columns!(F, a, i)
        b != k + i && _swap_columns!(F, b, k + i)
        for c in (i + 1):k
            _project_col!(F, c, u, w, k, d, fast)
        end
        for c in (k + i + 1):S
            _project_col!(F, c, u, w, k, d, fast)
        end
        @inbounds for q in 1:S
            F[q, i] = u[q]
            F[q, k + i] = w[q]
        end
    end
    return F
end

@inline function _set_identity!(F::Matrix{Int})
    fill!(F, 0)
    @inbounds for i in axes(F, 1)
        F[i, i] = 1
    end
    return F
end

#################################
# Uniformly random operators    #
#################################

"""
    random_clifford([rng::AbstractRNG,] d::Int, targets::AbstractVector{<:Integer}) -> CliffordOperator

A uniformly random Clifford unitary on the ordered support `targets` at prime
dimension `d`, returned as a [`CliffordOperator`](@ref).

The distribution is uniform over the Clifford group modulo global phase: every
symplectic matrix `F` is equally likely, and so is each of its `d^(2k)` valid
raw phase vectors, where `k = length(targets)`. At `d = 2` the valid phases are
the ones that keep every generator image Hermitian. Every coordinate is drawn
with `rand(rng, 0:(d - 1))`, so the result is exactly uniform for an ideal
RNG. The RNG-free form uses `Random.default_rng()`. Sampling costs `O(k³)`
expected time.

# Arguments
- `rng::AbstractRNG`: source of randomness.
- `d::Int`: prime qudit dimension.
- `targets`: ordered, distinct, positive qudit indices, copied into the
  operator. Pass `1:k` for the first `k` qudits.

# Throws
`ArgumentError`, before any draw, for a non-prime `d`, offset-indexed targets,
a target count or target value that does not fit in `Int`, nonpositive or
repeated targets, or an integer in place of the target vector.

# Notes
A fixed seed reproduces the result for the same RNG type, Julia version and
package version. The draw stream is not a cross-version guarantee.

# Examples
```julia
using Random
U = random_clifford(Xoshiro(1), 3, 1:2)    # a random two-qutrit Clifford
apply!(tab, U)
```

See also [`random_clifford!`](@ref), [`random_state!`](@ref).
"""
function random_clifford(rng::Random.AbstractRNG, d::Int,
                         targets::AbstractVector{<:Integer})
    Primes.isprime(d) || throw(ArgumentError("Qudit dimension d must be a prime number."))
    Base.require_one_based_indexing(targets)
    k = _target_count(targets)
    S = _clifford_size(k)
    t = _copy_targets(targets, k)
    # The identity is a valid operator, as the owned boundary requires; the
    # refill replaces its action. NEVER invmod(2, 2), which is undefined.
    inv2 = d == 2 ? 0 : Base.invmod(2, d)
    U = _owned_clifford_operator(d, t, _set_identity!(Matrix{Int}(undef, S, S)),
                                 zeros(Int, S), zeros(Int, S), inv2,
                                 clifford_fast_dots(S, d), zeros(Int, S),
                                 zeros(Int, S), zeros(Int, k))
    return random_clifford!(rng, U)
end

random_clifford(d::Int, targets::AbstractVector{<:Integer}) =
    random_clifford(Random.default_rng(), d, targets)

# A bare integer could mean "on qudit k" or "on k qudits", so neither is guessed.
random_clifford(::Random.AbstractRNG, ::Int, k::Integer) = _throw_integer_targets(k)
random_clifford(::Int, k::Integer) = _throw_integer_targets(k)

@noinline _throw_integer_targets(k::Integer) = throw(ArgumentError(
    "random_clifford takes a vector of target qudits, got the integer $k. " *
    "Pass 1:$k for the first $k qudits, or [$k] for qudit $k alone."))

"""
    random_clifford!([rng::AbstractRNG,] U::CliffordOperator) -> U

Refill `U` in place with a new uniformly random Clifford on the same dimension
and ordered support, and return `U`.

The result depends only on the draws, never on `U`'s previous action, so any
operator can be refilled, including one built with `check=false`. `U` keeps its
dimension, its targets and every one of its arrays; only `F`, the raw phases
and the derived cache change. A warmed refill with a standard RNG allocates
nothing, which suits random circuits: keep one operator per support and refill
it for every layer. This is the one package operation that changes an
operator's action in place.

# Notes
- If the RNG throws during a refill, `U` is unusable until a later refill
  succeeds; nothing is rolled back.
- A refill changes `==` and `hash`, so do not refill an operator that is a
  dictionary key.
- One operator must not be refilled or applied from several tasks at once, as
  for [`apply!`](@ref); use independent copies instead.

# Examples
```julia
using Random
rng = Xoshiro(7)
U = random_clifford(rng, 2, [3, 4])
for layer in 1:10
    apply!(tab, random_clifford!(rng, U))
end
```

See also [`random_clifford`](@ref).
"""
function random_clifford!(rng::Random.AbstractRNG, U::CliffordOperator)
    k = length(U.targets)
    k == 0 && return U
    d = U.d
    F = _set_identity!(U.F)
    # `v`, `vout`, `a` and `image_xdotz` are the sampler's scratch until the
    # cache and the phases are rewritten below.
    _sample_pairs!(rng, F, k, k, d, U.fast, U.v, U.vout, U.a, U.image_xdotz)
    D = _image_xdotz_dense!(U.image_xdotz, F, k, d, U.fast)
    a = U.a
    if d == 2
        # The two Hermitian phases a ≡ x·z (mod 2). D[i] is 0 or 1, so the sum
        # is already reduced mod 4.
        for i in 1:(2k)
            a[i] = D[i] + 2 * _draw(rng, 2)
        end
    else
        for i in 1:(2k)
            a[i] = _draw(rng, d)
        end
    end
    return U
end

random_clifford!(U::CliffordOperator) = random_clifford!(Random.default_rng(), U)

###########################
# Uniformly random states #
###########################

"""
    random_state!([rng::AbstractRNG,] tab::AbstractTableau; m::Int = tab.n) -> tab

Replace the state of `tab` with a uniformly random stabilizer state that has
`m` independent commuting generators, and return `tab`.

With `m == tab.n` the result is a uniformly random pure stabilizer state. With
`0 < m < n` it is the normalized projector onto a uniformly random stabilizer
code: the stabilizer group has `d^m` elements and the density matrix has rank
`d^(n - m)`. `m == 0` gives the maximally mixed state and draws nothing. The
state is the image of `|0⟩⟨0|^⊗m ⊗ I/d^(n-m)` under a uniformly random
Clifford, sampled directly rather than by building that Clifford.

With `storephase=false` the result is uniform over `m`-dimensional isotropic
subspaces, and no phases are drawn. A [`DestabilizerTableau`](@ref) also gets
its dual basis and cache, with no separate reconstruction.

# Keyword Arguments
- `m::Int = tab.n`: number of generators, `0 ≤ m ≤ tab.n`.

# Throws
`ArgumentError` unless `0 ≤ m ≤ tab.n`, before any draw or mutation.

# Notes
- Keeps `d`, `n`, `storephase`, the inversion strategy and every array of
  `tab`, and clears `tab.iscanonical`.
- For `m > 0`, each call allocates a `2n × 2n` scratch matrix and four
  length-`2n` vectors, and costs `O(n²m)`. Every draw happens before `tab` is
  modified, so an exception from the RNG leaves it unchanged, although the RNG
  has advanced. An interrupt that arrives during the final copy is outside
  this guarantee.
- A fixed seed reproduces the state for the same RNG type, Julia version and
  package version.

# Examples
```julia
using Random
tab = DestabilizerTableau(3, 4)
random_state!(Xoshiro(1), tab)            # a random pure state
random_state!(Xoshiro(2), tab; m = 2)     # a random stabilizer code
```

See also [`random_clifford`](@ref), [`reset!`](@ref).
"""
function random_state!(rng::Random.AbstractRNG, tab::AbstractTableau;
                       m::Int = tab.n)
    n = tab.n
    0 <= m <= n || throw(ArgumentError("m must satisfy 0 ≤ m ≤ n = $n, got m = $m."))
    m == 0 && return reset!(tab; state = :mixed)
    d = tab.d
    # The scratch is larger than the tableau, so check its own element and byte
    # counts: Julia 1.10's array constructor accepts sizes whose element count
    # overflows.
    S = _clifford_size(n)
    G = _set_identity!(Matrix{Int}(undef, S, S))
    u, w = Vector{Int}(undef, S), Vector{Int}(undef, S)
    α, β = Vector{Int}(undef, S), Vector{Int}(undef, S)
    fast = clifford_fast_dots(S, d)
    _sample_pairs!(rng, G, n, m, d, fast, u, w, α, β)
    # Stage the commit in the sampler's spent vectors: β[j] is stabilizer j's
    # x·z, α[j] its raw phase. These are the last draws.
    for j in 1:m
        β[j] = _col_xdotz_matrix(G, n + j, n, d, fast)
    end
    if tab.storephase
        for j in 1:m
            α[j] = d == 2 ? β[j] + 2 * _draw(rng, 2) : _draw(rng, d)
        end
    end
    # Commit. Stabilizer j is the Z image G[:, n + j] and its dual the X image
    # G[:, j], so <G[:, j], G[:, n + l]> = δ_jl without any repair pass.
    stab = tab.stab
    fill!(stab, 0)
    @inbounds for j in 1:m
        for q in 1:S
            stab[q, j] = G[q, n + j]
        end
        tab.storephase && (stab[S + 1, j] = α[j])
    end
    _write_random_duals!(tab, G, m, β)
    tab.m = m
    tab.iscanonical = false
    return tab
end

random_state!(tab::AbstractTableau; m::Int = tab.n) =
    random_state!(Random.default_rng(), tab; m = m)

# Dual-basis bookkeeping for `random_state!`: nothing for a plain tableau.
_write_random_duals!(::AbstractTableau, G::Matrix{Int}, m::Int, D::Vector{Int}) = nothing

# The first m X images become the destabilizers, and D[1:m] the stabilizers'
# x·z values; the unused columns and cache entries are zeroed.
function _write_random_duals!(tab::DestabilizerTableau, G::Matrix{Int}, m::Int,
                              D::Vector{Int})
    destab = tab.destab
    cache = tab.xdotz_cache
    fill!(destab, 0)
    fill!(cache, 0)
    @inbounds for j in 1:m
        for q in axes(destab, 1)
            destab[q, j] = G[q, j]
        end
        cache[j] = D[j]
    end
    return nothing
end
