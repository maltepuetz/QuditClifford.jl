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
