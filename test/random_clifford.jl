using QuditClifford
using Test
using Random
using LinearAlgebra

const QC = QuditClifford

# Every top-level name in this file starts with `rc_` (functions) or `RC`
# (types): runtests.jl includes all test files into one module.

#############################
# Scripted and counted draws #
#############################

# Replays fixed residues. Each draw is checked against its requested range, and
# drawing past the end is an error, so a test can assert exact consumption.
mutable struct RCScript <: AbstractRNG
    vals::Vector{Int}
    pos::Int
end
RCScript(vals::AbstractVector{<:Integer}) = RCScript(collect(Int, vals), 0)
function QC._draw(s::RCScript, d::Int)
    s.pos < length(s.vals) || error("RCScript exhausted after $(s.pos) draws")
    s.pos += 1
    v = s.vals[s.pos]
    0 <= v < d || error("RCScript draw $v is outside 0:$(d - 1)")
    return v
end
rc_consumed_all(s::RCScript) = s.pos == length(s.vals)

# Counts the draws it forwards to another source.
mutable struct RCCounting{R<:AbstractRNG} <: AbstractRNG
    inner::R
    count::Int
end
RCCounting(inner::AbstractRNG) = RCCounting(inner, 0)
QC._draw(c::RCCounting, d::Int) = (c.count += 1; QC._draw(c.inner, d))

# Forwards `allowed` draws, then throws on the next one.
struct RCInjectedFailure <: Exception end
mutable struct RCThrowing{R<:AbstractRNG} <: AbstractRNG
    inner::R
    allowed::Int
    count::Int
end
RCThrowing(inner::AbstractRNG, allowed::Int) = RCThrowing(inner, allowed, 0)
function QC._draw(t::RCThrowing, d::Int)
    t.count == t.allowed && throw(RCInjectedFailure())
    t.count += 1
    return QC._draw(t.inner, d)
end

##########################
# Independent exact oracles #
##########################

# <x, y> = x(x)·z(y) − z(x)·x(y) over Z_d, from the residues of the entries.
# Every partial sum is bounded by k(d - 1)^2, so plain Int arithmetic is used
# only when that bound fits this host's word; otherwise BigInt.
function rc_pair(x, y, k::Int, d::Integer)
    r(v) = mod(v, d)
    s = if k == 0 || d - 1 <= isqrt(typemax(Int) ÷ k)
        sum(r(x[q]) * r(y[k + q]) - r(x[k + q]) * r(y[q]) for q in 1:k; init = 0)
    else
        sum(big(r(x[q])) * r(y[k + q]) - big(r(x[k + q])) * r(y[q]) for q in 1:k;
            init = big(0))
    end
    return Int(mod(s, d))
end

# Ω's entry (i, j): the pairing a symplectic frame's columns i and j must have.
rc_omega(i::Int, j::Int, k::Int, d::Integer) =
    (i <= k && j == i + k) ? 1 : (j <= k && i == j + k) ? Int(d - 1) : 0

function rc_is_symplectic(F, k::Int, d::Integer)
    for i in 1:2k, j in 1:2k
        rc_pair(view(F, :, i), view(F, :, j), k, d) == rc_omega(i, j, k, d) ||
            return false
    end
    return true
end

# Reduced row echelon form of the rows of M over Z_d, and its rank. Plain Int
# arithmetic, exact while (d - 1)^2 + (d - 1) fits this host's word.
function rc_rref(M::AbstractMatrix, d::Int)
    @assert d - 1 <= isqrt(typemax(Int) ÷ 2)
    A = mod.(Matrix{Int}(M), d)
    rows, cols = size(A)
    r = 0
    for c in 1:cols
        r == rows && break
        p = findfirst(i -> A[i, c] != 0, (r + 1):rows)
        p === nothing && continue
        p += r
        r += 1
        A[r, :], A[p, :] = A[p, :], A[r, :]
        A[r, :] = mod.(A[r, :] .* invmod(A[r, c], d), d)
        for i in 1:rows
            i == r || (A[i, :] = mod.(A[i, :] .- A[i, c] .* A[r, :], d))
        end
    end
    return A, r
end

# After `steps` steps: the pairs obey <u_i, w_j> = δ_ij and <u_i, u_j> =
# <w_i, w_j> = 0, and the other columns are a basis orthogonal to every pair.
function rc_prefix_ok(F, k::Int, steps::Int, d::Int)
    pairs = [1:steps; (k + 1):(k + steps)]
    rest = [(steps + 1):k; (k + steps + 1):(2k)]
    for p in pairs, q in pairs
        rc_pair(view(F, :, p), view(F, :, q), k, d) == rc_omega(p, q, k, d) ||
            return false
    end
    for c in rest, p in pairs
        rc_pair(view(F, :, c), view(F, :, p), k, d) == 0 || return false
    end
    return last(rc_rref(permutedims(F[:, rest]), d)) == length(rest)
end

# The first failed check of one enumerated case, tagged with that case, or
# nothing. Exhaustive loops keep the first failure and stop, so a single
# `@test failure === nothing` names the script and the property that broke.
function rc_failed(case::NamedTuple, checks::Pair{Symbol,Bool}...)
    for (name, ok) in checks
        ok || return (; case..., failed = name)
    end
    return nothing
end

#######################
# Enumerating scripts #
#######################

# The α blocks of a rejection-free script of `steps` steps at support k, and
# the script's length: step i, with r = k - i + 1, reads 2r α coordinates and
# then 2r - 1 free β coordinates.
function rc_alpha_blocks(k::Int, steps::Int)
    blocks = UnitRange{Int}[]
    pos = 1
    for i in 1:steps
        r = k - i + 1
        push!(blocks, pos:(pos + 2r - 1))
        pos += 4r - 1
    end
    return blocks, pos - 1
end

# Every script of length L over 0:(d - 1) whose α blocks are all nonzero.
function rc_scripts(d::Int, blocks, L::Int)
    out = Vector{Vector{Int}}()
    for idx in 0:(d^L - 1)
        v = digits(idx; base = d, pad = L)
        all(b -> any(!iszero, view(v, b)), blocks) && push!(out, v)
    end
    return out
end

# Host-dependent large primes: at RC_D1 the fast tier accepts k = 1 and rejects
# k = 2; RC_D2 is the largest prime the safe tier handles on this word size.
const RC_D1, RC_D2 = Sys.WORD_SIZE == 64 ? (2147483647, 9223372036854775783) :
                                           (32749, 2147483647)

function rc_identity!(F::Matrix{Int})
    fill!(F, 0)
    for i in axes(F, 1)
        F[i, i] = 1
    end
    return F
end

# Run the pair sampler on the identity. The scratch starts as garbage, so a
# read before a write shows up as a wrong or non-canonical frame.
function rc_sample_pairs(src::AbstractRNG, d::Int, k::Int, steps::Int = k;
                         fast::Bool = QC.clifford_fast_dots(2k, d))
    S = 2k
    F = rc_identity!(Matrix{Int}(undef, S, S))
    QC._sample_pairs!(src, F, k, steps, d, fast, fill(-1, S), fill(-1, S),
                      fill(-1, S), fill(-1, S))
    return F
end

################
# Pair sampler #
################

@testset "The pairing oracle is exact where Int sums wrap" begin
    # k(d - 1)^2 exceeds typemax(Int) at RC_D1 and k = 3 on either word size;
    # the exact pairing is k(d - 1)^2 ≡ k (mod d).
    d, k = RC_D1, 3
    x = [fill(d - 1, k); zeros(Int, k)]
    y = [zeros(Int, k); fill(d - 1, k)]
    @test mod(sum(x[q] * y[k + q] for q in 1:k), d) != k  # the unguarded sum wraps
    @test rc_pair(x, y, k, d) == k
    @test rc_pair(y, x, k, d) == d - k
end

@testset "Rejection-free pair scripts have m(4k - 2m + 1) draws" begin
    for k in 0:6, m in 0:k
        @test last(rc_alpha_blocks(k, m)) == m * (4k - 2m + 1)
    end
end

@testset "Pair sampler reaches every symplectic frame exactly once" begin
    for (d, k, order) in ((2, 1, 6), (3, 1, 24), (2, 2, 720), (3, 2, 51840))
        blocks, L = rc_alpha_blocks(k, k)
        scripts = rc_scripts(d, blocks, L)
        @test length(scripts) == order
        frames = Set{Matrix{Int}}()
        failure = nothing
        for (index, script) in enumerate(scripts)
            src = RCScript(script)
            F = rc_sample_pairs(src, d, k)
            push!(frames, F)
            failure = rc_failed((; d, k, index),
                :consumed => rc_consumed_all(src),
                :symplectic => rc_is_symplectic(F, k, d),
                :canonical => all(x -> 0 <= x < d, F),
                # The forced-safe tier makes the same draws and the same frame.
                :tiers => rc_sample_pairs(RCScript(script), d, k; fast = false) == F)
            # Every prefix, steps = 0 included, is a partial symplectic frame.
            for steps in 0:k
                failure === nothing || break
                prefix = RCScript(script[1:last(rc_alpha_blocks(k, steps))])
                Fp = rc_sample_pairs(prefix, d, k, steps)
                failure = rc_failed((; d, k, index, steps),
                    :consumed => rc_consumed_all(prefix),
                    :prefix => rc_prefix_ok(Fp, k, steps, d))
            end
            failure === nothing || break
        end
        @test failure === nothing
        @test length(frames) == order
    end
    # No support: nothing to sample and nothing drawn.
    src = RCScript(Int[])
    @test size(rc_sample_pairs(src, 3, 0)) == (0, 0)
    @test rc_consumed_all(src)
end

@testset "Zero α blocks are redrawn at exactly their own cost" begin
    d, k = 3, 2
    blocks, L = rc_alpha_blocks(k, k)
    for script in rc_scripts(d, blocks, L)[1:101:end]
        F = rc_sample_pairs(RCScript(script), d, k)
        # One rejected block before step 1 (2r = 4 draws) ...
        once = RCScript([zeros(Int, 4); script])
        @test rc_sample_pairs(once, d, k) == F && once.pos == L + 4
        # ... and two before step 1, three before step 2 (2r = 2 draws each).
        # Step 1 reads 4r - 1 = 7 draws.
        several = RCScript([zeros(Int, 8); script[1:7]; zeros(Int, 6); script[8:end]])
        @test rc_sample_pairs(several, d, k) == F && several.pos == L + 14
    end
end

@testset "Both arithmetic tiers draw the same frames at larger d" begin
    for d in (5, 7), k in (2, 3), seed in 1:10
        F = rc_sample_pairs(Xoshiro(seed), d, k)
        @test F == rc_sample_pairs(Xoshiro(seed), d, k; fast = false)
        @test rc_is_symplectic(F, k, d)
    end
end

function rc_pairs_allocations(rng::AbstractRNG, d::Int, k::Int, fast::Bool)
    S = 2k
    F = zeros(Int, S, S)
    u, w, α, β = zeros(Int, S), zeros(Int, S), zeros(Int, S), zeros(Int, S)
    for _ in 1:3
        QC._sample_pairs!(rng, rc_identity!(F), k, k, d, fast, u, w, α, β)
    end
    rc_identity!(F)
    return @allocated QC._sample_pairs!(rng, F, k, k, d, fast, u, w, α, β)
end

@testset "The pair sampler allocates nothing" begin
    for d in (2, 3, 5), k in (1, 2, 8, 33), fast in (true, false)
        @test rc_pairs_allocations(Xoshiro(k), d, k, fast) == 0
    end
    # A modulus only the safe tier can handle at k = 2.
    @test !QC.clifford_fast_dots(4, RC_D1)
    @test rc_pairs_allocations(Xoshiro(1), RC_D1, 2, false) == 0
end
