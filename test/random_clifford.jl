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

#############################
# Uniformly random operators #
#############################

# A rejection-free script: every α block a unit vector, every free β
# coordinate and every trailing phase draw zero.
function rc_simple_script(k::Int, steps::Int, nphases::Int)
    blocks, L = rc_alpha_blocks(k, steps)
    script = zeros(Int, L + nphases)
    for b in blocks
        script[first(b)] = 1
    end
    return script
end

rc_identity_operator(d::Int, k::Int) =
    CliffordOperator(d, 1:k, rc_identity!(Matrix{Int}(undef, 2k, 2k)), zeros(Int, 2k))

# Every one-qudit Clifford with its valid phases: all of SL(2, Z_d), which is
# Sp(2, Z_d), with every raw phase pair at odd d and, at d = 2, the two phases
# per image with a ≡ x·z (mod 2).
function rc_one_qudit_cliffords(d::Int)
    out = Set{Tuple{Matrix{Int},Vector{Int}}}()
    for p in 0:(d - 1), q in 0:(d - 1), r in 0:(d - 1), s in 0:(d - 1)
        mod(p * s - q * r, d) == 1 || continue
        F = [p q; r s]
        D = [mod(F[1, j] * F[2, j], d) for j in 1:2]
        phases = d == 2 ? [[D[1] + 2x, D[2] + 2y] for x in 0:1 for y in 0:1] :
                          [[x, y] for x in 0:(d - 1) for y in 0:(d - 1)]
        for a in phases
            push!(out, (F, a))
        end
    end
    return out
end

# One-based data behind zero-based axes.
# A representable length whose operator could never be stored; reading an
# entry fails, so a rejection proves the targets were not traversed.
struct RCUnreadableHugeTargets <: AbstractVector{Int} end
Base.size(::RCUnreadableHugeTargets) = (typemax(Int) ÷ 2,)
Base.getindex(::RCUnreadableHugeTargets, ::Int) = error("targets traversed before rejection")

struct RCOffsetTargets <: AbstractVector{Int}
    data::Vector{Int}
end
Base.size(v::RCOffsetTargets) = size(v.data)
Base.axes(v::RCOffsetTargets) = (0:(length(v.data) - 1),)
Base.getindex(v::RCOffsetTargets, i::Int) = v.data[i + 1]

@testset "Refills reach every one-qudit Clifford exactly once" begin
    for (d, count) in ((2, 24), (3, 216))
        blocks, L = rc_alpha_blocks(1, 1)
        scripts = rc_scripts(d, blocks, L + 2)
        @test length(scripts) == count
        U = rc_identity_operator(d, 1)
        seen = Set{Tuple{Matrix{Int},Vector{Int}}}()
        failure = nothing
        for (index, script) in enumerate(scripts)
            src = RCScript(script)
            random_clifford!(src, U)
            push!(seen, (copy(U.F), copy(U.a)))
            failure = rc_failed((; d, index),
                :consumed => rc_consumed_all(src),
                :cached => U.image_xdotz == QC._image_xdotz_dense(U.F, 1, d, U.fast))
            failure === nothing || break
        end
        @test failure === nothing
        @test seen == rc_one_qudit_cliffords(d)
    end
end

@testset "random_clifford validates before drawing" begin
    # Any draw from an empty script is an ErrorException, so an ArgumentError
    # here proves validation came first.
    none = RCScript(Int[])
    @test_throws ArgumentError random_clifford(none, 4, [1])
    @test_throws ArgumentError random_clifford(none, 1, [1])
    @test_throws ArgumentError random_clifford(none, 3, [1, 1])
    @test_throws ArgumentError random_clifford(none, 3, [0])
    @test_throws ArgumentError random_clifford(none, 3, [2, -1])
    @test_throws ArgumentError random_clifford(none, 3, [big(typemax(Int)) + 1])
    @test_throws ArgumentError random_clifford(none, 3, RCOffsetTargets([1]))
    @test_throws ArgumentError random_clifford(none, 3, RCUnreadableHugeTargets())
    # Huge lazy ranges fail on their count, without being traversed.
    for huge in (1:typemax(Int), typemin(Int):typemax(Int), UInt(0):typemax(UInt),
                 big(1):(big(typemax(Int)) + 1))
        @test_throws ArgumentError random_clifford(none, 3, huge)
    end
    # A bare integer is not an arity shorthand.
    err = try
        random_clifford(none, 3, 2)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("1:2", err.msg) && occursin("[2]", err.msg)
    @test_throws ArgumentError random_clifford(3, 2)
    @test none.pos == 0
end

@testset "Range targets behave like the equivalent vectors" begin
    for (targets, dense) in ((big(2):big(3), [2, 3]), (UInt(2):UInt(3), [2, 3]),
                             (Int128(2):Int128(3), [2, 3]), (5:-2:1, [5, 3, 1]),
                             (Int128(5):Int128(-2):Int128(1), [5, 3, 1]),
                             (big(1):big(0), Int[]), (UInt(1):UInt(0), Int[]))
        a_src, b_src = RCCounting(Xoshiro(3)), RCCounting(Xoshiro(3))
        a = random_clifford(a_src, 3, targets)
        b = random_clifford(b_src, 3, dense)
        @test a == b && a.targets == dense && a_src.count == b_src.count
    end
end

@testset "Refills make 2k^2 + 3k draws without rejection" begin
    for d in (2, 3), k in 0:4
        script = rc_simple_script(k, k, 2k)
        @test length(script) == 2k^2 + 3k
        src = RCScript(script)
        U = random_clifford(src, d, 1:k)
        @test rc_consumed_all(src)
        again = RCScript(script)
        @test random_clifford!(again, U) === U && rc_consumed_all(again)
    end
end

@testset "A refill keeps every array and forgets the old action" begin
    fields = (:targets, :F, :a, :image_xdotz, :v, :vout, :zpref)
    for d in (2, 3, 5), k in (1, 3)
        targets = collect(k:-1:1) .+ 1
        # Neither symplectic nor, at d = 2, Hermitian: check=false keeps it.
        U = CliffordOperator(d, targets, fill(1, 2k, 2k), fill(1, 2k); check = false)
        arrays = map(f -> getfield(U, f), fields)
        fill!(U.v, -9); fill!(U.vout, -9); fill!(U.zpref, -9)
        @test random_clifford!(Xoshiro(11), U) === U
        @test all(map(f -> getfield(U, f), fields) .=== arrays)
        @test U.d == d && U.targets == targets
        @test U.fast == QC.clifford_fast_dots(2k, d)
        @test U.inv2 == (d == 2 ? 0 : invmod(2, d))
        @test U.image_xdotz == QC._image_xdotz_dense(U.F, k, d, U.fast)
        # The checked constructor accepts the result: symplectic, and at
        # d = 2 every image Hermitian.
        @test CliffordOperator(d, targets, U.F, U.a) == U
        # Old action and scratch played no part.
        V = CliffordOperator(d, targets, rc_identity!(Matrix{Int}(undef, 2k, 2k)),
                             zeros(Int, 2k))
        @test random_clifford!(Xoshiro(11), V) == U
    end
end

@testset "A sampled operator owns independent arrays" begin
    targets = [4, 1, 3]
    U = random_clifford(Xoshiro(8), 3, targets)
    @test U.targets == targets && U.targets !== targets
    targets[1] = 99
    @test U.targets == [4, 1, 3]
    fields = (:targets, :F, :a, :image_xdotz, :v, :vout, :zpref)
    arrays = map(f -> getfield(U, f), fields)
    @test all(!Base.mightalias(arrays[i], arrays[j])
              for i in eachindex(arrays) for j in (i + 1):length(arrays))
    V = random_clifford(Xoshiro(8), 3, [4, 1, 3])
    @test U == V
    @test all(!Base.mightalias(getfield(U, f), getfield(V, g)) for f in fields, g in fields)
end

@testset "A failed refill is recovered by the next one" begin
    for d in (2, 3), k in (1, 2)
        script = rc_simple_script(k, k, 2k)
        reference = random_clifford!(RCScript(script), rc_identity_operator(d, k))
        for allowed in 0:(length(script) - 1)
            U = random_clifford(Xoshiro(5), d, 1:k)
            failing = RCThrowing(RCScript(script), allowed)
            @test_throws RCInjectedFailure random_clifford!(failing, U)
            @test failing.count == allowed
            random_clifford!(RCScript(script), U)
            @test U == reference && U.image_xdotz == reference.image_xdotz
        end
    end
end

@testset "Seeds reproduce operators; RNG-free forms use the default RNG" begin
    for d in (2, 3)
        @test random_clifford(Xoshiro(21), d, 1:3) == random_clifford(Xoshiro(21), d, 1:3)
        @test random_clifford(MersenneTwister(21), d, 1:3) ==
              random_clifford(MersenneTwister(21), d, 1:3)
        Random.seed!(99)
        a = random_clifford(d, 1:3)
        Random.seed!(99)
        @test random_clifford(Random.default_rng(), d, 1:3) == a
        U, V = random_clifford(Xoshiro(1), d, 1:3), random_clifford(Xoshiro(1), d, 1:3)
        Random.seed!(7)
        random_clifford!(U)
        Random.seed!(7)
        random_clifford!(Random.default_rng(), V)
        @test U == V
    end
    # Empty support draws nothing, in either form.
    counting = RCCounting(Xoshiro(1))
    E = random_clifford(counting, 3, Int[])
    @test E.targets == Int[] && size(E.F) == (0, 0) && isempty(E.a)
    @test random_clifford!(counting, E) === E
    @test counting.count == 0
end

@testset "Sampled operators are valid and invertible ($(nameof(typeof(rng))))" for rng in
        (Xoshiro(2026), MersenneTwister(2026))
    for d in (2, 3, 5), k in (3, 5, 8)
        U = random_clifford(rng, d, 1:k)
        @test CliffordOperator(d, U.targets, U.F, U.a) == U
        E = rc_identity_operator(d, k)
        @test inv(U) ∘ U == E
        @test U ∘ inv(U) == E
        for TT in (StabilizerTableau, DestabilizerTableau)
            tab = TT(d, k + 2; state = :ghz)
            before = deepcopy(tab)
            apply!(tab, U)
            apply!(tab, inv(U))
            @test tab.stab == before.stab
            if TT === DestabilizerTableau
                @test tab.destab == before.destab
                @test tab.xdotz_cache == before.xdotz_cache
            end
        end
    end
    # An unsorted, noncontiguous support keeps its order.
    U = random_clifford(rng, 3, [7, 2, 5])
    @test U.targets == [7, 2, 5]
    tab = DestabilizerTableau(3, 8; state = :ghz)
    before = deepcopy(tab)
    apply!(tab, U)
    apply!(tab, inv(U))
    @test tab.stab == before.stab && tab.destab == before.destab
end

@testset "A k = 65 qubit sample round-trips through a 65-qubit tableau" begin
    # Dense images drive the vector-prefix qubit evaluator past coordinate 64.
    U = random_clifford(Xoshiro(65), 2, 1:65)
    @test CliffordOperator(2, U.targets, U.F, U.a) == U
    for TT in (StabilizerTableau, DestabilizerTableau)
        tab = TT(2, 65; state = :ghz)
        before = deepcopy(tab)
        apply!(tab, U)
        @test is_pure(tab; verify = true)
        apply!(tab, inv(U))
        @test tab.stab == before.stab
    end
end

@testset "Large moduli sample exact Cliffords in both tiers" begin
    @test QC.clifford_fast_dots(2, RC_D1) && !QC.clifford_fast_dots(4, RC_D1)
    for d in (RC_D1, RC_D2), k in (1, 2, 3)
        U = random_clifford(Xoshiro(k), d, 1:k)
        @test U.fast == QC.clifford_fast_dots(2k, d)
        @test all(x -> 0 <= x < d, U.F) && all(x -> 0 <= x < d, U.a)
        @test rc_is_symplectic(U.F, k, d)
        W = inv(U) ∘ U
        @test W.F == rc_identity!(Matrix{Int}(undef, 2k, 2k)) && all(iszero, W.a)
    end
end

function rc_refill_allocations(rng::AbstractRNG, U::CliffordOperator)
    for _ in 1:3
        random_clifford!(rng, U)
    end
    return @allocated random_clifford!(rng, U)
end

@testset "Warmed refills allocate nothing" begin
    for d in (2, 3, 5), k in (0, 1, 2, 8, 33)
        U = random_clifford(Xoshiro(1), d, 1:k)
        @test rc_refill_allocations(Xoshiro(2), U) == 0
        @test rc_refill_allocations(Random.default_rng(), U) == 0
    end
    # The safe tier, at a modulus the fast tier rejects for k = 2.
    U = random_clifford(Xoshiro(1), RC_D1, 1:2)
    @test !U.fast
    @test rc_refill_allocations(Xoshiro(2), U) == 0
end

###########################
# Uniformly random states #
###########################

const RC_TABLEAU_TYPES = [("StabilizerTableau", StabilizerTableau),
                          ("DestabilizerTableau", DestabilizerTableau)]

# ζ^h ⊗_q X^x_q Z^z_q with ζ = i at d = 2 and ω = exp(2πi/d) otherwise; qudit 1
# is the leftmost Kronecker factor.
function rc_pauli_matrix(xz, h, d::Int)
    n = length(xz) ÷ 2
    ω = cis(2π / d)
    ζ = d == 2 ? complex(0.0, 1.0) : ω
    X = zeros(ComplexF64, d, d)
    for j in 0:(d - 1)
        X[mod(j + 1, d) + 1, j + 1] = 1
    end
    Z = Matrix{ComplexF64}(Diagonal([ω^j for j in 0:(d - 1)]))
    M = ones(ComplexF64, 1, 1)
    for q in 1:n
        M = kron(M, X^xz[q] * Z^xz[n + q])
    end
    return ζ^h * M
end

# The density matrix of the group the active columns generate:
# ρ = ∏_j (Σ_t g_j^t / d) / d^(n - m), from the stored phases.
function rc_density(tab)
    d, n, m = tab.d, tab.n, tab.m
    ρ = Matrix{ComplexF64}(I, d^n, d^n)
    for j in 1:m
        g = rc_pauli_matrix(tab.stab[1:(2n), j], tab.stab[2n + 1, j], d)
        ρ *= sum(g^t for t in 0:(d - 1)) / d
    end
    return ρ / d^(n - m)
end

# Rounded for use as a Set key; `+ 0.0` folds -0.0 into 0.0.
rc_density_key(ρ) = map(z -> complex(round(real(z); digits = 8) + 0.0,
                                     round(imag(z); digits = 8) + 0.0), ρ)

rc_same(a, b) = a == b
rc_same(a::QC.PrecomputedInvMod, b::QC.PrecomputedInvMod) = a.lookuptable == b.lookuptable
# Every field's object and a deep copy of its contents: an unchanged tableau
# holds the same arrays with the same entries.
rc_snapshot(tab) = [(getfield(tab, f), deepcopy(getfield(tab, f))) for f in fieldnames(typeof(tab))]
rc_unchanged(tab, snap) =
    all(getfield(tab, f) === ref && rc_same(getfield(tab, f), contents)
        for (f, (ref, contents)) in zip(fieldnames(typeof(tab)), snap))

# (d, n, m) => stabilizer groups with phases, isotropic subspaces. There are
# I(n, m, d) = ∏_{i=0}^{m-1} (d^(2(n-i)) - 1) / ∏_{j=1}^{m} (d^j - 1) subspaces
# and d^m · I groups; only the full products divide exactly.
const RC_STATE_COUNTS = [((2, 1, 1), 6, 3), ((3, 1, 1), 12, 4), ((2, 2, 1), 30, 15),
                         ((2, 2, 2), 60, 15), ((3, 2, 1), 120, 40)]

@testset "States are uniform over stabilizer groups ($label)" for (label, TT) in RC_TABLEAU_TYPES
    for ((d, n, m), nstates, nspaces) in RC_STATE_COUNTS
        blocks, L = rc_alpha_blocks(n, m)
        tab = TT(d, n)
        states = Dict{Matrix{ComplexF64},Int}()
        failure = nothing
        for (index, script) in enumerate(rc_scripts(d, blocks, L + m))
            src = RCScript(script)
            random_state!(src, tab; m)
            ρ = rc_density(tab)
            failure = rc_failed((; d, n, m, index),
                :consumed => rc_consumed_all(src),
                :physical => ρ ≈ ρ' && tr(ρ) ≈ 1 && ρ * ρ ≈ ρ / d^(n - m))
            failure === nothing || break
            key = rc_density_key(ρ)
            states[key] = get(states, key, 0) + 1
        end
        @test failure === nothing
        @test length(states) == nstates
        @test allequal(values(states))

        # Phase-free runs have no phase suffix and are keyed by row space.
        plain = TT(d, n; storephase = false)
        spaces = Dict{Matrix{Int},Int}()
        failure = nothing
        for (index, script) in enumerate(rc_scripts(d, blocks, L))
            src = RCScript(script)
            random_state!(src, plain; m)
            failure = rc_failed((; d, n, m, index, storephase = false),
                :consumed => rc_consumed_all(src))
            failure === nothing || break
            key = first(rc_rref(permutedims(plain.stab[:, 1:m]), d))
            spaces[key] = get(spaces, key, 0) + 1
        end
        @test failure === nothing
        @test length(spaces) == nspaces
        @test allequal(values(spaces))
    end
end

@testset "random_state! validates m before drawing or mutating ($label)" for (label, TT) in RC_TABLEAU_TYPES
    tab = TT(3, 3; state = :ghz)
    snap = rc_snapshot(tab)
    none = RCScript(Int[])
    @test_throws ArgumentError random_state!(none, tab; m = -1)
    @test_throws ArgumentError random_state!(none, tab; m = 4)
    @test_throws ArgumentError random_state!(tab; m = 4)
    @test rc_unchanged(tab, snap)
    # m = 0 is the maximally mixed state, drawn from nothing, even at n = 0.
    @test random_state!(none, tab; m = 0) === tab
    @test tab.m == 0 && all(iszero, tab.stab) && !tab.iscanonical
    TT === DestabilizerTableau && @test all(iszero, tab.destab) && all(iszero, tab.xdotz_cache)
    empty = TT(3, 0)
    @test random_state!(none, empty) === empty && empty.m == 0
    @test_throws ArgumentError random_state!(none, empty; m = 1)
    @test none.pos == 0
end

@testset "State draws: m(4n - 2m + 1), plus m phases when stored" begin
    for (_, TT) in RC_TABLEAU_TYPES, d in (2, 3), storephase in (false, true), n in 1:4, m in 0:n
        script = rc_simple_script(n, m, storephase ? m : 0)
        @test length(script) == m * (4n - 2m + 1) + (storephase ? m : 0)
        src = RCScript(script)
        tab = TT(d, n; storephase)
        @test random_state!(src, tab; m) === tab && rc_consumed_all(src)
    end
end

@testset "A failed draw leaves the tableau untouched ($label)" for (label, TT) in RC_TABLEAU_TYPES
    for d in (2, 3), storephase in (false, true), m in (1, 2)
        accepted = rc_simple_script(2, m, storephase ? m : 0)
        # The same script after one rejected α block (2r = 4 draws at step 1).
        for script in (accepted, [zeros(Int, 4); accepted]), allowed in 0:(length(script) - 1)
            tab = TT(d, 2; state = :ghz, storephase)
            canonicalize!(tab)
            snap = rc_snapshot(tab)
            failing = RCThrowing(RCScript(script), allowed)
            @test_throws RCInjectedFailure random_state!(failing, tab; m)
            @test failing.count == allowed
            @test rc_unchanged(tab, snap)
        end
    end
end

@testset "Seeds reproduce states; the RNG-free form uses the default RNG" begin
    for (_, TT) in RC_TABLEAU_TYPES, d in (2, 3)
        a = random_state!(Xoshiro(4), TT(d, 5); m = 3)
        b = random_state!(Xoshiro(4), TT(d, 5); m = 3)
        @test a.stab == b.stab
        Random.seed!(12)
        c = random_state!(TT(d, 5); m = 3)
        Random.seed!(12)
        @test random_state!(Random.default_rng(), TT(d, 5); m = 3).stab == c.stab
        Random.seed!(13)
        e = random_state!(TT(d, 5))
        Random.seed!(13)
        @test random_state!(Random.default_rng(), TT(d, 5); m = 5).stab == e.stab
    end
end

@testset "Random states keep every tableau invariant ($label)" for (label, TT) in RC_TABLEAU_TYPES
    rng = Xoshiro(33)
    for d in (2, 3), storephase in (false, true)
        cases = [(n, m) for n in (0, 1, 3) for m in 0:n]
        append!(cases, [(16, m) for m in (0, 1, 8, 16)])
        for (n, m) in cases, start in (:mixed, :ghz, :canonical)
            tab = TT(d, n; state = start === :mixed ? :mixed : :ghz, storephase,
                     inversemod = QC.JustInTimeInvMod())
            start === :canonical && canonicalize!(tab)
            fields = fieldnames(typeof(tab))
            before = map(f -> getfield(tab, f), fields)
            @test random_state!(rng, tab; m) === tab
            @test tab.d == d && tab.n == n && tab.m == m && tab.storephase == storephase
            @test !tab.iscanonical && tab.inversemod === QC.JustInTimeInvMod()
            @test all(f -> !(getfield(tab, f) isa Array) ||
                           getfield(tab, f) === before[findfirst(==(f), fields)], fields)
            @test all(x -> 0 <= x < d, tab.stab[1:(2n), :])
            storephase && @test all(x -> 0 <= x < QC.phase_modulus(d), tab.stab[2n + 1, :])
            @test all(iszero, tab.stab[:, (m + 1):n])
            # The raw constructor checks commutation and independence. At m = 0
            # there is nothing to check, and a 0 × 0 raw matrix (n = 0 without a
            # phase row) hangs the constructor's @turbo pass.
            m > 0 && @test TT(d, tab.stab[:, 1:m]; m, storephase) isa TT
            if storephase && d == 2
                @test all(j -> iseven(tab.stab[2n + 1, j] -
                                      sum(tab.stab[q, j] * tab.stab[n + q, j] for q in 1:n;
                                          init = 0)), 1:m)
            end
            @test is_pure(tab; verify = true) == (m == n)
            if TT === DestabilizerTableau
                @test all(iszero, tab.destab[:, (m + 1):n])
                @test all(iszero, tab.xdotz_cache[(m + 1):n])
            end
        end
    end
end

# The state stabilized by Z_1, …, Z_m with duals X_1, …, X_m: a Z-basis product
# tableau with its columns after m cleared. A bare reset! would be pure.
function rc_z_fixture(TT, d::Int, n::Int, m::Int, storephase::Bool)
    tab = TT(d, n; state = :product, basis = :Z, storephase)
    tab.stab[:, (m + 1):n] .= 0
    if tab isa DestabilizerTableau
        tab.destab[:, (m + 1):n] .= 0
        tab.xdotz_cache[(m + 1):n] .= 0
    end
    tab.m = m
    return tab
end

@testset "random_state! equals the completed Clifford applied to ⟨Z_1, …, Z_m⟩" begin
    n = 2
    rng = Xoshiro(20260925)
    for (d, stride, frames) in ((2, 1, 720), (3, 97, 535))
        blocks, L = rc_alpha_blocks(n, n)
        scripts = rc_scripts(d, blocks, L)[1:stride:end]
        @test length(scripts) == frames
        failure = nothing
        for (index, script) in enumerate(scripts)
            phase_draws = rand(rng, 0:(d - 1), 2n)
            U = random_clifford!(RCScript([script; phase_draws]), rc_identity_operator(d, n))
            for m in 0:n, storephase in (false, true), (_, TT) in RC_TABLEAU_TYPES
                # State phase draw j is operator phase draw n + j; a phase-free
                # state draws no phases at all.
                prefix = script[1:last(rc_alpha_blocks(n, m))]
                src = RCScript(storephase ? [prefix; phase_draws[(n + 1):(n + m)]] : prefix)
                # A canonicalized start leaves a stale canonicalization workspace.
                target = TT(d, n; state = :ghz, storephase)
                canonicalize!(target)
                random_state!(src, target; m)
                expected = apply!(rc_z_fixture(TT, d, n, m, storephase), U)
                destab = TT === DestabilizerTableau
                failure = rc_failed((; d, index, m, storephase, tableau = nameof(TT)),
                    :consumed => rc_consumed_all(src),
                    :m => target.m == expected.m == m,
                    :iscanonical => !target.iscanonical && !expected.iscanonical,
                    :stab => target.stab == expected.stab,
                    :destab => !destab || target.destab == expected.destab,
                    :xdotz_cache => !destab || target.xdotz_cache == expected.xdotz_cache)
                failure === nothing || break
            end
            failure === nothing || break
        end
        @test failure === nothing
    end
end

@testset "States at a large modulus use the safe tier" begin
    d, n = RC_D1, 2
    @test !QC.clifford_fast_dots(2n, d)
    for (_, TT) in RC_TABLEAU_TYPES, m in 1:n
        tab = TT(d, n; inversemod = QC.JustInTimeInvMod())
        random_state!(Xoshiro(m), tab; m)
        S = tab.stab[1:(2n), 1:m]
        @test all(x -> 0 <= x < d, S)
        @test all(rc_pair(view(S, :, i), view(S, :, j), n, d) == 0 for i in 1:m, j in 1:m)
        # Independence: a nonzero 2 × 2 minor when m = 2, a nonzero vector when m = 1.
        @test m == 1 ? any(!iszero, S) :
              any(mod(big(S[i, 1]) * S[j, 2] - big(S[j, 1]) * S[i, 2], d) != 0
                  for i in 1:(2n), j in 1:(2n))
        if TT === DestabilizerTableau
            @test all(rc_pair(view(tab.destab, :, i), view(S, :, j), n, d) == (i == j)
                      for i in 1:m, j in 1:m)
            @test all(tab.xdotz_cache[j] ==
                      mod(sum(big(S[q, j]) * S[n + q, j] for q in 1:n), d) for j in 1:m)
        end
    end
end

# The number of allocations and the bytes of one warmed call. Byte totals of
# large arrays vary between calls on some platforms (Windows), so growth with m
# is judged by the allocation count, and the bytes only against a budget.
function rc_state_allocations(rng::AbstractRNG, tab, m::Int)
    for _ in 1:3
        random_state!(rng, tab; m)
    end
    return (@allocations random_state!(rng, tab; m)), (@allocated random_state!(rng, tab; m))
end

@testset "State scratch does not grow with m" begin
    n = 12
    # One 2n × 2n matrix and four length-2n vectors, with room for array headers.
    budget = 2 * sizeof(Int) * ((2n)^2 + 4 * 2n)
    for (_, TT) in RC_TABLEAU_TYPES, d in (2, 3), storephase in (false, true)
        tab = TT(d, n; storephase)
        rng = Xoshiro(3)
        count1, bytes1 = rc_state_allocations(rng, tab, 1)
        countn, bytesn = rc_state_allocations(rng, tab, n)
        @test count1 == countn
        @test bytes1 <= budget && bytesn <= budget
        @test rc_state_allocations(rng, tab, 0) == (0, 0)
    end
end
