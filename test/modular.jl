using QuditClifford
using Test
using Random

const QC = QuditClifford

# Trial division. Self-contained so the test suite needs no Primes dependency.
function _is_prime(n::Int)
    n < 2 && return false
    n < 4 && return true
    iseven(n) && return false
    i = 3
    while i * i <= n
        n % i == 0 && return false
        i += 2
    end
    return true
end
_primes_upto(n::Int) = [p for p in 2:n if _is_prime(p)]

# nextprime(10^k) for k = 3..15, as literals: test/Project.toml has no Primes.
const LARGE_PRIMES = [1009, 10007, 100003, 1000003, 10000019, 100000007,
                      1000000007, 10000000019, 100000000003, 1000000000039,
                      10000000000037, 100000000000031, 1000000000000037]

# d = 131 is in here because the guard REJECTS Barrett there, so the fallback
# path is exercised; 127 is the largest prime under 128, the ceiling for a
# future Int8 element type; 443 is the largest accepted Barrett modulus.
# NB top level: `const` inside a `@testset` body is a syntax error.
const SMALL_D = (2, 3, 5, 7, 11, 13, 17, 19, 127, 131, 443)

@testset "Modular primitives" begin
    @testset "Barrett guard" begin
        K = QC._BARRETT_K
        @test K == 20

        # The guard is exactly right -- not merely sound -- for every prime it
        # is cheap to check exhaustively.
        @testset "agrees with exhaustive truth for every prime <= 700" begin
            for d in _primes_upto(700)
                M = QC._barrett_mul(d)
                truth = all(((x * M) >> K) == div(x, d) for x in 0:(d - 1)^2)
                @test QC._barrett_ok(M, d) == truth
            end
        end

        @testset "acceptance set" begin
            accepted = [d for d in _primes_upto(100_000) if QC._barrett_ok(QC._barrett_mul(d), d)]
            @test length(accepted) == 42
            @test minimum(accepted) == 2
            @test maximum(accepted) == 443
        end

        # Acceptance is NOT monotonic in d, so nobody may replace the derived
        # guard with a `d <= limit` comparison. Two independent witnesses.
        @testset "acceptance is not monotonic" begin
            @test !QC._barrett_ok(QC._barrett_mul(131), 131)
            @test QC._barrett_ok(QC._barrett_mul(137), 137)
            @test QC._barrett_ok(QC._barrett_mul(443), 443)
            for d in (421, 431, 433, 439, 449, 457)
                @test !QC._barrett_ok(QC._barrett_mul(d), d)
            end
        end

        # Regression: the direct encoding `(M*d - 2^K) * (d-1)^2 < 2^K`
        # overflows Int and wraps negative here, accepting an invalid tier.
        # This test exists to stop the guard being "simplified" back.
        @testset "overflow regression at d = 2511241" begin
            d = 2511241
            M = QC._barrett_mul(d)
            @test !QC._barrett_ok(M, d)
            @test (M * d - (1 << K)) * (d - 1)^2 < 0        # the wrap itself
        end

        # The guard never over-claims at large d. Exhaustive checking is
        # infeasible past d ~ 1e4, so worst-case x are probed deliberately
        # rather than trusting random sampling.
        @testset "never over-claims at large d" begin
            rng = Random.MersenneTwister(20260911)
            for d in vcat(LARGE_PRIMES[1:7], [2511241], _primes_upto(700))
                M = QC._barrett_mul(d)
                QC._barrett_ok(M, d) || continue
                X = (d - 1)^2
                probes = Int[0, 1, d - 1, d, d + 1, X, X - 1, X ÷ 2,
                             (X ÷ d) * d, (X ÷ d) * d + (d - 1)]
                append!(probes, rand(rng, 0:X, 2000))
                for x in probes
                    0 <= x <= X || continue
                    @test ((x * M) >> K) == div(x, d)
                end
            end
        end

        # M*d cannot overflow for any Int d: it is about 2^K for d <= 2^K,
        # and exactly d (M == 1) above that.
        @testset "M*d never overflows" begin
            for d in (2, 3, 1 << 20, (1 << 20) + 1, 10^9,
                      (typemax(Int) >> 1) + 1, typemax(Int))
                M = QC._barrett_mul(d)
                @test widemul(M, d) == M * d
            end
        end

        # What item C will depend on: every accepted tier's largest
        # intermediate (d-1)^2 * M fits in Int32.
        @testset "accepted intermediates fit in Int32" begin
            worst = 0
            for d in _primes_upto(100_000)
                M = QC._barrett_mul(d)
                QC._barrett_ok(M, d) || continue
                worst = max(worst, (d - 1)^2 * M)
            end
            @test 0 < worst <= typemax(Int32)
        end
    end

    # Naive references. Deliberately written the way the call sites read today,
    # so a mismatch means the primitive is wrong, not that the reference drifted.
    _ref_submul(dst, src, a, d) = [mod(dst[i] - mod(a * src[i], d), d) for i in eachindex(dst)]
    _ref_addmul(dst, src, a, d) = [mod(dst[i] + mod(a * src[i], d), d) for i in eachindex(dst)]

    @testset "Exhaustive at small d" begin
        for d in SMALL_D, a in 0:(d - 1)
            dst0 = collect(0:(d - 1))
            src = collect((d - 1):-1:0)

            @test QC.submul_mod!(copy(dst0), src, a, d) == _ref_submul(dst0, src, a, d)
            @test QC.addmul_mod!(copy(dst0), src, a, d) == _ref_addmul(dst0, src, a, d)
        end
    end

    @testset "Exhaustive over every (dst, src) pair at small d" begin
        for d in (2, 3, 5, 7, 11, 13), a in 0:(d - 1)
            dst0 = [x for x in 0:(d - 1) for _ in 0:(d - 1)]
            src = [y for _ in 0:(d - 1) for y in 0:(d - 1)]

            @test QC.submul_mod!(copy(dst0), src, a, d) == _ref_submul(dst0, src, a, d)
            @test QC.addmul_mod!(copy(dst0), src, a, d) == _ref_addmul(dst0, src, a, d)
        end
    end

    # The guard is load-bearing, not decorative. At d = 131 the Barrett
    # identity fails exactly for x congruent to -1 mod 131, and 118*121 = 14278
    # is such an x, reachable as a product of two residues. The unguarded
    # result is off by exactly d, which the conditional add then MASKS whenever
    # dst < mod(a*src, d) -- so the witness must pin dst on the other side.
    # Without this testset, deleting the `_barrett_ok` check from submul_mod!
    # leaves the whole suite green. (Verified: it does.)
    @testset "the Barrett guard is load-bearing" begin
        d, a, s = 131, 118, 121
        M = QC._barrett_mul(d)
        @test !QC._barrett_ok(M, d)
        @test ((a * s * M) >> QC._BARRETT_K) != div(a * s, d)   # identity really fails
        @test mod(a * s, d) == d - 1                            # so masking needs dst < d-1

        # The masking side is OPPOSITE for the two. An unguarded Barrett tier
        # returns t - d instead of t, and the single conditional correction
        # absorbs that whenever submul_mod! has dst < t, or addmul_mod! has
        # dst + t >= d. So probe both ends of the range for both functions:
        # dst = d-1 catches submul_mod!, dst = 0 catches addmul_mod!.
        for dst in (0, d - 1)
            @test QC.submul_mod!([dst], [s], a, d) == [mod(dst - mod(a * s, d), d)]
            @test QC.addmul_mod!([dst], [s], a, d) == [mod(dst + mod(a * s, d), d)]
        end
    end

    # The conditional add/subtract tier at large d. The reference uses Int128
    # so it cannot itself overflow.
    @testset "Conditional tier at large primes" begin
        rng = Random.MersenneTwister(20260911)
        for d in LARGE_PRIMES, a in (1, d - 1)
            dst = rand(rng, 0:(d - 1), 10^5)
            src = rand(rng, 0:(d - 1), 10^5)

            want_sub = [Int(mod(Int128(dst[i]) - Int128(a) * src[i], Int128(d))) for i in eachindex(dst)]
            want_add = [Int(mod(Int128(dst[i]) + Int128(a) * src[i], Int128(d))) for i in eachindex(dst)]

            @test QC.submul_mod!(copy(dst), src, a, d) == want_sub
            @test QC.addmul_mod!(copy(dst), src, a, d) == want_add
        end
    end

    # The documented ceiling of the add form, pinned by a test rather than by
    # a comment. dst + src <= 2(d-1) must not overflow Int, i.e. d <= 2^62.
    @testset "Conditional tier ceiling" begin
        dmax = (typemax(Int) >> 1) + 1          # 2^62 = 4611686018427387904
        @test dmax == 4611686018427387904

        # Exact at the ceiling: a = d-1 in submul_mod! is the add form.
        @test QC.submul_mod!([dmax - 1], [dmax - 1], dmax - 1, dmax) ==
              [Int(mod(Int128(dmax - 1) - Int128(dmax - 1)^2, Int128(dmax)))]
        @test QC.addmul_mod!([dmax - 1], [dmax - 1], 1, dmax) ==
              [Int(mod(Int128(dmax - 1) + Int128(dmax - 1), Int128(dmax)))]

        # One past the ceiling the add form wraps. This asserts the documented
        # envelope is where it is claimed to be -- it is not a bug report.
        over = dmax + 1
        @test QC.addmul_mod!([over - 1], [over - 1], 1, over) !=
              [Int(mod(Int128(over - 1) + Int128(over - 1), Int128(over)))]

        # The subtract form has no ceiling: dst - src lies in (-d, d) always.
        big_d = typemax(Int)
        @test QC.submul_mod!([0], [big_d - 1], 1, big_d) == [1]
        @test QC.addmul_mod!([big_d - 1], [big_d - 1], big_d - 1, big_d) == [0]
    end

    @testset "Vector level: views, lengths, strides" begin
        rng = Random.MersenneTwister(20260912)
        for d in (2, 3, 5, 7, 131, 443), len in (0, 1, 2, 7, 300)
            a = rand(rng, 0:(d - 1))
            dst0 = rand(rng, 0:(d - 1), len)
            src = rand(rng, 0:(d - 1), len)

            @test QC.submul_mod!(copy(dst0), src, a, d) == _ref_submul(dst0, src, a, d)
            @test QC.addmul_mod!(copy(dst0), src, a, d) == _ref_addmul(dst0, src, a, d)

            # offset (non-1-based) contiguous views
            pad = vcat(rand(rng, 0:(d - 1), 3), dst0, rand(rng, 0:(d - 1), 3))
            dv = view(pad, 4:(3 + len))
            QC.submul_mod!(dv, view(src, 1:len), a, d)
            @test collect(dv) == _ref_submul(dst0, src, a, d)
        end

        # Strided views: rows of a matrix. The call sites all pass columns, but
        # the signature says AbstractVector, so the contract is tested.
        d = 7
        A = [mod(i * j, d) for i in 1:6, j in 1:9]
        want = _ref_submul(collect(view(A, 2, :)), collect(view(A, 4, :)), 3, d)
        QC.submul_mod!(view(A, 2, :), view(A, 4, :), 3, d)
        @test collect(view(A, 2, :)) == want
    end

    # Disjoint columns of ONE matrix are a supported aliasing case and are used
    # at canonicalize.jl:250. Overlapping ranges are documented as unsupported
    # and are not tested.
    @testset "Disjoint columns of one matrix" begin
        d = 5
        M = [mod(3i + 7j, d) for i in 1:8, j in 1:4]
        want = _ref_addmul(collect(view(M, :, 1)), collect(view(M, :, 3)), 2, d)
        QC.addmul_mod!(view(M, :, 1), view(M, :, 3), 2, d)
        @test collect(view(M, :, 1)) == want
    end

    # The invariant the whole file depends on: everything that reaches a
    # tableau column is reduced into [0, d) on write. If this ever stops being
    # true, the conditional tiers become silently wrong.
    @testset "Tableau entries are reduced on write" begin
        for d in (2, 3, 5)
            raw = fill(7 * d + 3, 2 * 3 + 1, 3)      # deliberately unreduced
            tab = StabilizerTableau(d, raw; m=3, storephase=true)
            @test all(0 .<= tab.stab[1:(2 * 3), :] .< d)

            t2 = StabilizerTableau(d, 3; state=:product, basis=:Z)
            ws = zeros(Int, 2 * 3 + 1)
            QuditClifford.set_operator!(ws, t2, SinglePauli(1, 5 * d + 2, 3 * d + 1))
            @test all(0 .<= ws[1:(2 * 3)] .< d)
        end
    end
end
