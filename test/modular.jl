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
end
