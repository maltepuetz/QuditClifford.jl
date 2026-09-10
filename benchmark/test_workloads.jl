using Test
include(joinpath(@__DIR__, "workloads.jl"))

@testset "Workloads" begin
    @testset "Purification is deterministic and reaches full rank" begin
        h1, l1 = purification_trajectory(d = 2, n = 8, seed = 42)
        h2, l2 = purification_trajectory(d = 2, n = 8, seed = 42)
        @test h1 == h2
        @test l1 == l2
        @test l1 <= 10_000              # purified before the cutoff
        @test h1[end] == 8              # full rank
        @test issorted(h1)              # rank never decreases

        h3, _ = purification_trajectory(d = 2, n = 8, seed = 43)
        @test h1 != h3                  # the seed actually matters
    end

    @testset "Ising is deterministic and stays pure" begin
        s1 = ising_trajectory!(L = 8, depth = 16, seed = 7)
        s2 = ising_trajectory!(L = 8, depth = 16, seed = 7)
        @test s1.stab == s2.stab
        @test s1.m == s2.m
        # Started from a pure product state; projective measurement keeps it pure.
        @test is_pure(s1)

        s3 = ising_trajectory!(L = 8, depth = 16, seed = 8)
        @test s1.stab != s3.stab         # the seed actually matters

        # Exercise the default `depth = 8L` keyword path (every call above sets
        # depth explicitly) and pin its value: omitting depth must match passing
        # depth = 8*8 explicitly, for the same seed.
        s4 = ising_trajectory!(L = 8, seed = 7)
        s5 = ising_trajectory!(L = 8, depth = 8 * 8, seed = 7)
        @test s4.stab == s5.stab
        @test is_pure(s4)
    end

    @testset "Micro group builds for both tableau types" begin
        for T in (StabilizerTableau, DestabilizerTableau)
            g = micro_group(d = 3, n = 8, T = T)
            @test length(keys(g)) == 8
        end
    end

    @testset "Snapshot/restore round-trips" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            tab = T(d, 16; state = :ghz)
            snap = snapshot(tab)
            stab0, m0 = copy(tab.stab), tab.m

            # A mutating op, then restore: the tableau must be bit-identical.
            measure!(tab, SinglePauli(1, 1, 0); outcome = 0, phase_policy = 2)
            canonicalize!(tab)
            @test tab.stab != stab0          # the ops really did change it
            restore!(snap)
            @test tab.stab == stab0
            @test tab.m == m0
            @test tab.iscanonical == false
            T === DestabilizerTableau && @test snap.destab !== nothing
        end
    end

    @testset "canonicalize! benchmark measures the real path" begin
        # If the restored tableau were already canonical, canonicalize! would
        # hit every `β == 0 && continue` guard and collapse to O(n*m) -- the
        # benchmark would silently measure the wrong thing, at ~1/100 the cost.
        for T in (StabilizerTableau, DestabilizerTableau)
            tab = T(3, 16; state = :ghz)
            snap = snapshot(tab)
            before = copy(tab.stab)

            canonicalize!(tab)
            @test tab.stab != before         # :ghz is genuinely non-canonical
            @test tab.iscanonical

            restore!(snap)
            @test tab.stab == before         # back to the non-canonical data
            @test !tab.iscanonical

            # ...so a second canonicalize! does the full job again, which is
            # the property the benchmark depends on sample after sample.
            canonicalize!(tab)
            @test tab.stab != before
        end

        # A :product tableau is already in RCEF -- the state the benchmark
        # must NOT use. Pinned so a future "simplification" to :product fails.
        tab = DestabilizerTableau(3, 16; state = :product)
        before = copy(tab.stab)
        canonicalize!(tab)
        @test tab.stab == before
    end

    @testset "Measurements stay quiet at d = 2" begin
        # phase_policy = 2 everywhere: the default policy @warns on every
        # non-Hermitian qubit Pauli, which a random circuit produces constantly.
        @test_logs ising_trajectory!(L = 6, depth = 8, seed = 3)
        @test_logs purification_trajectory(d = 2, n = 6, seed = 3)
    end

    @testset "Spine operators hit the branches they claim" begin
        # Each spine leaf is named for a measure! branch. If the operator stops
        # selecting that branch the leaf silently measures something else, and
        # nothing else in the suite would notice.
        for n in (8, 64, 256), d in (2, 3)
            i, j = spread_sites(n)
            @test 1 <= i < j <= n
            @test length(unique(spread_eight(n))) == 8
            @test all(1 .<= spread_eight(n) .<= n)

            tab = DestabilizerTableau(d, n; state = :product)

            # noncommuting: X on a Z-basis site anticommutes with its stabilizer
            @test any(mod(QuditClifford.commutation_col(tab.stab, k, spread_x(n)), d) != 0
                      for k in 1:tab.m)
            # deterministic: Z*Z is a product of two generators, so in span
            @test expect_int!(tab, spread_z(n)) != -1
            # ...and spread_x is not, which is what expect!/out_of_span needs
            @test expect_int!(tab, spread_x(n)) == -1

            # Hermitian at d = 2, so phase_policy cannot matter for these two
            if d == 2
                @test_logs measure!(DestabilizerTableau(2, n; state = :product),
                                    spread_x(n); outcome = 0)
            end
        end
    end

    @testset "Mid-circuit state is scrambled, pure and deterministic" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            a = scrambled_state(T, d, 16; seed = 11)
            b = scrambled_state(T, d, 16; seed = 11)
            @test a.stab == b.stab                     # deterministic
            @test a.stab != T(d, 16; state = :product).stab   # actually scrambled
            @test a.m == 16                            # stayed pure, so measure! cannot
            @test is_pure(a)                           # hit the m == n append error
            @test scrambled_state(T, d, 16; seed = 12).stab != a.stab
        end
    end

    @testset "Mid-circuit operators hit the branches they claim" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            tab = scrambled_state(T, d, 16; seed = 11)

            # in_span_operator is one of the generators, so expect! must find
            # it (-1 means <P> = 0, i.e. not in the stabilizer group).
            @test expect_int!(tab, in_span_operator(tab)) != -1

            # first_anticommuting must genuinely anticommute, or measure! takes
            # the commuting branch and throws once m == n.
            anti = first_anticommuting(tab, [SinglePauli(i, 1, 0) for i in 1:16])
            @test any(mod(QuditClifford.commutation_col(tab.stab, j, anti), d) != 0
                      for j in 1:tab.m)
            @test_throws ErrorException first_anticommuting(tab, SinglePauli[])
        end
    end

    @testset "entropy benchmark measures a non-canonical tableau" begin
        # entanglement_entropy never canonicalizes, so most of its cost is the
        # elimination in rank_fp_cols! -- and how much there is to eliminate
        # depends entirely on how close the generators already are to echelon
        # form. The leaf therefore has to hold a dirty tableau, and has to stay
        # dirty sample after sample.
        for T in (StabilizerTableau, DestabilizerTableau)
            tab = scrambled_state(T, 3, 16; seed = 1)
            sub = collect(1:8)
            @test !tab.iscanonical

            S = entanglement_entropy(tab, sub)
            @test !tab.iscanonical           # ...and every later sample sees the same state

            before = copy(tab.stab)
            canonicalize!(tab)
            @test tab.stab != before         # the data really was non-canonical
            @test entanglement_entropy(tab, sub) == S   # same answer either way
        end
    end

    @testset "Mid-circuit group builds and runs" begin
        for T in (StabilizerTableau, DestabilizerTableau)
            g = midcircuit_group(d = 3, n = 16, T = T)
            @test length(keys(g)) == 6
            # Actually execute every leaf: @benchmarkable bodies are quoted, so
            # a broken one is invisible until something runs it. This is also
            # what proves the two measure! leaves do not throw.
            run(g; samples = 1, evals = 1, seconds = 0.05)
        end
    end

    @testset "Destabilizer memory is 3x stabilizer" begin
        for n in (128, 512)
            rs = tableau_bytes(StabilizerTableau(2, n; state = :product))
            rd = tableau_bytes(DestabilizerTableau(2, n; state = :product))
            @test 2.8 < rd / rs < 3.2
        end
    end
end
