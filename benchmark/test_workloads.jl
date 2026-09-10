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

    @testset "Destabilizer memory is 3x stabilizer" begin
        for n in (128, 512)
            rs = tableau_bytes(StabilizerTableau(2, n; state = :product))
            rd = tableau_bytes(DestabilizerTableau(2, n; state = :product))
            @test 2.8 < rd / rs < 3.2
        end
    end
end
