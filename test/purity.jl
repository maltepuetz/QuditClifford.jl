using QuditClifford
using Test
using Random

@testset "Purity checks" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                # Pure product state |00⟩ (commuting, independent, m == n).
                tab_pure = zeros(Int, 5, 2)
                tab_pure[3, 1] = 1  # Z₁
                tab_pure[4, 2] = 1  # Z₂
                tab_pure_comm = TT(2, tab_pure; m=2, storephase=true)
                @test QuditClifford.is_commuting(tab_pure_comm)

                tab_pure_indep = TT(2, tab_pure; m=2, storephase=true)
                @test QuditClifford.is_independent(tab_pure_indep)

                tab_pure = TT(2, tab_pure; m=2, storephase=true)
                @test QuditClifford.is_pure(tab_pure)

                # Non-commuting generators (X₁ and Z₁) should fail purity.
                tab_noncomm = zeros(Int, 5, 2)
                tab_noncomm[1, 1] = 1  # X₁
                tab_noncomm[3, 2] = 1  # Z₁
                tab_noncomm_comm = TT(2, tab_noncomm; m=2, storephase=true)
                @test !QuditClifford.is_commuting(tab_noncomm_comm)

                tab_noncomm = TT(2, tab_noncomm; m=2, storephase=true)
                @test !is_pure(tab_noncomm)

                # Linearly dependent generators (Z₁, Z₁) should fail independence and purity.
                tab_dep = zeros(Int, 5, 2)
                tab_dep[3, 1] = 1  # Z₁
                tab_dep[3, 2] = 1  # Z₁ again
                tab_dep_comm = TT(2, tab_dep; m=2, storephase=true)
                @test QuditClifford.is_commuting(tab_dep_comm)

                tab_dep_indep = TT(2, tab_dep; m=2, storephase=true)
                @test !QuditClifford.is_independent(tab_dep_indep)

                tab_dep = TT(2, tab_dep; m=2, storephase=true)
                @test !QuditClifford.is_pure(tab_dep)
            end

            @testset "Qudit (d=3)" begin
                # Pure product state |00⟩ (commuting, independent, m == n).
                tab_pure = zeros(Int, 5, 2)
                tab_pure[3, 1] = 1  # Z₁
                tab_pure[4, 2] = 1  # Z₂
                tab_pure_comm = TT(3, tab_pure; m=2, storephase=true)
                @test QuditClifford.is_commuting(tab_pure_comm)

                tab_pure_indep = TT(3, tab_pure; m=2, storephase=true)
                @test QuditClifford.is_independent(tab_pure_indep)

                tab_pure = TT(3, tab_pure; m=2, storephase=true)
                @test QuditClifford.is_pure(tab_pure)

                # Non-commuting generators (X₁ and Z₁) should fail purity.
                tab_noncomm = zeros(Int, 5, 2)
                tab_noncomm[1, 1] = 1  # X₁
                tab_noncomm[3, 2] = 1  # Z₁
                tab_noncomm_comm = TT(3, tab_noncomm; m=2, storephase=true)
                @test !QuditClifford.is_commuting(tab_noncomm_comm)

                tab_noncomm = TT(3, tab_noncomm; m=2, storephase=true)
                @test !is_pure(tab_noncomm)

                # Linearly dependent generators (Z₁, Z₁) should fail independence and purity.
                tab_dep = zeros(Int, 5, 2)
                tab_dep[3, 1] = 1  # Z₁
                tab_dep[3, 2] = 1  # Z₁ again
                tab_dep_comm = TT(3, tab_dep; m=2, storephase=true)
                @test QuditClifford.is_commuting(tab_dep_comm)

                tab_dep_indep = TT(3, tab_dep; m=2, storephase=true)
                @test !QuditClifford.is_independent(tab_dep_indep)

                tab_dep = TT(3, tab_dep; m=2, storephase=true)
                @test !QuditClifford.is_pure(tab_dep)
            end
        end
    end
end

# The symplectic Gram kernel behind `is_commuting`. Tested directly on raw
# matrices rather than through a tableau so it stays reachable once the
# constructors start rejecting non-commuting input.
#
# `_first_noncommuting_pair` computes P = X'Z once and reads the symplectic
# form off it as P[j,i] - P[i,j], which relies on the form being
# antisymmetric. A reference written the other way round -- summing
# x_j*z_i - z_j*x_i directly, per pair -- is what pins that identity.
@testset "First non-commuting pair" begin
    # Independent reference: the definition, per pair, no Gram matrix.
    function reference_pair(stab, n, m, d)
        for j in 1:m, i in (j+1):m
            s = 0
            for q in 1:n
                s += stab[q, j] * stab[n+q, i] - stab[n+q, j] * stab[q, i]
            end
            mod(s, d) != 0 && return (j, i)
        end
        return (0, 0)
    end

    kernel(stab, n, m, d) = QuditClifford._first_noncommuting_pair(
        stab, n, m, d, zeros(Int, max(m, 1), max(m, 1)))

    @testset "agrees with the definition on random tableaux" begin
        rng = MersenneTwister(20260915)
        for d in (2, 3, 5, 7), n in (1, 2, 3, 5, 9), m in 0:n
            for _ in 1:12
                stab = rand(rng, 0:(d-1), 2n, max(m, 1))
                @test kernel(stab, n, m, d) == reference_pair(stab, n, m, d)
            end
        end
    end

    # Random tableaux are almost never commuting, so on their own they would
    # never exercise the (0,0) return. These do.
    @testset "commuting tableaux report no pair" begin
        for d in (2, 3, 5), n in (1, 2, 4, 7)
            # All-Z generators pairwise commute: every x entry is zero.
            zonly = zeros(Int, 2n, n)
            for j in 1:n
                zonly[n+j, j] = 1
            end
            @test kernel(zonly, n, n, d) == (0, 0)

            # A genuine state's generators, reached through the package.
            tab = StabilizerTableau(d, n; state = :ghz)
            @test kernel(tab.stab, n, tab.m, d) == (0, 0)
        end
    end

    @testset "reports the first pair in column order" begin
        # X1 and Z1 anticommute; everything else here commutes, so (2,4) is
        # the answer only if the scan really is ordered by j and then i.
        # Column 1 is deliberately Z2 rather than Z1 -- Z1 would anticommute
        # with the X1 in column 2 and make (1,2) the first pair instead.
        d, n = 3, 2
        stab = zeros(Int, 2n, 4)
        stab[n+2, 1] = 1   # Z2
        stab[1, 2] = 1     # X1
        stab[n+2, 3] = 1   # Z2
        stab[n+1, 4] = 1   # Z1
        @test reference_pair(stab, n, 4, d) == (2, 4)   # the fixture is what I claim
        @test kernel(stab, n, 4, d) == (2, 4)
    end

    @testset "m <= 1 has no pairs to check" begin
        for d in (2, 3), n in (1, 3)
            stab = rand(MersenneTwister(1), 0:(d-1), 2n, 1)
            @test kernel(stab, n, 0, d) == (0, 0)
            @test kernel(stab, n, 1, d) == (0, 0)
        end
    end
end
