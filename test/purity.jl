using QuditClifford
using Test
using Logging

@testset "Purity checks" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                # Pure product state |00⟩ (commuting, independent, m == n).
                tab_pure = zeros(Int, 5, 2)
                tab_pure[3, 1] = 1  # Z₁
                tab_pure[4, 2] = 1  # Z₂
                # Note: is_independent is destructive (it Gauss-Jordan reduces a view),
                # so use fresh tableaux for each property to keep tests isolated.
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
                @test with_logger(NullLogger()) do
                    !QuditClifford.is_commuting(tab_noncomm_comm)
                end

                tab_noncomm = TT(2, tab_noncomm; m=2, storephase=true)
                @test with_logger(NullLogger()) do
                    !QuditClifford.is_pure(tab_noncomm)
                end

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
                # Note: is_independent is destructive (it Gauss-Jordan reduces a view),
                # so use fresh tableaux for each property to keep tests isolated.
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
                @test with_logger(NullLogger()) do
                    !QuditClifford.is_commuting(tab_noncomm_comm)
                end

                tab_noncomm = TT(3, tab_noncomm; m=2, storephase=true)
                @test with_logger(NullLogger()) do
                    !QuditClifford.is_pure(tab_noncomm)
                end

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
