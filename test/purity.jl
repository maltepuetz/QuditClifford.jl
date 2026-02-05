using QuditClifford
using Test
using Logging

@testset "Purity checks" begin
    @testset "Qubits (d=2)" begin
        # Pure product state |00⟩ (commuting, independent, m == n).
        tab_pure = zeros(Int, 5, 2)
        tab_pure[3, 1] = 1  # Z₁
        tab_pure[4, 2] = 1  # Z₂
        # Note: is_independent is destructive (it Gauss-Jordan reduces a view),
        # so use fresh tableaux for each property to keep tests isolated.
        stab_pure_comm = StabilizerTableau(2, tab_pure; m=2, storephase=true)
        @test QuditClifford.is_commuting(stab_pure_comm)

        stab_pure_indep = StabilizerTableau(2, tab_pure; m=2, storephase=true)
        @test QuditClifford.is_independent(stab_pure_indep)

        stab_pure = StabilizerTableau(2, tab_pure; m=2, storephase=true)
        @test QuditClifford.is_pure(stab_pure)

        # Non-commuting generators (X₁ and Z₁) should fail purity.
        tab_noncomm = zeros(Int, 5, 2)
        tab_noncomm[1, 1] = 1  # X₁
        tab_noncomm[3, 2] = 1  # Z₁
        stab_noncomm_comm = StabilizerTableau(2, tab_noncomm; m=2, storephase=true)
        @test with_logger(NullLogger()) do
            !QuditClifford.is_commuting(stab_noncomm_comm)
        end

        stab_noncomm = StabilizerTableau(2, tab_noncomm; m=2, storephase=true)
        @test with_logger(NullLogger()) do
            !QuditClifford.is_pure(stab_noncomm)
        end

        # Linearly dependent generators (Z₁, Z₁) should fail independence and purity.
        tab_dep = zeros(Int, 5, 2)
        tab_dep[3, 1] = 1  # Z₁
        tab_dep[3, 2] = 1  # Z₁ again
        stab_dep_comm = StabilizerTableau(2, tab_dep; m=2, storephase=true)
        @test QuditClifford.is_commuting(stab_dep_comm)

        stab_dep_indep = StabilizerTableau(2, tab_dep; m=2, storephase=true)
        @test !QuditClifford.is_independent(stab_dep_indep)

        stab_dep = StabilizerTableau(2, tab_dep; m=2, storephase=true)
        @test !QuditClifford.is_pure(stab_dep)
    end

    @testset "Qudits (d=3)" begin
        # Pure product state |00⟩ (commuting, independent, m == n).
        tab_pure = zeros(Int, 5, 2)
        tab_pure[3, 1] = 1  # Z₁
        tab_pure[4, 2] = 1  # Z₂
        # Note: is_independent is destructive (it Gauss-Jordan reduces a view),
        # so use fresh tableaux for each property to keep tests isolated.
        stab_pure_comm = StabilizerTableau(3, tab_pure; m=2, storephase=true)
        @test QuditClifford.is_commuting(stab_pure_comm)

        stab_pure_indep = StabilizerTableau(3, tab_pure; m=2, storephase=true)
        @test QuditClifford.is_independent(stab_pure_indep)

        stab_pure = StabilizerTableau(3, tab_pure; m=2, storephase=true)
        @test QuditClifford.is_pure(stab_pure)

        # Non-commuting generators (X₁ and Z₁) should fail purity.
        tab_noncomm = zeros(Int, 5, 2)
        tab_noncomm[1, 1] = 1  # X₁
        tab_noncomm[3, 2] = 1  # Z₁
        stab_noncomm_comm = StabilizerTableau(3, tab_noncomm; m=2, storephase=true)
        @test with_logger(NullLogger()) do
            !QuditClifford.is_commuting(stab_noncomm_comm)
        end

        stab_noncomm = StabilizerTableau(3, tab_noncomm; m=2, storephase=true)
        @test with_logger(NullLogger()) do
            !QuditClifford.is_pure(stab_noncomm)
        end

        # Linearly dependent generators (Z₁, Z₁) should fail independence and purity.
        tab_dep = zeros(Int, 5, 2)
        tab_dep[3, 1] = 1  # Z₁
        tab_dep[3, 2] = 1  # Z₁ again
        stab_dep_comm = StabilizerTableau(3, tab_dep; m=2, storephase=true)
        @test QuditClifford.is_commuting(stab_dep_comm)

        stab_dep_indep = StabilizerTableau(3, tab_dep; m=2, storephase=true)
        @test !QuditClifford.is_independent(stab_dep_indep)

        stab_dep = StabilizerTableau(3, tab_dep; m=2, storephase=true)
        @test !QuditClifford.is_pure(stab_dep)
    end
end
