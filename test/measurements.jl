using QuditClifford
using Test
using Random

@testset "Measurements" begin
    @testset "Qubits (d=2)" begin
        # Fixed seed for deterministic random-outcome tests.
        Random.seed!(42)

        # Pure qubit stabilizer |0⟩ with generator Z.
        # Noncommuting measurement (X) should be random, replace the generator,
        # and set the stabilizer phase to match the sampled outcome.
        outcomes = Int[]
        for _ in 1:10
            tab = reshape(Int[0, 1, 0], 3, 1)
            stab = StabilizerTableau(2, 1, tab; m=1, storephase=true)

            out = measure!(stab, Int[1, 0]) # measure X
            @test out in 0:1
            @test stab.m == 1
            @test stab.tableau[1, 1] == 1  # X
            @test stab.tableau[2, 1] == 0
            @test stab.tableau[3, 1] == mod(2 * out, 4)
            push!(outcomes, out)
        end
        @test sort!(unique(outcomes)) == [0, 1]  # ensure we got both outcomes at least once

        # Commuting measurement in span (Z) should be deterministic and leave the state.
        tab_det = reshape(Int[0, 1, 0], 3, 1)
        stab_det = StabilizerTableau(2, 1, tab_det; m=1, storephase=true)
        out_det = measure!(stab_det, Int[0, 1]) # measure Z
        @test out_det == 0
        @test stab_det.tableau[1, 1] == 0
        @test stab_det.tableau[2, 1] == 1
        @test stab_det.tableau[3, 1] == 0

        # Maximally mixed qubit (m=0): commuting but not in span should append a generator.
        outcomes2 = Int[]
        for _ in 1:10
            tab2 = zeros(Int, 3, 1)
            stab2 = StabilizerTableau(2, 1, tab2; m=0, storephase=true)

            out2 = measure!(stab2, Int[0, 1]) # measure Z
            @test out2 in 0:1
            @test stab2.m == 1
            @test stab2.tableau[1, 1] == 0
            @test stab2.tableau[2, 1] == 1
            @test stab2.tableau[3, 1] == mod(2 * out2, 4)
            push!(outcomes2, out2)
        end
        @test sort!(unique(outcomes2)) == [0, 1]  # ensure we got both outcomes at least once
    end

    @testset "Qudits (d=3)" begin
        # Fixed seed for deterministic random-outcome tests.
        Random.seed!(42)

        # Pure qutrit stabilizer |0⟩ with generator Z.
        # Noncommuting measurement (X) should be random, replace the generator,
        # and set the stabilizer phase to match the sampled outcome.
        outcomes = Int[]
        for _ in 1:20
            tab = reshape(Int[0, 1, 0], 3, 1)
            stab = StabilizerTableau(3, 1, tab; m=1, storephase=true)

            out = measure!(stab, Int[1, 0]) # measure X
            @test out in 0:2
            @test stab.m == 1
            @test stab.tableau[1, 1] == 1  # X
            @test stab.tableau[2, 1] == 0
            @test stab.tableau[3, 1] == mod(-out, 3)
            push!(outcomes, out)
        end
        @test sort!(unique(outcomes)) == [0, 1, 2]  # ensure we got all outcomes at least once

        # Commuting measurement in span (Z) should be deterministic and leave the state.
        tab_det = reshape(Int[0, 1, 0], 3, 1)
        stab_det = StabilizerTableau(3, 1, tab_det; m=1, storephase=true)
        out_det = measure!(stab_det, Int[0, 1]) # measure Z
        @test out_det == 0
        @test stab_det.tableau[1, 1] == 0
        @test stab_det.tableau[2, 1] == 1
        @test stab_det.tableau[3, 1] == 0

        # Maximally mixed qutrit (m=0): commuting but not in span should append a generator.
        outcomes2 = Int[]
        for _ in 1:20
            tab2 = zeros(Int, 3, 1)
            stab2 = StabilizerTableau(3, 1, tab2; m=0, storephase=true)

            out2 = measure!(stab2, Int[0, 1]) # measure Z
            @test out2 in 0:2
            @test stab2.m == 1
            @test stab2.tableau[1, 1] == 0
            @test stab2.tableau[2, 1] == 1
            @test stab2.tableau[3, 1] == mod(-out2, 3)
            push!(outcomes2, out2)
        end
        @test sort!(unique(outcomes2)) == [0, 1, 2]  # ensure we got all outcomes at least once
    end
end
