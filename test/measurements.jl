using QuditClifford
using Test
using Random

@testset "Measurements" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                # Fixed seed for deterministic random-outcome tests.
                Random.seed!(42)

                # Pure qubit stabilizer |0⟩ with generator Z.
                # Noncommuting measurement (X) should be random, replace the generator,
                # and set the stabilizer phase to match the sampled outcome.
                outcomes = Int[]
                for _ in 1:10
                    tab = reshape(Int[0, 1, 0], 3, 1)
                    tab = TT(2, tab; m=1, storephase=true)

                    out = measure!(tab, Int[1, 0]) # measure X
                    @test out in 0:1
                    @test tab.m == 1
                    @test tab.stab[1, 1] == 1  # X
                    @test tab.stab[2, 1] == 0
                    @test tab.stab[3, 1] == mod(2 * out, 4)
                    push!(outcomes, out)
                end
                @test sort!(unique(outcomes)) == [0, 1]  # ensure we got both outcomes at least once

                # Commuting measurement in span (Z) should be deterministic and leave the state.
                tab_det = reshape(Int[0, 1, 0], 3, 1)
                tab_det = TT(2, tab_det; m=1, storephase=true)
                out_det = measure!(tab_det, Int[0, 1]) # measure Z
                @test out_det == 0
                @test tab_det.stab[1, 1] == 0
                @test tab_det.stab[2, 1] == 1
                @test tab_det.stab[3, 1] == 0

                # Maximally mixed qubit (m=0): commuting but not in span should append a generator.
                outcomes2 = Int[]
                for _ in 1:10
                    tab2 = zeros(Int, 3, 1)
                    tab2 = TT(2, tab2; m=0, storephase=true)

                    out2 = measure!(tab2, Int[0, 1]) # measure Z
                    @test out2 in 0:1
                    @test tab2.m == 1
                    @test tab2.stab[1, 1] == 0
                    @test tab2.stab[2, 1] == 1
                    @test tab2.stab[3, 1] == mod(2 * out2, 4)
                    push!(outcomes2, out2)
                end
                @test sort!(unique(outcomes2)) == [0, 1]  # ensure we got both outcomes at least once
            end

            @testset "Qudit (d=3)" begin
                # Fixed seed for deterministic random-outcome tests.
                Random.seed!(42)

                # Pure qutrit stabilizer |0⟩ with generator Z.
                # Noncommuting measurement (X) should be random, replace the generator,
                # and set the stabilizer phase to match the sampled outcome.
                outcomes = Int[]
                for _ in 1:20
                    tab = reshape(Int[0, 1, 0], 3, 1)
                    tab = TT(3, tab; m=1, storephase=true)

                    out = measure!(tab, Int[1, 0]) # measure X
                    @test out in 0:2
                    @test tab.m == 1
                    @test tab.stab[1, 1] == 1  # X
                    @test tab.stab[2, 1] == 0
                    @test tab.stab[3, 1] == mod(-out, 3)
                    push!(outcomes, out)
                end
                @test sort!(unique(outcomes)) == [0, 1, 2]  # ensure we got all outcomes at least once

                # Commuting measurement in span (Z) should be deterministic and leave the state.
                tab_det = reshape(Int[0, 1, 0], 3, 1)
                tab_det = TT(3, tab_det; m=1, storephase=true)
                out_det = measure!(tab_det, Int[0, 1]) # measure Z
                @test out_det == 0
                @test tab_det.stab[1, 1] == 0
                @test tab_det.stab[2, 1] == 1
                @test tab_det.stab[3, 1] == 0

                # Maximally mixed qutrit (m=0): commuting but not in span should append a generator.
                outcomes2 = Int[]
                for _ in 1:20
                    tab2 = zeros(Int, 3, 1)
                    tab2 = TT(3, tab2; m=0, storephase=true)

                    out2 = measure!(tab2, Int[0, 1]) # measure Z
                    @test out2 in 0:2
                    @test tab2.m == 1
                    @test tab2.stab[1, 1] == 0
                    @test tab2.stab[2, 1] == 1
                    @test tab2.stab[3, 1] == mod(-out2, 3)
                    push!(outcomes2, out2)
                end
                @test sort!(unique(outcomes2)) == [0, 1, 2]  # ensure we got all outcomes at least once
            end
        end
    end
end
