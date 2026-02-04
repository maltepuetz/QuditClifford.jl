using QuditClifford
using Test
using Aqua

@testset "QuditClifford.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(QuditClifford; piracies = VERSION < v"1.12.0")
    end
    @testset "measure! accepts vector operators" begin
        # pure qubit stabilizer |0⟩ with generator Z
        tab = reshape(Int64[0, 1, 0], 3, 1)
        stab = StabilizerTableau(2, 1, tab; m=1, storephase=true)

        out = measure!(stab, Int[1, 0]) # measure X
        @test out in 0:1
        @test stab.m == 1
        @test stab.tableau[1, 1] == 1
        @test stab.tableau[2, 1] == 0
        @test stab.tableau[3, 1] == mod(2 * out, 4)

        # maximally mixed qubit (m=0): commuting, not-in-span appends a generator
        tab2 = zeros(Int64, 3, 1)
        stab2 = StabilizerTableau(2, 1, tab2; m=0, storephase=true)

        out2 = measure!(stab2, Int[0, 1]) # measure Z
        @test out2 in 0:1
        @test stab2.m == 1
        @test stab2.tableau[1, 1] == 0
        @test stab2.tableau[2, 1] == 1
        @test stab2.tableau[3, 1] == mod(2 * out2, 4)
    end
end
