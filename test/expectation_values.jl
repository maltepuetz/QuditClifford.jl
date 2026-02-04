using QuditClifford
using Test

@testset "Expectation values" begin
    @testset "Qubits (d=2)" begin
        # Qubit |0⟩ stabilized by Z.
        tab = reshape(Int64[0, 1, 0], 3, 1)
        stab = StabilizerTableau(2, 1, tab; m=1, storephase=true)

        # Identity always has expectation 1.
        @test QuditClifford.expectation_value!(stab, Int[0, 0]) ≈ 1.0 + 0.0im

        # Z is in the stabilizer span → expectation 1.
        @test QuditClifford.expectation_value!(stab, Int[0, 1]) ≈ 1.0 + 0.0im

        # X is not in the span → expectation 0.
        @test QuditClifford.expectation_value!(stab, Int[1, 0]) ≈ 0.0 + 0.0im

        # Include an explicit phase: -Z corresponds to phase exponent 2 (i^2 = -1).
        @test QuditClifford.expectation_value!(stab, Int[0, 1, 2]) ≈ -1.0 + 0.0im

        # If phases are not stored, in-span operators still return 1 by convention.
        tab_nophase = reshape(Int64[0, 1], 2, 1)
        stab_nophase = StabilizerTableau(2, 1, tab_nophase; m=1, storephase=false)
        @test QuditClifford.expectation_value!(stab_nophase, Int[0, 1]) ≈ 1.0 + 0.0im
    end

    @testset "Qudits (d=3)" begin
        # Qutrit |0⟩ stabilized by Z.
        tab = reshape(Int64[0, 1, 0], 3, 1)
        stab = StabilizerTableau(3, 1, tab; m=1, storephase=true)

        # Identity always has expectation 1.
        @test QuditClifford.expectation_value!(stab, Int[0, 0]) ≈ 1.0 + 0.0im

        # Z is in the stabilizer span → expectation 1.
        @test QuditClifford.expectation_value!(stab, Int[0, 1]) ≈ 1.0 + 0.0im

        # X is not in the span → expectation 0.
        @test QuditClifford.expectation_value!(stab, Int[1, 0]) ≈ 0.0 + 0.0im

        # Include an explicit phase: ω^1 Z, where ω = exp(2πi/3).
        ω = cis(2π / 3)
        @test QuditClifford.expectation_value!(stab, Int[0, 1, 1]) ≈ ω

        # If phases are not stored, in-span operators still return 1 by convention.
        tab_nophase = reshape(Int64[0, 1], 2, 1)
        stab_nophase = StabilizerTableau(3, 1, tab_nophase; m=1, storephase=false)
        @test QuditClifford.expectation_value!(stab_nophase, Int[0, 1]) ≈ 1.0 + 0.0im
    end
end
