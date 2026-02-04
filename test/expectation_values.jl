using QuditClifford
using Test

@testset "Expectation values" begin
    @testset "Qubits (d=2)" begin
        # Qubit |0⟩ stabilized by Z.
        tab = reshape(Int[0, 1, 0], 3, 1)
        stab = StabilizerTableau(2, 1, tab; m=1, storephase=true)

        # Identity always has expectation 1.
        @test QuditClifford.expect!(stab, Int[0, 0]) ≈ 1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, Int[0, 0]) == 0

        # Z is in the stabilizer span → expectation 1.
        @test QuditClifford.expect!(stab, Int[0, 1]) ≈ 1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, Int[0, 1]) == 0

        # X is not in the span → expectation 0.
        @test QuditClifford.expect!(stab, Int[1, 0]) ≈ 0.0 + 0.0im
        @test QuditClifford.expect_int!(stab, Int[1, 0]) == -1

        # Include an explicit phase: -Z corresponds to phase exponent 2 (i^2 = -1).
        @test QuditClifford.expect!(stab, Int[0, 1, 2]) ≈ -1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, Int[0, 1, 2]) == 2

        # If phases are not stored, in-span operators still return 1 by convention.
        tab_nophase = reshape(Int[0, 1], 2, 1)
        stab_nophase = StabilizerTableau(2, 1, tab_nophase; m=1, storephase=false)
        @test QuditClifford.expect!(stab_nophase, Int[0, 1]) ≈ 1.0 + 0.0im
        @test QuditClifford.expect_int!(stab_nophase, Int[0, 1]) == 0

        # Operator structs and non-Vector AbstractVector inputs should also work.
        @test QuditClifford.expect!(stab, QuditClifford.SinglePauli(1, 0, 1)) ≈ 1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, QuditClifford.SinglePauli(1, 1, 0)) == -1
        @test QuditClifford.expect!(stab, QuditClifford.SinglePauli(1, 0, 1, 2)) ≈ -1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, QuditClifford.SinglePauli(1, 0, 1, 2)) == 2

        op_view = view(Int[0, 1, 2], :)
        @test QuditClifford.expect!(stab, op_view) ≈ -1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, op_view) == 2
    end

    @testset "Qudits (d=3)" begin
        # Qutrit |0⟩ stabilized by Z.
        tab = reshape(Int[0, 1, 0], 3, 1)
        stab = StabilizerTableau(3, 1, tab; m=1, storephase=true)

        # Identity always has expectation 1.
        @test QuditClifford.expect!(stab, Int[0, 0]) ≈ 1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, Int[0, 0]) == 0

        # Z is in the stabilizer span → expectation 1.
        @test QuditClifford.expect!(stab, Int[0, 1]) ≈ 1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, Int[0, 1]) == 0

        # X is not in the span → expectation 0.
        @test QuditClifford.expect!(stab, Int[1, 0]) ≈ 0.0 + 0.0im
        @test QuditClifford.expect_int!(stab, Int[1, 0]) == -1

        # Include an explicit phase: ω^1 Z, where ω = exp(2πi/3).
        ω = cis(2π / 3)
        @test QuditClifford.expect!(stab, Int[0, 1, 1]) ≈ ω
        @test QuditClifford.expect_int!(stab, Int[0, 1, 1]) == 1

        # If phases are not stored, in-span operators still return 1 by convention.
        tab_nophase = reshape(Int[0, 1], 2, 1)
        stab_nophase = StabilizerTableau(3, 1, tab_nophase; m=1, storephase=false)
        @test QuditClifford.expect!(stab_nophase, Int[0, 1]) ≈ 1.0 + 0.0im
        @test QuditClifford.expect_int!(stab_nophase, Int[0, 1]) == 0

        # Operator structs and non-Vector AbstractVector inputs should also work.
        @test QuditClifford.expect!(stab, QuditClifford.SinglePauli(1, 0, 1)) ≈ 1.0 + 0.0im
        @test QuditClifford.expect_int!(stab, QuditClifford.SinglePauli(1, 1, 0)) == -1
        @test QuditClifford.expect!(stab, QuditClifford.SinglePauli(1, 0, 1, 1)) ≈ ω
        @test QuditClifford.expect_int!(stab, QuditClifford.SinglePauli(1, 0, 1, 1)) == 1

        op_view = view(Int[0, 1, 1], :)
        @test QuditClifford.expect!(stab, op_view) ≈ ω
        @test QuditClifford.expect_int!(stab, op_view) == 1
    end
end
