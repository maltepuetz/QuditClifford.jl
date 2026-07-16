using QuditClifford
using Test

@testset "Expectation values" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                # Qubit |0⟩ stabilized by Z.
                tab = reshape(Int[0, 1, 0], 3, 1)
                tab = TT(2, tab; m=1, storephase=true)

                # Identity always has expectation 1.
                @test QuditClifford.expect!(tab, Int[0, 0]) ≈ 1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, Int[0, 0]) == 0

                # Z is in the stabilizer span → expectation 1.
                @test QuditClifford.expect!(tab, Int[0, 1]) ≈ 1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, Int[0, 1]) == 0

                # X is not in the span → expectation 0.
                @test QuditClifford.expect!(tab, Int[1, 0]) ≈ 0.0 + 0.0im
                @test QuditClifford.expect_int!(tab, Int[1, 0]) == -1

                # Include an explicit phase: -Z corresponds to phase exponent 2 (i^2 = -1).
                @test QuditClifford.expect!(tab, Int[0, 1, 2]) ≈ -1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, Int[0, 1, 2]) == 2

                # If phases are not stored, in-span operators still return 1 by convention.
                tab_nophase = reshape(Int[0, 1], 2, 1)
                tab_nophase = TT(2, tab_nophase; m=1, storephase=false)
                @test QuditClifford.expect!(tab_nophase, Int[0, 1]) ≈ 1.0 + 0.0im
                @test QuditClifford.expect_int!(tab_nophase, Int[0, 1]) == 0

                # Operator structs and non-Vector AbstractVector inputs should also work.
                @test QuditClifford.expect!(tab, QuditClifford.SinglePauli(1, 0, 1)) ≈ 1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, QuditClifford.SinglePauli(1, 1, 0)) == -1
                @test QuditClifford.expect!(tab, QuditClifford.SinglePauli(1, 0, 1, 2)) ≈ -1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, QuditClifford.SinglePauli(1, 0, 1, 2)) == 2

                op_view = view(Int[0, 1, 2], :)
                @test QuditClifford.expect!(tab, op_view) ≈ -1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, op_view) == 2
            end

            @testset "Qudit (d=3)" begin
                # Qutrit |0⟩ stabilized by Z.
                tab = reshape(Int[0, 1, 0], 3, 1)
                tab = TT(3, tab; m=1, storephase=true)

                # Identity always has expectation 1.
                @test QuditClifford.expect!(tab, Int[0, 0]) ≈ 1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, Int[0, 0]) == 0

                # Z is in the stabilizer span → expectation 1.
                @test QuditClifford.expect!(tab, Int[0, 1]) ≈ 1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, Int[0, 1]) == 0

                # X is not in the span → expectation 0.
                @test QuditClifford.expect!(tab, Int[1, 0]) ≈ 0.0 + 0.0im
                @test QuditClifford.expect_int!(tab, Int[1, 0]) == -1

                # Include an explicit phase: ω^1 Z, where ω = exp(2πi/3).
                ω = cis(2π / 3)
                @test QuditClifford.expect!(tab, Int[0, 1, 1]) ≈ ω
                @test QuditClifford.expect_int!(tab, Int[0, 1, 1]) == 1

                # If phases are not stored, in-span operators still return 1 by convention.
                tab_nophase = reshape(Int[0, 1], 2, 1)
                tab_nophase = TT(3, tab_nophase; m=1, storephase=false)
                @test QuditClifford.expect!(tab_nophase, Int[0, 1]) ≈ 1.0 + 0.0im
                @test QuditClifford.expect_int!(tab_nophase, Int[0, 1]) == 0

                # Operator structs and non-Vector AbstractVector inputs should also work.
                @test QuditClifford.expect!(tab, QuditClifford.SinglePauli(1, 0, 1)) ≈ 1.0 + 0.0im
                @test QuditClifford.expect_int!(tab, QuditClifford.SinglePauli(1, 1, 0)) == -1
                @test QuditClifford.expect!(tab, QuditClifford.SinglePauli(1, 0, 1, 1)) ≈ ω
                @test QuditClifford.expect_int!(tab, QuditClifford.SinglePauli(1, 0, 1, 1)) == 1

                op_view = view(Int[0, 1, 1], :)
                @test QuditClifford.expect!(tab, op_view) ≈ ω
                @test QuditClifford.expect_int!(tab, op_view) == 1
            end
        end
    end
end

@testset "Expectation values agree across Pauli representations" begin
    dense = GeneralPauli(Int[0, 0, 0, 1, 2, 1], 2)
    triple = TriplePauli(1, 0, 1, 2, 0, 2, 3, 0, 1, 2)
    sparse = NPauli((1, 2, 3), (0, 0, 0), (1, 2, 1), 2)

    for (label, TT) in [
        ("StabilizerTableau", StabilizerTableau),
        ("DestabilizerTableau", DestabilizerTableau),
    ]
        @testset "$label" begin
            tab = TT(3, 3; state=:product, basis=:Z)
            for op in (dense, triple, sparse)
                @test expect_int!(tab, op) == 2
                @test expect!(tab, op) ≈ cis(4π / 3)
            end

            # An X component takes each representation outside the Z-stabilizer span.
            dense_outside = GeneralPauli(Int[1, 0, 0, 1, 2, 1], 0)
            triple_outside = TriplePauli(1, 1, 1, 2, 0, 2, 3, 0, 1)
            sparse_outside = NPauli((1, 2, 3), (1, 0, 0), (1, 2, 1))
            for op in (dense_outside, triple_outside, sparse_outside)
                @test expect_int!(tab, op) == -1
                @test expect!(tab, op) == 0.0 + 0.0im
            end
        end
    end
end
