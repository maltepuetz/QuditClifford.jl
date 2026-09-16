using QuditClifford
using Test

const QC = QuditClifford

@testset "Safe scalar modular arithmetic" begin
    moduli = [2, 3, 4, 11, isqrt(typemax(Int)), isqrt(typemax(Int)) + 2, typemax(Int)]
    for M in moduli
        for a in unique([0, 1, M ÷ 2, M - 1]), b in unique([0, 1, M ÷ 2, M - 1])
            @test QC.add_mod(a, b, M) == mod(big(a) + b, M)
            @test QC.sub_mod(a, b, M) == mod(big(a) - b, M)
            @test QC.mul_mod(a, b, M) == mod(big(a) * b, M)
        end
    end
end

@testset "Dot tier guard" begin
    @test QC.clifford_fast_dots(0, 2)
    # Qubit raw phases reach 3 even though exponent coordinates are bits.
    @test QC.clifford_fast_dots(4, 2)
    if Sys.WORD_SIZE == 64
        d = 2147483647
        @test QC.clifford_fast_dots(2, d)
        @test !QC.clifford_fast_dots(4, d)
    end
    # Cover acceptance AND rejection on the actual host, with BigInt before
    # squaring. These boundaries work on the x86 CI jobs as well as x64.
    for S in (2, 4, 8, 64)
        boundary = isqrt(typemax(Int) ÷ S)
        for d in unique([2, 3, 5, 1031, boundary + 1, boundary + 2])
            B = max(d - 1, (d == 2 ? 4 : d) - 1)
            expected = big(S) * big(B)^2 <= typemax(Int)
            @test QC.clifford_fast_dots(S, d) == expected
        end
    end
end
