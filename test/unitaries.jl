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

phase_modulus_for(d) = d == 2 ? 4 : d

# Symplectic form check on the column-tuple representation.
function is_symplectic(F::NTuple{S,NTuple{S,Int}}, d::Int) where {S}
    k = S ÷ 2
    pair(u, v) = mod(sum(u[i] * v[k+i] - u[k+i] * v[i] for i in 1:k), d)
    for i in 1:S, j in 1:S
        expected = (i <= k && j == i + k) ? 1 : (j <= k && i == j + k) ? d - 1 : 0
        pair(F[i], F[j]) == expected || return false
    end
    return true
end

@testset "Named gate data" begin
    jit = QC.JustInTimeInvMod()
    for d in (2, 3, 5)
        gates = Any[Fourier(1), Phase(1), PauliGate(1, 1, 1),
                    SUM(1, 2, 1), CPhase(1, 2, 1), SWAP(1, 2)]
        d > 2 && push!(gates, Multiplier(1, 2))
        for g in gates
            targets, F, a = QC._clifford_data(g, d, jit)
            @test is_symplectic(F, d)
            @test all(0 .<= a .< phase_modulus_for(d))
            @test all(all(0 .<= col .< d) for col in F)
        end
    end

    # Catalogue spot checks, spec section 4.
    targets, F, a = QC._clifford_data(Fourier(3), 5, jit)
    @test targets == (3,)
    @test QC._clifford_targets(Fourier(3)) == (3,)
    @test QC._clifford_targets(SUM(3, 1)) == (3, 1)
    @test QC._clifford_data(SUM(1, 2), 5, jit) == QC._clifford_data(SUM(1, 2, 1), 5, jit)
    @test QC._clifford_data(CPhase(1, 2), 5, jit) == QC._clifford_data(CPhase(1, 2, 1), 5, jit)
    @test F == ((0, 1), (4, 0))          # X -> Z, Z -> X^(d-1)
    @test a == (0, 0)

    _, F, a = QC._clifford_data(Phase(1), 5, jit)
    @test F == ((1, 1), (0, 1))          # X -> XZ, Z -> Z
    @test a == (0, 0)
    _, _, a2 = QC._clifford_data(Phase(1), 2, jit)
    @test a2 == (1, 0)                   # X -> iXZ at d = 2

    _, _, a3 = QC._clifford_data(PauliGate(1, 1, 2), 5, jit)
    @test a3 == (2, 4)                   # (z0, -x0) mod 5
    _, _, a4 = QC._clifford_data(PauliGate(1, 1, 1), 2, jit)
    @test a4 == (2, 2)                   # (2z0, 2x0) mod 4

    _, F, _ = QC._clifford_data(SUM(1, 2, 1), 5, jit)
    @test F == ((1, 1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 4, 1))
    _, F, _ = QC._clifford_data(CPhase(1, 2, 1), 5, jit)
    @test F == ((1, 0, 0, 1), (0, 1, 1, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    _, F, _ = QC._clifford_data(SWAP(1, 2), 5, jit)
    @test F == ((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))

    # Multiplier inverts its coefficient, and rejects a zero residue.
    _, F, _ = QC._clifford_data(Multiplier(1, 2), 5, jit)
    @test F == ((2, 0), (0, 3))          # 2 * 3 = 1 mod 5
    @test_throws ArgumentError QC._clifford_data(Multiplier(1, 5), 5, jit)
    @test_throws ArgumentError QC._clifford_data(Multiplier(1, 0), 3, jit)

    # Parameters are normalized before arithmetic, so extreme inputs are safe.
    _, _, a5 = QC._clifford_data(PauliGate(1, typemin(Int), typemin(Int)), 3, jit)
    @test all(0 .<= a5 .< 3)
end

@testset "Gate construction validation" begin
    @test_throws ArgumentError Fourier(0)
    @test_throws ArgumentError Phase(-1)
    @test_throws ArgumentError SUM(2, 2, 1)       # equal targets
    @test_throws ArgumentError CPhase(1, 1, 1)
    @test_throws ArgumentError SWAP(3, 3)
    @test_throws ArgumentError Multiplier(0, 2)
end

@testset "Gate printing" begin
    @test sprint(show, Fourier(1)) == "Fourier(1)"
    @test sprint(show, Phase(2)) == "Phase(2)"
    @test sprint(show, Multiplier(1, 3)) == "Multiplier(1, 3)"
    @test sprint(show, PauliGate(2, 1, 1)) == "PauliGate(2, 1, 1)"
    @test sprint(show, SUM(1, 3, 2)) == "SUM(1, 3, 2)"
    @test sprint(show, CPhase(1, 3, 2)) == "CPhase(1, 3, 2)"
    @test sprint(show, SWAP(1, 3)) == "SWAP(1, 3)"
end
