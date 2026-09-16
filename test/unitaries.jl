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

using LinearAlgebra

# ---- independent oracle: explicit matrices, ZX = ωXZ ----
oracle_ω(d) = cispi(2 / d)
oracle_ζ(d) = d == 2 ? im : oracle_ω(d)
oracle_X(d) = ComplexF64[(mod(i - j - 1, d) == 0) ? 1 : 0 for i in 0:d-1, j in 0:d-1]
oracle_Z(d) = Matrix(Diagonal(ComplexF64[oracle_ω(d)^j for j in 0:d-1]))
oracle_pow(M, e) = e == 0 ? Matrix{ComplexF64}(I, size(M, 1), size(M, 1)) : M^e

function oracle_pauli(d, x, z)
    M = ones(ComplexF64, 1, 1)
    for q in eachindex(x)
        M = kron(M, oracle_pow(oracle_X(d), mod(x[q], d)) *
                    oracle_pow(oracle_Z(d), mod(z[q], d)))
    end
    return M
end

oracle_index(v, d) = foldl((acc, x) -> acc * d + x, v; init = 0)

function oracle_unitary(g, d)
    if g isa Fourier
        return ComplexF64[oracle_ω(d)^(i * j) / sqrt(d) for i in 0:d-1, j in 0:d-1]
    elseif g isa Phase
        d == 2 && return ComplexF64[1 0; 0 im]
        i2 = invmod(2, d)
        return Matrix(Diagonal(ComplexF64[oracle_ω(d)^mod(j * (j - 1) * i2, d)
                                          for j in 0:d-1]))
    elseif g isa Multiplier
        return ComplexF64[(mod(i - g.a * j, d) == 0) ? 1 : 0 for i in 0:d-1, j in 0:d-1]
    elseif g isa PauliGate
        return oracle_pow(oracle_X(d), mod(g.x, d)) * oracle_pow(oracle_Z(d), mod(g.z, d))
    elseif g isa SUM
        M = zeros(ComplexF64, d^2, d^2)
        for u in 0:d-1, v in 0:d-1
            M[oracle_index([u, mod(v + g.a * u, d)], d) + 1,
              oracle_index([u, v], d) + 1] = 1
        end
        return M
    elseif g isa CPhase
        M = zeros(ComplexF64, d^2, d^2)
        for u in 0:d-1, v in 0:d-1
            i = oracle_index([u, v], d) + 1
            M[i, i] = oracle_ω(d)^mod(g.a * u * v, d)
        end
        return M
    else # SWAP
        M = zeros(ComplexF64, d^2, d^2)
        for u in 0:d-1, v in 0:d-1
            M[oracle_index([v, u], d) + 1, oracle_index([u, v], d) + 1] = 1
        end
        return M
    end
end

# ---- reference phase polynomial, spec section 3.1 ----
function phase_reference(F, a, v, d)
    S = length(v); K = S ÷ 2
    p = d == 2 ? 4 : d
    γ = d == 2 ? 2 : 1
    colxz(col) = sum(big(col[q]) * col[K + q] for q in 1:K; init = big(0))
    lin = sum(big(a[i]) * v[i] for i in 1:S; init = big(0))
    quad = sum(binomial(big(v[i]), 2) * colxz(F[i]) for i in 1:S; init = big(0))
    quad += sum(sum(big(F[i][K + r]) * F[j][r] for r in 1:K; init = big(0)) * v[i] * v[j]
                for i in 1:S for j in (i + 1):S; init = big(0))
    return Int(mod(lin + γ * quad, p))
end

digits_of(i, d, n) = (v = zeros(Int, n); for q in n:-1:1; v[q] = i % d; i ÷= d; end; v)

catalogue(d) = begin
    gs = Any[Fourier(1), Phase(1), PauliGate(1, 1, 1), PauliGate(1, 0, 1),
             SUM(1, 2, 1), CPhase(1, 2, 1), SWAP(1, 2)]
    if d > 2
        append!(gs, Any[Multiplier(1, 2), SUM(1, 2, d - 1), CPhase(1, 2, d - 1)])
    end
    gs
end

@testset "Phase evaluators against polynomial and matrices" begin
    jit = QC.JustInTimeInvMod()
    for d in (2, 3, 5)
        for g in catalogue(d)
            targets, F, a = QC._clifford_data(g, d, jit)
            K = length(targets); S = 2K
            fast = QC.clifford_fast_dots(S, d)
            D = QC._image_xdotz(F, K, d, fast)
            Dsafe = QC._image_xdotz(F, K, d, false)
            @test Dsafe == D
            inv2 = d == 2 ? 0 : jit(2, d)
            U = oracle_unitary(g, d)
            for vi in 0:d^S-1
                v = ntuple(i -> digits_of(vi, d, S)[i], S)
                vout = ntuple(j -> mod(sum(F[i][j] * v[i] for i in 1:S), d), S)
                φ = d == 2 ? QC._phase_qubit(v, F, a, K) :
                             QC._phase_odd(v, vout, a, D, K, d, inv2, fast)
                @test φ == phase_reference(F, a, v, d)
                if d != 2
                    @test QC._phase_odd(v, vout, a, Dsafe, K, d, inv2, false) == φ
                end
                lhs = U * oracle_pauli(d, v[1:K], v[K+1:S]) * U'
                rhs = oracle_ζ(d)^φ * oracle_pauli(d, vout[1:K], vout[K+1:S])
                @test norm(lhs - rhs) < 1e-9
            end
        end
    end
end

@testset "Exhaustive one-qudit Clifford enumeration" begin
    # Spec 9.2: enumerate every admissible one-qudit (F, a) at d = 2, 3
    # (24 and 216 Cliffords), rather than spot-checking only the named
    # catalogue -- of the seven shipped gates only Phase has a nonzero D,
    # so this is what actually exercises the D·v term and the qubit
    # ordered-product accumulation order broadly.

    # d = 2: brute force over all 16 candidate column pairs, keep the
    # symplectic ones (Sp(2, Z_2) = SL(2, Z_2), order 6), then every
    # admissible raw phase (a_i ≡ D_i mod 2, spec 3.3): 2 choices per
    # generator, 4 per F, 24 Cliffords total.
    symplectics2 = NTuple{2,NTuple{2,Int}}[]
    for x1 in 0:1, z1 in 0:1, x2 in 0:1, z2 in 0:1
        F = ((x1, z1), (x2, z2))
        is_symplectic(F, 2) && push!(symplectics2, F)
    end
    @test length(symplectics2) == 6

    total2 = 0
    for F in symplectics2
        D = QC._image_xdotz(F, 1, 2, QC.clifford_fast_dots(2, 2))
        admissible2 = NTuple{2,Int}[]
        for a1 in 0:3, a2 in 0:3
            (mod(a1, 2) == mod(D[1], 2) && mod(a2, 2) == mod(D[2], 2)) &&
                push!(admissible2, (a1, a2))
        end
        @test length(admissible2) == 4
        for a in admissible2
            total2 += 1
            for v1 in 0:1, v2 in 0:1
                v = (v1, v2)
                @test QC._phase_qubit(v, F, a, 1) == phase_reference(F, a, v, 2)
            end
        end
    end
    @test total2 == 24

    # d = 3: every F with FᵀΩF = Ω mod 3 (SL(2, Z_3), order 24); every raw
    # phase a ∈ {0,1,2}² is admissible at odd d (no Hermiticity constraint),
    # so 24 × 9 = 216 Cliffords.
    symplectics3 = NTuple{2,NTuple{2,Int}}[]
    for x1 in 0:2, z1 in 0:2, x2 in 0:2, z2 in 0:2
        F = ((x1, z1), (x2, z2))
        is_symplectic(F, 3) && push!(symplectics3, F)
    end
    @test length(symplectics3) == 24

    jit = QC.JustInTimeInvMod()
    inv2_3 = jit(2, 3)
    fast3 = QC.clifford_fast_dots(2, 3)
    total3 = 0
    for F in symplectics3
        D = QC._image_xdotz(F, 1, 3, fast3)
        Dsafe = QC._image_xdotz(F, 1, 3, false)
        @test Dsafe == D
        for a1 in 0:2, a2 in 0:2
            a = (a1, a2)
            total3 += 1
            for v1 in 0:2, v2 in 0:2
                v = (v1, v2)
                vout = ntuple(j -> mod(sum(F[i][j] * v[i] for i in 1:2), 3), 2)
                φ = QC._phase_odd(v, vout, a, D, 1, 3, inv2_3, fast3)
                @test φ == phase_reference(F, a, v, 3)
                @test QC._phase_odd(v, vout, a, Dsafe, 1, 3, inv2_3, false) == φ
            end
        end
    end
    @test total3 == 216
end

@testset "Phase evaluation at a large dimension" begin
    # A prime near the two-term accumulation boundary on each supported host.
    d = Sys.WORD_SIZE == 64 ? 2147483647 : 32749
    jit = QC.JustInTimeInvMod()
    targets, F, a = QC._clifford_data(Phase(1), d, jit)
    fast = QC.clifford_fast_dots(2, d)
    @test fast                                   # k = 1 is inside the guard
    D = QC._image_xdotz(F, 1, d, fast)
    v = (d - 1, 0)
    vout = ntuple(j -> mod(sum(F[i][j] * v[i] for i in 1:2), d), 2)
    # C(d-1, 2) = 1 mod d. Multiplying the unreduced bracket by inv2 before
    # reducing -- instead of reducing at each step, as `_phase_odd` does --
    # overflows Int and returns 1073741828 instead on 64-bit hosts (not
    # 1073741824, which is just inv2 itself); Task 4 tests the four-term
    # matvec too.
    @test QC._phase_odd(v, vout, a, D, 1, d, jit(2, d), fast) == 1
    @test QC._phase_odd(v, vout, a, QC._image_xdotz(F, 1, d, false),
                       1, d, jit(2, d), false) == 1
    @test !QC.clifford_fast_dots(4, d)
end

@testset "Qubit bitmask bound is enforced" begin
    F = ((1, 0), (0, 1))
    @test QC._phase_qubit((1, 1), F, (0, 0), 1) == 0
    # The bound is a precondition of the qubit regime, enforced once when
    # preparation resolves it -- not re-tested inside the column loop, and not
    # reachable through the public API, where every named gate has K ≤ 2.
    @test QC._check_qubit_bitmask_bound(0) === nothing
    @test QC._check_qubit_bitmask_bound(64) === nothing
    @test_throws ArgumentError QC._check_qubit_bitmask_bound(65)
    @test_throws ArgumentError QC._check_qubit_bitmask_bound(-1)
end
