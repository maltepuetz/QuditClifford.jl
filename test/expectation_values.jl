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

# The odd-prime span-phase path: phase_exponent_from_coeffs! accumulates zacc
# with addmul_mod! and then reads an ORDERED cross term x_j . zacc off it, so a
# generator pair whose cross term is nonzero is what actually exercises it.
# g1 = X_1 Z_2 and g2 = Z_1 X_2 commute, but neither cross term vanishes, and
# coefficients above one keep the aj multiplier off the multiply-free tier.
#
# Checked against a dense generalized-Pauli oracle rather than against the other
# tableau type: both types share phase_exponent_from_coeffs!, so a differential
# test moves both together and cannot see a sign error in it. Written with plain
# arrays so the test environment needs no LinearAlgebra.
@testset "Odd-prime span phase with an ordered cross term" begin
    # w^k X^x Z^z, the package's stored convention, as a dense matrix.
    function _paulimat(d, n, x, z, k)
        w = cispi(2 / d)
        X = zeros(ComplexF64, d, d)
        for j in 0:(d - 1)
            X[mod(j + 1, d) + 1, j + 1] = 1
        end
        Z = zeros(ComplexF64, d, d)
        for j in 0:(d - 1)
            Z[j + 1, j + 1] = w^j
        end
        _pow(A, t) = (B = Matrix{ComplexF64}(zeros(size(A))); for i in axes(A, 1); B[i, i] = 1; end;
                      for _ in 1:t; B = B * A; end; B)
        M = ComplexF64[1;;]
        for q in 1:n
            M = kron(M, _pow(X, mod(x[q], d)) * _pow(Z, mod(z[q], d)))
        end
        return (w^k) * M
    end
    _tr(A) = sum(A[i, i] for i in axes(A, 1))

    function _oracle(d, n, tab, x, z, k)
        D = d^n
        P = zeros(ComplexF64, D, D)
        for i in 1:D
            P[i, i] = 1
        end
        for j in 1:tab.m
            g = _paulimat(d, n, tab.stab[1:n, j], tab.stab[(n + 1):(2n), j], tab.stab[2n + 1, j])
            S = zeros(ComplexF64, D, D)
            gt = zeros(ComplexF64, D, D)
            for i in 1:D
                gt[i, i] = 1
            end
            for _ in 1:d
                S += gt
                gt = gt * g
            end
            P = P * (S / d)
        end
        v = _tr(P * _paulimat(d, n, x, z, k)) / _tr(P)
        abs(v) < 1e-9 && return -1
        w = cispi(2 / d)
        for m in 0:(d - 1)
            abs(v - w^m) < 1e-8 && return m
        end
        return -2                      # not a d-th root of unity: oracle failure
    end

    for d in (3, 5, 7)
        n = 2
        raw = zeros(Int, 2n + 1, n)
        raw[1, 1] = 1; raw[4, 1] = 1      # g1 = X_1 Z_2
        raw[2, 2] = 1; raw[3, 2] = 1      # g2 = Z_1 X_2

        st = StabilizerTableau(d, copy(raw); m=n, storephase=true)
        dt = DestabilizerTableau(d, copy(raw); m=n, storephase=true)

        # they commute, so this is a legal stabilizer group ...
        @test mod(QuditClifford.commutation_col(st.stab, 1, st.stab[:, 2]), d) == 0
        # ... and the ordered cross terms are genuinely nonzero
        @test QuditClifford.dot_xz_col(st.stab, n, 1, 2, d) == 1
        @test QuditClifford.dot_xz_col(st.stab, n, 2, 1, d) == 1

        for a in 1:(d - 1), b in 1:(d - 1)
            (a == 1 && b == 1) && continue          # want coefficients > 1
            # g1^a g2^b has x = (a, b) and z = (b, a)
            op = GeneralPauli(n, d, [a, b, b, a, 0])
            want = _oracle(d, n, StabilizerTableau(d, copy(raw); m=n, storephase=true),
                           [a, b], [b, a], 0)
            @test want >= 0                         # in span, and a valid root of unity
            @test expect_int!(st, op) == want
            @test expect_int!(dt, op) == want
        end
    end
end
