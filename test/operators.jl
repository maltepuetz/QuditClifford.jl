using QuditClifford
using Test

@testset "Few-qudit Pauli rendering" begin
    @test sprint(show, SinglePauli(2, 1, 0)) == "X₂"
    @test sprint(show, SinglePauli(2, 2, 3)) == "X₂² Z₂³"
    @test sprint(show, DoublePauli(1, 1, 0, 3, 0, 1)) == "X₁ Z₃"
    @test sprint(show, TriplePauli(1, 1, 0, 2, 0, 1, 3, 1, 1)) == "X₁ Z₂ X₃ Z₃"
    @test sprint(show, NPauli((1, 2, 4, 7), (1, 0, 1, 0), (0, 1, 1, 2))) ==
          "X₁ Z₂ X₄ Z₄ Z₇²"

    @test sprint(show, SinglePauli(1, 0, 1, 2)) == "[phase=2] Z₁"
    @test sprint(show, NPauli((), (), ())) == "I"
    @test sprint(show, NPauli((), (), (), 3)) == "[phase=3] I"

    # Exponents are displayed as stored because reduction depends on the
    # dimension of the tableau that eventually consumes the operator.
    @test sprint(show, SinglePauli(10, -1, 12)) == "X₁₀⁻¹ Z₁₀¹²"
    @test repr(MIME"text/plain"(), SinglePauli(2, 2, 3)) == "X₂² Z₂³"
    @test !showable(MIME"text/latex"(), SinglePauli(2, 2, 3))
    @test !showable(MIME"text/markdown"(), SinglePauli(2, 2, 3))
end

@testset "General Pauli rendering" begin
    @test sprint(show, GeneralPauli([1, 0, 0, 0, 1, 0], 0)) == "X₁ Z₂"
    @test sprint(show, GeneralPauli([0, 2, 0, 3, 0, -1], 0)) == "Z₁³ X₂² Z₃⁻¹"
    @test sprint(show, GeneralPauli([0, 0, 0, 1], 2)) == "[phase=2] Z₂"
    @test sprint(show, GeneralPauli([0, 0, 0, 0], 0)) == "I"
    @test sprint(show, GeneralPauli(Int[], 3)) == "[phase=3] I"
    @test repr(MIME"text/plain"(), GeneralPauli([1, 0, 0, 1], 0)) == "X₁ Z₂"
end

@testset "General Pauli construction" begin
    # Vector{Int} inputs are retained without a copy, while other integer
    # vector representations are normalized to the package's dense format.
    xz = Int[1, 0, 0, 1]
    op = GeneralPauli(xz, 3)
    @test op.xz === xz
    @test op.phase == 3

    xz_backing = Int[1, 0, -1, 2]
    xz_view = view(xz_backing, :)
    copied_op = GeneralPauli(xz_view, -2)
    @test copied_op.xz == Int[1, 0, -1, 2]
    @test copied_op.xz isa Vector{Int}
    xz_backing[1] = 9
    @test copied_op.xz[1] == 1

    # The n-aware constructor accepts both XZ-only and phase-bearing vectors.
    phase_bearing = view(Int[1, 0, 2, 0, 1, -1, 5], :)
    with_phase = GeneralPauli(3, 3, phase_bearing)
    @test with_phase.xz == Int[1, 0, 2, 0, 1, -1]
    @test with_phase.phase == 5

    xz_only = view(Int[1, 0, 2, 0, 1, -1, 99], 1:6)
    without_phase = GeneralPauli(3, 3, xz_only)
    @test without_phase.xz == Int[1, 0, 2, 0, 1, -1]
    @test without_phase.phase == 0

    @test_throws ArgumentError GeneralPauli(view(Int[1, 2, 3], :), 0)
    @test_throws ArgumentError GeneralPauli(2, 3, Int[1, 2, 3])
end

@testset "Operator materialization" begin
    tab = StabilizerTableau(3, 3; state=:mixed, storephase=true)
    operators_and_dense = [
        (
            SinglePauli(2, 4, -1, 5),
            Int[0, 1, 0, 0, 2, 0, 2],
        ),
        (
            DoublePauli(1, 4, -1, 3, -2, 5, 7),
            Int[1, 0, 1, 2, 0, 2, 1],
        ),
        (
            TriplePauli(1, 4, -1, 2, -2, 5, 3, 0, 4, 8),
            Int[1, 1, 0, 2, 2, 1, 2],
        ),
        (
            NPauli((1, 3), (4, -2), (-1, 5), 7),
            Int[1, 0, 1, 2, 0, 2, 1],
        ),
        (
            GeneralPauli(Int[4, 0, -2, -1, 0, 5], 7),
            Int[1, 0, 1, 2, 0, 2, 1],
        ),
    ]

    # Materializing into a workspace must clear unrelated entries, reduce XZ
    # exponents mod d, and reduce the phase with the appropriate modulus.
    for (op, expected) in operators_and_dense
        dst = fill(99, 7)
        @test QuditClifford.set_operator!(dst, tab, op) === nothing
        @test dst == expected

        dst_nophase = fill(99, 6)
        QuditClifford.set_operator!(dst_nophase, tab, op)
        @test dst_nophase == expected[1:6]
    end

    vector_op = view(Int[4, 0, -2, -1, 0, 5, 7], :)
    dst = fill(99, 7)
    QuditClifford.set_operator!(dst, tab, vector_op)
    @test dst == Int[1, 0, 1, 2, 0, 2, 1]

    # Omitting an input phase deterministically clears a phase-bearing target.
    QuditClifford.set_operator!(dst, tab, view(vector_op, 1:6))
    @test dst == Int[1, 0, 1, 2, 0, 2, 0]

    # Insertion into a tableau column follows exactly the same conversion rules.
    for (col, (op, expected)) in enumerate(operators_and_dense[1:3])
        QuditClifford.set_operator!(tab, col, op)
        @test tab.stab[:, col] == expected
    end
    QuditClifford.set_operator!(tab, 1, operators_and_dense[4][1])
    @test tab.stab[:, 1] == operators_and_dense[4][2]
    QuditClifford.set_operator!(tab, 2, vector_op)
    @test tab.stab[:, 2] == Int[1, 0, 1, 2, 0, 2, 1]
    QuditClifford.set_operator!(tab, 3, view(vector_op, 1:6))
    @test tab.stab[:, 3] == Int[1, 0, 1, 2, 0, 2, 0]
end
