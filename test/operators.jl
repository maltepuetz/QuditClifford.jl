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
