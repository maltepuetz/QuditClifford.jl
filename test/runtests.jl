using QuditClifford
using Test
using Aqua

@testset "QuditClifford.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(QuditClifford)
    end
    # Write your tests here.
end
