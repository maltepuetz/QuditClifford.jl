using QuditClifford
using Test
using Aqua

@testset "QuditClifford.jl" begin
    # Aqua checks (skip piracy on Julia ≥ 1.12 because Aqua's piracy
    # checker currently fails there due to an upstream API change).
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(QuditClifford; piracies = VERSION < v"1.12.0")
    end

    # Functional tests split by topic for clarity.
    include("measurements.jl")
    include("expectation_values.jl")
    include("entanglement.jl")
    include("purity.jl")
end
