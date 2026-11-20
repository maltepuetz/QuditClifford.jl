using QuditClifford
using Test
using Aqua

# The layout every constructor uses without a `layout` keyword, so the whole
# suite runs generator-major (the default) or qudit-major.
let layout = get(ENV, "QC_TEST_LAYOUT", "generator")
    layout in ("generator", "qudit") ||
        error("QC_TEST_LAYOUT must be \"generator\" or \"qudit\", got \"$layout\"")
    QuditClifford._DEFAULT_LAYOUT[] = Symbol(layout)
    println("Testing with default tableau layout :", layout)
end

@testset "QuditClifford.jl" begin
    # Aqua checks (skip piracy on Julia ≥ 1.12 because Aqua's piracy
    # checker currently fails there due to an upstream API change).
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(QuditClifford; piracies = VERSION < v"1.12.0")
    end

    # Functional tests split by topic for clarity.
    include("modular.jl")
    include("tableau_storage.jl")
    include("storage_types.jl")
    include("layout.jl")
    include("operators.jl")
    include("measurements.jl")
    include("measurement_kernel.jl")
    include("commuting_measurement.jl")
    include("channels.jl")
    include("expectation_values.jl")
    include("destabilizers.jl")
    include("entanglement.jl")
    include("entanglement_profile.jl")
    include("rank_workspaces.jl")
    include("entanglement_negativity.jl")
    include("mutual_information.jl")
    include("conditional_mutual_information.jl")
    include("purity.jl")
    include("initialization.jl")
    include("unitaries.jl")
    include("clifford_operators.jl")
    include("random_clifford.jl")
    include("dense_states.jl")
    include("tableau_copy_equality.jl")
    include("differential.jl")
end
