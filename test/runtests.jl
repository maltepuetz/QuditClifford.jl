using QuditClifford
using Test
using Aqua

# The shipped defaults: generator-major, and `UInt8` entries at d = 2.
@test QuditClifford._DEFAULT_LAYOUT[] === :generator
@test QuditClifford._default_qubit_eltype() === QuditClifford._default_eltype(2) === UInt8

# QC_TEST_LAYOUT and QC_TEST_ELTYPE: the default layout and the default element
# type at d = 2.
include("env_switch.jl")
println("Testing with default tableau layout :", QuditClifford._DEFAULT_LAYOUT[])
println("Testing with default qubit element type ",
        Base.invokelatest(QuditClifford._default_qubit_eltype))

# The part of the suite to run, so CI can split it over parallel jobs:
# "all" (the default), "kernel", "functional", or one part of either,
# "kernel-1", "kernel-2" and "kernel-3", "functional-1" and "functional-2".
# The kernel parts are for the CI jobs with coverage, where layout.jl and the
# two gate-kernel files run two to three times slower, relative to the other
# kernel files, than on Apple silicon; each part holds one of those three.
#
# The kernel files compare storage types, layouts and packed storage with each
# other and build most of their tableaux explicitly; a default enters only on
# the axis a test does not vary. CI runs them with the shipped defaults and
# once with both switched. The functional files take both defaults throughout
# and run in all four configurations.
const KERNEL_FILES_1 = [
    "modular.jl",
    "layout.jl",
    "storage_types.jl",
]
const KERNEL_FILES_2 = [
    "gate_kernel_stabilizer.jl",
    "bits.jl",
]
const KERNEL_FILES_3 = [
    "tableau_storage.jl",
    "gate_kernel_destabilizer.jl",
]
const KERNEL_FILES = vcat(KERNEL_FILES_1, KERNEL_FILES_2, KERNEL_FILES_3)
const FUNCTIONAL_FILES_1 = [
    "operators.jl",
    "measurements.jl",
    "measurement_kernel.jl",
    "commuting_measurement.jl",
    "channels.jl",
    "expectation_values.jl",
    "destabilizers.jl",
    "entanglement.jl",
    "entanglement_profile.jl",
]
const FUNCTIONAL_FILES_2 = [
    "rank_workspaces.jl",
    "entanglement_negativity.jl",
    "mutual_information.jl",
    "conditional_mutual_information.jl",
    "purity.jl",
    "initialization.jl",
    "unitaries.jl",
    "clifford_operators.jl",
    "random_clifford.jl",
    "dense_states.jl",
    "tableau_copy_equality.jl",
    "differential.jl",
]
const SHARDS = Dict(
    "all" => vcat(KERNEL_FILES, FUNCTIONAL_FILES_1, FUNCTIONAL_FILES_2),
    "kernel" => KERNEL_FILES,
    "kernel-1" => KERNEL_FILES_1,
    "kernel-2" => KERNEL_FILES_2,
    "kernel-3" => KERNEL_FILES_3,
    "functional" => vcat(FUNCTIONAL_FILES_1, FUNCTIONAL_FILES_2),
    "functional-1" => FUNCTIONAL_FILES_1,
    "functional-2" => FUNCTIONAL_FILES_2,
)
const SHARD = get(ENV, "QC_TEST_SHARD", "all")
haskey(SHARDS, SHARD) ||
    error("QC_TEST_SHARD must be one of $(join(sort!(collect(keys(SHARDS))), ", ")), got \"$SHARD\"")
println("Testing shard ", SHARD)

# Every test file is in exactly one of the five lists; the other files in
# test/ are this runner and the helpers the test files include.
@test sort!(vcat(KERNEL_FILES, FUNCTIONAL_FILES_1, FUNCTIONAL_FILES_2)) ==
      sort!(filter(f -> endswith(f, ".jl") &&
                        !(f in ("runtests.jl", "env_switch.jl", "storage_helpers.jl",
                                "layout_helpers.jl")),
                   readdir(@__DIR__)))

@testset "QuditClifford.jl" begin
    # Aqua checks (skip piracy on Julia ≥ 1.12 because Aqua's piracy
    # checker currently fails there due to an upstream API change). They do
    # not depend on the test configuration, so only the shards that include
    # kernel-1 run them.
    if SHARD in ("all", "kernel", "kernel-1")
        seconds = @elapsed @testset "Code quality (Aqua.jl)" begin
            Aqua.test_all(QuditClifford; piracies = VERSION < v"1.12.0")
        end
        println("Aqua: ", round(seconds; digits = 1), " s")
    end

    # The test files, split by topic. Each prints its time, so a CI log shows
    # where a job's time goes.
    for file in SHARDS[SHARD]
        seconds = @elapsed include(file)
        println(file, ": ", round(seconds; digits = 1), " s")
        flush(stdout)
    end
end
