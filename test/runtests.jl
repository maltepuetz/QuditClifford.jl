using QuditClifford
using Test
using Aqua

# The shipped defaults: generator-major, and `UInt8` entries at d = 2.
@test QuditClifford._DEFAULT_LAYOUT[] === :generator
@test QuditClifford._default_qubit_eltype() === QuditClifford._default_eltype(2) === UInt8

# The layout every constructor uses without a `layout` keyword, so the whole
# suite runs generator-major (the default) or qudit-major.
let layout = get(ENV, "QC_TEST_LAYOUT", "generator")
    layout in ("generator", "qudit") ||
        error("QC_TEST_LAYOUT must be \"generator\" or \"qudit\", got \"$layout\"")
    QuditClifford._DEFAULT_LAYOUT[] = Symbol(layout)
    println("Testing with default tableau layout :", layout)
end

# The element type every d = 2 constructor uses without an `eltype` keyword, so
# the whole suite runs on `UInt8` entries (the default) or on packed `Bits`.
let eltype = get(ENV, "QC_TEST_ELTYPE", "uint8")
    eltype in ("uint8", "bits") ||
        error("QC_TEST_ELTYPE must be \"uint8\" or \"bits\", got \"$eltype\"")
    # The shipped method is deleted before the replacement is defined, so no
    # method is overwritten (which `Pkg.test` would report as a warning).
    if eltype == "bits"
        Base.delete_method(which(QuditClifford._default_qubit_eltype, ()))
        @eval QuditClifford _default_qubit_eltype() = Bits
    end
    println("Testing with default qubit element type ",
            Base.invokelatest(QuditClifford._default_qubit_eltype))
end

# The part of the suite to run, so CI can split it over parallel jobs:
# "all" (the default), "kernel", "functional", or one half of the functional
# files, "functional-1" and "functional-2".
#
# The kernel files compare storage types, layouts and packed storage with each
# other and build most of their tableaux explicitly; a default enters only on
# the axis a test does not vary. CI runs them with the shipped defaults and
# once with both switched. The functional files take both defaults throughout
# and run in all four configurations.
const KERNEL_FILES = [
    "modular.jl",
    "tableau_storage.jl",
    "storage_types.jl",
    "layout.jl",
    "bits.jl",
]
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
    "functional" => vcat(FUNCTIONAL_FILES_1, FUNCTIONAL_FILES_2),
    "functional-1" => FUNCTIONAL_FILES_1,
    "functional-2" => FUNCTIONAL_FILES_2,
)
const SHARD = get(ENV, "QC_TEST_SHARD", "all")
haskey(SHARDS, SHARD) ||
    error("QC_TEST_SHARD must be one of $(join(sort!(collect(keys(SHARDS))), ", ")), got \"$SHARD\"")
println("Testing shard ", SHARD)

# Every test file is in exactly one of the three lists.
@test sort!(vcat(KERNEL_FILES, FUNCTIONAL_FILES_1, FUNCTIONAL_FILES_2)) ==
      sort!(filter(f -> endswith(f, ".jl") && !(f in ("runtests.jl", "storage_helpers.jl")),
                   readdir(@__DIR__)))

@testset "QuditClifford.jl" begin
    # Aqua checks (skip piracy on Julia ≥ 1.12 because Aqua's piracy
    # checker currently fails there due to an upstream API change). They do
    # not depend on the test configuration, so the functional shards skip them.
    if SHARD in ("all", "kernel")
        @testset "Code quality (Aqua.jl)" begin
            Aqua.test_all(QuditClifford; piracies = VERSION < v"1.12.0")
        end
    end

    # Functional tests split by topic for clarity.
    for file in SHARDS[SHARD]
        include(file)
    end
end
