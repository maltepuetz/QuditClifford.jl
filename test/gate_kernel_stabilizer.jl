using QuditClifford
using Test
using Random
include(joinpath(@__DIR__, "env_switch.jl"))

# The `ly_` helpers live in test/layout_helpers.jl.
isdefined(Main, :ly_check_gate_stream) || include(joinpath(@__DIR__, "layout_helpers.jl"))

# A seeded stream of gates through the qudit-major kernel, compared with
# generator-major, on a `StabilizerTableau`; test/gate_kernel_destabilizer.jl
# runs the other tableau type.
@testset "qudit-major gate kernel, StabilizerTableau" begin
    for (TT, d, E, n, storephase, seed) in ly_gate_kernel_cases()
        TT === StabilizerTableau || continue
        @test ly_check_gate_stream(TT, d, E, n, storephase, seed)
    end
end
