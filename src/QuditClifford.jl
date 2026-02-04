module QuditClifford

export StabilizerTableau, entanglement_entropy, measure!, Operator, canonicalize!, expect!, expect_int!

import Random, Primes
import Random.rand!
import LoopVectorization
import LoopVectorization.@turbo


# include the sub files
include("helper.jl")
include("inversemod.jl")
include("stabilizer_tableau.jl")
include("canonicalize.jl")
include("operator.jl")
include("projective_measurement.jl")
include("entanglement_entropy.jl")
include("check_purity.jl")
include("expectation_value.jl")

end
