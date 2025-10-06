module QuditClifford

export StabilizerTableau, entanglement_entropy, measure!, Operator

import Random, Primes
import Random.rand!

using AllocCheck


# include the sub files
include("inversemod.jl")
include("stabilizer_tableau.jl")
include("operator.jl")
include("generator.jl")
include("projective_measurement.jl")
include("entanglement_entropy.jl")
include("check_purity.jl")

end
