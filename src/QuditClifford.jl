"""
    QuditClifford

Tools for prime-dimensional qudit stabilizer states represented by stabilizer
and destabilizer tableaux. The package supports pure and mixed states, Pauli
measurements, expectation values, canonicalization, and stabilizer
entanglement entropy.
"""
module QuditClifford

export AbstractTableau, StabilizerTableau, DestabilizerTableau,
    reset!, entanglement_entropy, measure!, AbstractPauli, FewQuditPauli, GeneralPauli,
    SinglePauli, DoublePauli, TriplePauli, NPauli, canonicalize!, expect!, expect_int!,
    is_pure

import Random, Primes
import Random.rand!
import LoopVectorization
import LoopVectorization.@turbo


# include the sub files
include("helper.jl")
include("inversemod.jl")
include("abstract_tableau.jl")
include("stabilizer_tableau.jl")
include("destabilizer_tableau.jl")
include("canonicalize.jl")
include("operator.jl")
include("span_decomposition.jl")
include("projective_measurement.jl")
include("entanglement_entropy.jl")
include("check_purity.jl")
include("expectation_value.jl")

end
