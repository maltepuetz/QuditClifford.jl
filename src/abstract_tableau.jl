"""
    AbstractTableau

Abstract supertype for stabilizer-tableau representations.

Concrete implementations are [`DestabilizerTableau`](@ref) and
[`StabilizerTableau`](@ref). They represent stabilizer density operators for
prime-dimensional qudits and share the mutating measurement, expectation,
canonicalization, and entropy interfaces.
"""
abstract type AbstractTableau end
