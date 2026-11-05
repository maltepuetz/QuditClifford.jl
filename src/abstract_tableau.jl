"""
    AbstractTableau

Abstract supertype for stabilizer-tableau representations.

Concrete implementations are [`DestabilizerTableau`](@ref) and
[`StabilizerTableau`](@ref). They represent stabilizer density operators for
prime-dimensional qudits and share the mutating measurement, expectation,
canonicalization, and entropy interfaces.
"""
abstract type AbstractTableau end

# `copy!` overwrites in place, so the destination must already have the
# source's dimension, size and phase layout. Checked before any write.
function _check_copy_compatible(dst::AbstractTableau, src::AbstractTableau)
    (dst.d == src.d && dst.n == src.n && dst.storephase == src.storephase) ||
        throw(ArgumentError(
            "copy! needs matching d, n and storephase; got destination " *
            "(d=$(dst.d), n=$(dst.n), storephase=$(dst.storephase)) and source " *
            "(d=$(src.d), n=$(src.n), storephase=$(src.storephase))."))
    return nothing
end
