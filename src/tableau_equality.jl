##### state equality and hashing for tableaux #####
#
# Two tableaux are equal when they describe the same stabilizer state: the same
# `d`, `n` and `storephase`, and generators spanning the same group with the
# same phases. The generator basis, the destabilizers, `iscanonical` and every
# workspace are representation, not state, and never enter `==` or `hash`.
#
# The comparison goes through the canonical form. Reduced column echelon form
# with unit pivots is unique for a given span of XZ vectors, and a stabilizer
# group holds each of its Pauli strings with exactly one phase, so the canonical
# generators, phases included, are a complete invariant of the state. Neither
# function modifies its arguments: a tableau not already canonical is
# canonicalized in a fresh copy.

# Generator columns `1:m` of `A` and `B` agree entry for entry over `rows`.
function _active_columns_equal(A::Matrix{Int}, B::Matrix{Int}, rows::Int, m::Int)
    @inbounds for j in 1:m, i in 1:rows
        A[i, j] == B[i, j] || return false
    end
    return true
end

# The tableau's own matrix when it is already canonical, otherwise a canonical
# copy. The scratch passed to `_canonicalize_tableau!` is fresh, so the
# tableau's pivot table and x·z cache are left alone.
function _canonical_stab(tab::AbstractTableau)
    tab.iscanonical && return tab.stab
    S = copy(tab.stab)
    _canonicalize_tableau!(S, tab.n, tab.m, tab.d, tab.storephase, tab.inversemod,
                           zeros(Int, 2 * tab.n), zeros(Int, tab.n))
    return S
end

"""
    ==(a::AbstractTableau, b::AbstractTableau) -> Bool

Whether `a` and `b` describe the same stabilizer state.

The tableaux are equal when they have the same `d`, `n` and `storephase`, and
their generators span the same stabilizer group with the same phases. The
choice of generator basis does not matter, so a tableau equals its own
[`canonicalize!`](@ref)d form, and a [`StabilizerTableau`](@ref) can equal a
[`DestabilizerTableau`](@ref). Destabilizers, `iscanonical` and the inversion
strategy are ignored. With `storephase=false` only the group of Pauli strings
is compared.

`hash` agrees with `==`, so equal states collapse in a `Set` or `Dict`. A
tableau used as a key must not be mutated afterwards, since mutation changes
its hash.

# Notes
- Neither argument is modified. When the generator columns already agree entry
  for entry, or both tableaux are canonical, the comparison is `O(nm)` and does
  not allocate. Otherwise each non-canonical tableau is canonicalized in a
  temporary copy, in `O(n m²)` time; `hash` does the same.
- The comparison relies on the generators forming a valid stabilizer group. A
  qubit tableau holding non-Hermitian generators, which `measure!` can store
  with `phase_policy=2`, is not covered.

# Examples
```julia
a = StabilizerTableau(3, 4; state=:ghz)
b = copy(a)
canonicalize!(b)
a == b                                   # true: same state, possibly another basis
a == DestabilizerTableau(3, 4; state=:ghz)   # true
a == StabilizerTableau(3, 4; state=:product) # false
```
"""
function Base.:(==)(a::AbstractTableau, b::AbstractTableau)
    a === b && return true
    (a.d == b.d && a.n == b.n && a.m == b.m && a.storephase == b.storephase) ||
        return false
    rows = size(a.stab, 1)
    m = a.m
    _active_columns_equal(a.stab, b.stab, rows, m) && return true
    a.iscanonical && b.iscanonical && return false
    return _active_columns_equal(_canonical_stab(a), _canonical_stab(b), rows, m)
end

function Base.hash(tab::AbstractTableau, h::UInt)
    h = hash(tab.storephase, hash(tab.m, hash(tab.n, hash(tab.d, hash(:AbstractTableau, h)))))
    S = _canonical_stab(tab)
    @inbounds for j in 1:tab.m, i in axes(S, 1)
        h = hash(S[i, j], h)
    end
    return h
end
