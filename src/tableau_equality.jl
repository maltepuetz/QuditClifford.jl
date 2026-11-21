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

# Generator columns `1:m` of the blocks `a` and `b` (unpacked or packed, through
# their leaves), with phases `pa` and `pb` when `storephase`, agree entry for
# entry.
function _active_columns_equal(a, pa::AbstractVector{Int},
                               b, pb::AbstractVector{Int},
                               m::Int, storephase::Bool)
    n = _nq(a)
    @inbounds for j in 1:m
        for q in 1:n
            _xget(a, q, j) == _xget(b, q, j) || return false
        end
        for q in 1:n
            _zget(a, q, j) == _zget(b, q, j) || return false
        end
        if storephase
            pa[j] == pb[j] || return false
        end
    end
    return true
end

# The same comparison for two qudit-major blocks, walked in storage order: the
# phases, then each site's column over generators `1:m`.
function _active_columns_equal(a::_QMBlock, pa::AbstractVector{Int},
                               b::_QMBlock, pb::AbstractVector{Int},
                               m::Int, storephase::Bool)
    if storephase
        @inbounds for j in 1:m
            pa[j] == pb[j] || return false
        end
    end
    return _qm_generators_equal(a, b, m)
end

# The tableau's own block and phase when it is already canonical, otherwise
# canonical copies. The scratch passed to `_canonicalize_tableau!` is fresh, so
# the tableau's pivot table and x·z cache are left alone.
function _canonical_stab(tab::AbstractTableau, L::TableauLayout)
    b, phase = _stab(tab, L), _phase(tab)
    tab.iscanonical && return b, phase
    S = _generator_major_copy(b)
    Sphase = copy(phase)
    _canonicalize_tableau!(S, Sphase, tab.m, tab.d, tab.storephase, tab.inversemod,
                           zeros(Int, 2 * tab.n), zeros(Int, tab.n))
    return S, Sphase
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
```jldoctest
julia> a = StabilizerTableau(3, 4; state = :ghz);

julia> b = copy(a);

julia> canonicalize!(b)

julia> a == b                                         # same state, possibly another basis
true

julia> a == DestabilizerTableau(3, 4; state = :ghz)
true

julia> a == StabilizerTableau(3, 4; state = :product)
false
```
"""
function Base.:(==)(a::AbstractTableau, b::AbstractTableau)
    a === b && return true
    (a.d == b.d && a.n == b.n && a.m == b.m && a.storephase == b.storephase) ||
        return false
    return _tableaux_equal(a, _layout_marker(a), b, _layout_marker(b))
end

# `==` for tableaux of equal `d`, `n`, `m` and `storephase`, with `La` and `Lb`
# the layout markers of `a` and `b`.
function _tableaux_equal(a::AbstractTableau, La::TableauLayout, b::AbstractTableau, Lb::TableauLayout)
    m, storephase = a.m, a.storephase
    _active_columns_equal(_stab(a, La), _phase(a), _stab(b, Lb), _phase(b), m, storephase) &&
        return true
    a.iscanonical && b.iscanonical && return false
    ca, pa = _canonical_stab(a, La)
    cb, pb = _canonical_stab(b, Lb)
    return _active_columns_equal(ca, pa, cb, pb, m, storephase)
end

# Mix the active generators `1:m` of `S` into `h`: per generator the X entries,
# then the Z entries, then the phase when `storephase`.
function _hash_active(h::UInt, S, phase::AbstractVector{Int},
                      m::Int, storephase::Bool)
    n = _nq(S)
    @inbounds for j in 1:m
        for q in 1:n
            h = hash(_xget(S, q, j), h)
        end
        for q in 1:n
            h = hash(_zget(S, q, j), h)
        end
        storephase && (h = hash(phase[j], h))
    end
    return h
end

function Base.hash(tab::AbstractTableau, h::UInt)
    h = hash(tab.storephase, hash(tab.m, hash(tab.n, hash(tab.d, hash(:AbstractTableau, h)))))
    return _hash_tableau(tab, _layout_marker(tab), h)
end

function _hash_tableau(tab::AbstractTableau, L::TableauLayout, h::UInt)
    # A canonical tableau is hashed in place. Any other is hashed through a
    # canonical copy, which `_canonical_stab` returns with a different block
    # type, so returning here keeps this path free of allocation.
    tab.iscanonical && return _hash_active(h, _stab(tab, L), _phase(tab), tab.m, tab.storephase)
    S, phase = _canonical_stab(tab, L)
    return _hash_active(h, S, phase, tab.m, tab.storephase)
end
