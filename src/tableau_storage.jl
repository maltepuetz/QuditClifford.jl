# Access to the tableau storage. A tableau holds its stabilizers in the fields
# `stab_x`, `stab_z` (`n × n`, generator `j` in column `j`) and `phase` (length
# `n`, or 0 without `storephase`), and a `DestabilizerTableau` its destabilizers
# in `destab_x` and `destab_z`. Each of the four blocks is a view into a padded
# parent of its own, laid out by `_padded_block` (src/xz_block.jl). Only this
# file and the two constructor files touch those fields; algorithms go through
# `_stab`, `_destab` and `_phase`.
#
# The four blocks share the element type `E`, the second type parameter of both
# tableau types; the phase vector is always `Int`.

# The storage element type of a tableau's X and Z blocks.
_storage_eltype(::StabilizerTableau{IM, E}) where {IM, E} = E
_storage_eltype(::DestabilizerTableau{IM, E}) where {IM, E} = E

# Whether `E` can store a tableau at dimension `d`: `Int` always, `UInt8` and
# `UInt16` when they hold every residue mod `d` and mod `phase_modulus(d)`, any
# other type never.
_eltype_admissible(::Type, ::Int) = false
_eltype_admissible(::Type{Int}, ::Int) = true
_eltype_admissible(::Type{E}, d::Int) where {E<:_NarrowEltype} =
    d - 1 <= typemax(E) && phase_modulus(d) - 1 <= typemax(E)

# The narrowest admissible storage element type for dimension `d`.
function _default_eltype(d::Int)
    _eltype_admissible(UInt8, d) && return UInt8
    _eltype_admissible(UInt16, d) && return UInt16
    return Int
end

function _check_eltype(E::Type, d::Int)
    _eltype_admissible(E, d) && return nothing
    ok = join(filter(T -> _eltype_admissible(T, d), (UInt8, UInt16, Int)), ", ")
    throw(ArgumentError("eltype = $E cannot hold entries mod d = $d; use one of $ok."))
end

# The stabilizer and destabilizer blocks and the phase vector of a tableau. They
# share the tableau's arrays, so writing through them changes the tableau.
_stab(tab::AbstractTableau) = XZBlock{GeneratorMajor}(tab.stab_x, tab.stab_z)
_destab(tab::DestabilizerTableau) = XZBlock{GeneratorMajor}(tab.destab_x, tab.destab_z)
_phase(tab::AbstractTableau) = tab.phase

@noinline _throw_removed_field(s::Symbol, accessor::Symbol) =
    throw(ArgumentError("tab.$s was removed; use $accessor(tab)."))

@inline function Base.getproperty(tab::StabilizerTableau, s::Symbol)
    s === :stab && _throw_removed_field(s, :stabilizer_matrix)
    return getfield(tab, s)
end

@inline function Base.getproperty(tab::DestabilizerTableau, s::Symbol)
    s === :stab && _throw_removed_field(s, :stabilizer_matrix)
    s === :destab && _throw_removed_field(s, :destabilizer_matrix)
    return getfield(tab, s)
end

# The blocks of `b` stacked as a fresh `Int` matrix, X rows first, with `extra`
# further rows left for the caller to fill.
function _stacked_int_matrix(b::_GMBlock, extra::Int)
    n = _nq(b)
    S = Matrix{Int}(undef, 2n + extra, _ncap(b))
    S[1:n, :] .= b.x
    S[n+1:2n, :] .= b.z
    return S
end

# The generator matrix of `tab` in generator-major layout.
function _stabilizer_generator_matrix(tab::AbstractTableau)
    S = _stacked_int_matrix(_stab(tab), tab.storephase ? 1 : 0)
    tab.storephase && (S[2tab.n+1, :] .= _phase(tab))
    return S
end

function _check_layout(layout::Symbol)
    layout === :generator || layout === :qudit ||
        throw(ArgumentError("layout must be :generator or :qudit, got :$layout"))
    return nothing
end

"""
    stabilizer_matrix(tab::AbstractTableau; layout::Symbol=:generator) -> Matrix{Int}

Return the stabilizer generators of `tab` as a fresh `Int` matrix.

```julia
tab = StabilizerTableau(3, 2; state = :product, basis = :Z)
S = stabilizer_matrix(tab)                    # 5×2, generators in columns
Sq = stabilizer_matrix(tab; layout = :qudit)  # 2×5, generators in rows
```

With `layout = :generator` (the default) the matrix is `(2n + storephase) × n`:
generator `j` is column `j`, rows `1:n` hold the X exponents, rows `n+1:2n` the Z
exponents, and row `2n+1` the phase exponent (only when `storephase` is `true`).
All `n` columns are returned, including the zeroed capacity columns `m+1:n`.
With `layout = :qudit` the result is the `permutedims` of that matrix, with
generator `j` in row `j`.

The result is a copy; changing it does not change `tab`. Any other symbol for
`layout` throws an `ArgumentError`.
"""
function stabilizer_matrix(tab::AbstractTableau; layout::Symbol = :generator)
    _check_layout(layout)
    S = _stabilizer_generator_matrix(tab)
    return layout === :generator ? S : permutedims(S)
end

"""
    destabilizer_matrix(tab::DestabilizerTableau; layout::Symbol=:generator) -> Matrix{Int}

Return the destabilizer generators of `tab` as a fresh `Int` matrix.

```julia
tab = DestabilizerTableau(3, 2; state = :product, basis = :Z)
D = destabilizer_matrix(tab)                    # 4×2, generators in columns
Dq = destabilizer_matrix(tab; layout = :qudit)  # 2×4, generators in rows
```

With `layout = :generator` (the default) the matrix is `2n × n`: destabilizer
`j` is column `j`, rows `1:n` hold the X exponents and rows `n+1:2n` the Z
exponents. Destabilizers carry no phase row. With `layout = :qudit` the result
is the `permutedims` of that matrix.

The result is a copy; changing it does not change `tab`. Any other symbol for
`layout` throws an `ArgumentError`.
"""
function destabilizer_matrix(tab::DestabilizerTableau; layout::Symbol = :generator)
    _check_layout(layout)
    D = _stacked_int_matrix(_destab(tab), 0)
    return layout === :generator ? D : permutedims(D)
end

function _check_generator_matrix_size(src::AbstractMatrix, nrows::Int, ncols::Int)
    size(src) == (nrows, ncols) || throw(ArgumentError(
        "expected a $(nrows)×$(ncols) matrix, got $(size(src, 1))×$(size(src, 2))"))
    return nothing
end

# Entry `s` reduced mod `d` in `Int` and stored as `E`, which holds every
# residue mod `d` of an admissible `d`.
@inline _reduced_entry(::Type{E}, s::Integer, d::Int) where {E} = mod(Int(s), d) % E

# Overwrite the generators of a tableau with a generator-major matrix of exactly
# the shape `stabilizer_matrix` or `destabilizer_matrix` returns, reducing its
# entries mod `d` and its phases mod `phase_modulus(d)`. `m`, `iscanonical` and
# the caches are left alone.
function _load_stabilizer_matrix!(tab::AbstractTableau, S::AbstractMatrix{<:Integer})
    n, d, E = tab.n, tab.d, _storage_eltype(tab)
    _check_generator_matrix_size(S, 2n + (tab.storephase ? 1 : 0), n)
    tab.stab_x .= _reduced_entry.(E, view(S, 1:n, :), d)
    tab.stab_z .= _reduced_entry.(E, view(S, n+1:2n, :), d)
    tab.storephase && (tab.phase .= _reduced_entry.(Int, view(S, 2n + 1, :), phase_modulus(d)))
    return tab
end

function _load_destabilizer_matrix!(tab::DestabilizerTableau, D::AbstractMatrix{<:Integer})
    n, d, E = tab.n, tab.d, _storage_eltype(tab)
    _check_generator_matrix_size(D, 2n, n)
    tab.destab_x .= _reduced_entry.(E, view(D, 1:n, :), d)
    tab.destab_z .= _reduced_entry.(E, view(D, n+1:2n, :), d)
    return tab
end
