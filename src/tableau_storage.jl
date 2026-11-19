# Access to the tableau storage. A tableau holds its stabilizers in the fields
# `stab_x`, `stab_z` (`n × n`, generator `j` in column `j`) and `phase` (length
# `n`, or 0 without `storephase`), and a `DestabilizerTableau` its destabilizers
# in `destab_x` and `destab_z`. Each of the four blocks is a view into a padded
# parent of its own, laid out by `_padded_block` (src/xz_block.jl). Only this
# file and the two constructor files touch those fields; algorithms go through
# `_stab`, `_destab` and `_phase`.

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

# The generator matrix of `tab` in generator-major layout.
function _stabilizer_generator_matrix(tab::AbstractTableau)
    b, phase = _stab(tab), _phase(tab)
    return tab.storephase ? vcat(b.x, b.z, permutedims(phase)) : vcat(b.x, b.z)
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

The result is a copy; changing it does not change `tab`. Any other `layout`
throws an `ArgumentError`.
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

The result is a copy; changing it does not change `tab`. Any other `layout`
throws an `ArgumentError`.
"""
function destabilizer_matrix(tab::DestabilizerTableau; layout::Symbol = :generator)
    _check_layout(layout)
    b = _destab(tab)
    D = vcat(b.x, b.z)
    return layout === :generator ? D : permutedims(D)
end

function _check_generator_matrix_size(src::AbstractMatrix, nrows::Int, ncols::Int)
    size(src) == (nrows, ncols) || throw(ArgumentError(
        "expected a $(nrows)×$(ncols) matrix, got $(size(src, 1))×$(size(src, 2))"))
    return nothing
end

# Overwrite the generators of a tableau with a generator-major matrix of exactly
# the shape `stabilizer_matrix` or `destabilizer_matrix` returns. `m`,
# `iscanonical` and the caches are left alone.
function _load_stabilizer_matrix!(tab::AbstractTableau, S::AbstractMatrix{<:Integer})
    n = tab.n
    _check_generator_matrix_size(S, 2n + (tab.storephase ? 1 : 0), n)
    tab.stab_x .= view(S, 1:n, :)
    tab.stab_z .= view(S, n+1:2n, :)
    tab.storephase && (tab.phase .= view(S, 2n + 1, :))
    return tab
end

function _load_destabilizer_matrix!(tab::DestabilizerTableau, D::AbstractMatrix{<:Integer})
    n = tab.n
    _check_generator_matrix_size(D, 2n, n)
    tab.destab_x .= view(D, 1:n, :)
    tab.destab_z .= view(D, n+1:2n, :)
    return tab
end
