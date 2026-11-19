# Access to the tableau storage: the generators of a tableau as plain `Int`
# matrices, in generator-major or qudit-major layout.

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
    return layout === :generator ? copy(tab.stab) : permutedims(tab.stab)
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
    return layout === :generator ? copy(tab.destab) : permutedims(tab.destab)
end

# Overwrite `dst` with the generator-major matrix `src` of exactly the same shape.
# `m`, `iscanonical` and the caches are left alone.
function _load_generator_matrix!(dst::Matrix{Int}, src::AbstractMatrix{<:Integer})
    size(src) == size(dst) || throw(ArgumentError(
        "expected a $(size(dst, 1))×$(size(dst, 2)) matrix, got $(size(src, 1))×$(size(src, 2))"))
    copyto!(dst, src)
    return dst
end

function _load_stabilizer_matrix!(tab::AbstractTableau, S::AbstractMatrix{<:Integer})
    _load_generator_matrix!(tab.stab, S)
    return tab
end

function _load_destabilizer_matrix!(tab::DestabilizerTableau, D::AbstractMatrix{<:Integer})
    _load_generator_matrix!(tab.destab, D)
    return tab
end

# Block and phase views of a tableau's storage.
_stab(tab::AbstractTableau) = _stacked_views(tab.stab, tab.n, tab.storephase)[1]
_destab(tab::DestabilizerTableau) = _stacked_views(tab.destab, tab.n, false)[1]
_phase(tab::AbstractTableau) = _stacked_views(tab.stab, tab.n, tab.storephase)[2]
