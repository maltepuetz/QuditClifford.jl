"""
    entanglement_entropy(tab::AbstractTableau, subsystem::AbstractVector; check::Bool=true)

Compute the stabilizer entanglement entropy of a subsystem.

# Arguments
- `tab::AbstractTableau`: Destabilizer or stabilizer tableau describing the
  (possibly mixed) state.
- `subsystem::AbstractVector{<:Integer}`: 1-based qudit indices defining subsystem `A`.
- `check::Bool=true`: verify that `subsystem` holds distinct sites in `1:tab.n`
  and throw an `ArgumentError` otherwise. The check costs `O(n)` and overwrites
  `tab.c_workspace`. With `check=false` it is skipped, and an invalid
  `subsystem` gives a wrong result or reads and writes out of bounds, which can
  crash Julia.

# Returns
- An integer `S(A)` in log-`d` units (so `S=1` means entropy `log(d)`).

# Examples
```jldoctest
julia> tab = DestabilizerTableau(2, 3; state = :ghz);

julia> entanglement_entropy(tab, [1])
1

julia> entanglement_entropy(tab, [1, 2])
1
```

# Notes
- For pure stabilizer states (`m == n`), the function uses `S(A) = S(B)` and computes the rank on the smaller side.
- For mixed stabilizer states (`m < n`), the function traces out the complement and uses the mixed-state stabilizer formula.
"""
function entanglement_entropy(
    tab::AbstractTableau,
    subsystem::T;
    check::Bool=true,
) where T<:AbstractVector

    # The check and the complement rank both write c_workspace while reading
    # `subsystem`, so a `subsystem` stored there is copied first.
    subsystem = Base.unalias(tab.c_workspace, subsystem)
    check && _check_subsystem!(tab, subsystem)
    N_A = length(subsystem)
    n = tab.n
    m = tab.m

    if m == n
        # Pure stabilizer state. Use smaller side since S(A) = S(B).
        if N_A <= n ÷ 2
            rank_A = rank_subsystem_cols!(tab, subsystem, m)
            return rank_A - N_A
        end
        N_B = n - N_A
        rank_B = rank_subsystem_cols!(tab, subsystem, m; complement=true)
        return rank_B - N_B
    end

    # Mixed stabilizer state: trace out the complement B.
    # If R_B is the restriction of generators to B, then
    # m_A = m - rank(R_B), and S(ρ_A) = |A| - m_A (in log_d units).
    rank_B = rank_subsystem_cols!(tab, subsystem, m; complement=true)
    return N_A - m + rank_B
end

"""
    entanglement_profile(tab::AbstractTableau) -> Vector{Int}
    entanglement_profile(tab::AbstractTableau, order::AbstractVector{<:Integer}) -> Vector{Int}

Return the entropies of all prefixes of the qudit chain in log-`d` units.
The result has length `tab.n + 1`: entry `l + 1` is the entropy of sites
`1:l`, including the empty prefix at `l = 0` and the whole state at `l = n`.
It agrees with `[entanglement_entropy(tab, 1:l) for l in 0:tab.n]` for both
pure and mixed stabilizer states. The first entry is zero and the last is
`tab.n - tab.m`.

With a site `order`, a permutation of `1:tab.n`, the prefixes follow that
order instead: entry `l + 1` is the entropy of sites `order[1:l]`, in
agreement with `entanglement_entropy(tab, order[1:l])`. A cyclic shift of
`1:n`, for example, gives every arc that starts at one site of a periodic
chain. The cost is the same as for the natural order.

A single right-endpoint elimination computes the whole profile in
`O(n*m^2 + n)` time (at most cubic in `n`), using the tableau's scratch
`workspace` and allocating only the result vector. Stabilizers, destabilizers,
phases and canonicalization caches are preserved. Phase storage is not required.
Use [`entanglement_profile!`](@ref) to reuse an existing output vector.

For a pure state, each prefix has the same entropy as its complementary
suffix; the prefix profile itself need not be symmetric. For a mixed state,
these are subsystem von Neumann entropies, not an entanglement measure.

# Examples
```jldoctest
julia> entanglement_profile(StabilizerTableau(3, 4; state = :ghz))
5-element Vector{Int64}:
 0
 1
 1
 1
 0

julia> entanglement_profile(StabilizerTableau(3, 3))
4-element Vector{Int64}:
 0
 1
 2
 3

julia> entanglement_profile(StabilizerTableau(3, 4; state = :ghz), [3, 4, 1, 2])
5-element Vector{Int64}:
 0
 1
 1
 1
 0
```
"""
function entanglement_profile(tab::AbstractTableau)
    return entanglement_profile!(Vector{Int}(undef, tab.n + 1), tab)
end

function entanglement_profile(tab::AbstractTableau, order::AbstractVector{<:Integer})
    return entanglement_profile!(Vector{Int}(undef, tab.n + 1), tab, order)
end

"""
    entanglement_profile!(dest::AbstractVector{Int}, tab::AbstractTableau)
    entanglement_profile!(dest::AbstractVector{Int}, tab::AbstractTableau, order::AbstractVector{<:Integer})

Write all prefix entropies into `dest` and return `dest`. The values, units
and the optional site `order` are the same as for
[`entanglement_profile`](@ref). Every entry is overwritten, so the vector can
be reused after the tableau's state or generator count changes.

`dest` must be writable, use one-based indexing, and have exactly `tab.n + 1`
entries. It must not alias any of the tableau's storage. Length and indexing
are checked before writing; the nonaliasing requirement is a caller precondition.
Ordinary `Vector{Int}` buffers and views into independent arrays are supported.

`order` must use one-based indexing and contain every site in `1:tab.n`
exactly once. This is checked before `dest` is written: a wrong length throws
a `DimensionMismatch`, and an out-of-range or repeated site throws an
`ArgumentError`. The check uses `tab.c_workspace` as its scratch mask, so a
rejected `order` leaves `dest`, the generators and the caches unchanged but
not that workspace. An `order` that shares memory with `tab.workspace` or
`tab.c_workspace` is copied before either is written, which allocates.

The method reuses `tab.workspace` (and `tab.c_workspace` to check `order`),
preserves generators and canonicalization caches, and allocates nothing once
compiled for ordinary vectors and views of independent arrays.

# Examples
```jldoctest
julia> tab = StabilizerTableau(3, 4; state = :ghz);

julia> dest = Vector{Int}(undef, tab.n + 1);

julia> entanglement_profile!(dest, tab);

julia> dest
5-element Vector{Int64}:
 0
 1
 1
 1
 0

julia> reset!(tab; state = :mixed);

julia> entanglement_profile!(dest, tab);   # reuses dest

julia> dest
5-element Vector{Int64}:
 0
 1
 2
 3
 4

julia> entanglement_profile!(dest, tab, [4, 3, 2, 1]);

julia> dest
5-element Vector{Int64}:
 0
 1
 2
 3
 4
```
"""
function entanglement_profile!(profile::AbstractVector{Int}, tab::AbstractTableau)
    return _entanglement_profile!(profile, tab, nothing)
end

function entanglement_profile!(
    profile::AbstractVector{Int},
    tab::AbstractTableau,
    order::AbstractVector{<:Integer},
)
    return _entanglement_profile!(profile, tab, order)
end

# Throw unless `order` is a permutation of `1:tab.n`. `c_workspace` serves as
# the visited mask, so the check allocates nothing and writes no output.
function _check_site_order!(tab::AbstractTableau, order::AbstractVector{<:Integer})
    n = tab.n
    Base.require_one_based_indexing(order)
    length(order) == n || throw(DimensionMismatch(
        "The site order must have length $n; got $(length(order))."))
    seen = tab.c_workspace
    fill!(seen, 0)
    @inbounds for k in 1:n
        s = order[k]
        1 <= s <= n || throw(ArgumentError(
            "The site order must contain sites in 1:$n; got $s at position $k."))
        seen[s] == 0 || throw(ArgumentError(
            "The site order must contain each site once; site $s is repeated."))
        seen[s] = 1
    end
    return nothing
end

# Throw unless `subsystem` holds distinct sites in `1:tab.n`. `c_workspace`
# serves as the visited mask, so the check allocates nothing. On return the
# mask holds 1 at the sites of `subsystem` and 0 elsewhere.
function _check_subsystem!(tab::AbstractTableau, subsystem::AbstractVector)
    fill!(tab.c_workspace, 0)
    _mark_subsystem!(tab, subsystem, 1, "subsystem")
    return nothing
end

# Write `label` (nonzero) into `c_workspace` at every site of `subsystem`,
# throwing unless each site is an integer in `1:tab.n` that is not marked yet.
# The caller zeroes the mask once and then marks one or more subsystems, so a
# site already carrying another label belongs to an earlier subsystem.
function _mark_subsystem!(
    tab::AbstractTableau,
    subsystem::AbstractVector,
    label::Int,
    name::String,
)
    n = tab.n
    seen = tab.c_workspace
    @inbounds for s in subsystem
        s isa Integer || throw(ArgumentError(
            "The $name must contain integer sites; got $(repr(s))."))
        1 <= s <= n || throw(ArgumentError(
            "The $name must contain sites in 1:$n; got $s."))
        seen[s] == label && throw(ArgumentError(
            "The $name must contain each site once; site $s is repeated."))
        seen[s] == 0 || throw(ArgumentError(
            "The subsystems must be disjoint; site $s is in more than one."))
        seen[s] = label
    end
    return nothing
end

# `order === nothing` selects the natural order 1:n, the specialization the
# two-argument method compiles to.
function _entanglement_profile!(
    profile::AbstractVector{Int},
    tab::AbstractTableau,
    order::Union{Nothing,AbstractVector{<:Integer}},
)
    n, m, d = tab.n, tab.m, tab.d
    Base.require_one_based_indexing(profile)
    length(profile) == n + 1 || throw(DimensionMismatch(
        "The entropy profile destination must have length $(n + 1); got $(length(profile))."))
    if order !== nothing
        # The check writes c_workspace and the copy below writes workspace
        # while both read `order`, so an `order` stored in either is copied.
        order = Base.unalias(tab.c_workspace, Base.unalias(tab.workspace, order))
        _check_site_order!(tab, order)
    end
    if m == 0
        for l in 0:n
            profile[l + 1] = l
        end
        return profile
    end
    ws = tab.workspace
    stab = tab.stab
    # Site-local row order makes a bottom-up pivot identify the rightmost
    # support of a generator: rows 2k-1 and 2k hold the X and Z exponents of
    # the k-th site in the order. Copy only active columns and omit the phase row.
    if order === nothing
        @turbo for j in 1:m, q in 1:n
            ws[2q - 1, j] = stab[q, j]
            ws[2q, j] = stab[n + q, j]
        end
    else
        @turbo for j in 1:m, k in 1:n
            q = order[k]
            ws[2k - 1, j] = stab[q, j]
            ws[2k, j] = stab[n + q, j]
        end
    end

    # Initially count generators by right endpoint in the result storage.
    # Each reduced generator has a distinct bottommost nonzero (pivot) row.
    # In any nontrivial combination, the largest pivot row involved cannot
    # cancel. Thus a combination is supported on the first l sites of the
    # order exactly when all its participating pivots are at rows <= 2l, so
    # counting those pivots gives the number of independent stabilizers
    # contained in the prefix.
    fill!(profile, 0)
    c = 1
    for r in (2n):-1:1
        c > m && break
        j = c
        while j <= m && ws[r, j] == 0
            j += 1
        end
        j > m && continue

        # Unprocessed columns are zero below r. Only rows 1:r need updating.
        if j != c
            @turbo for i in 1:r
                t = ws[i, c]
                ws[i, c] = ws[i, j]
                ws[i, j] = t
            end
        end
        scale_mod!(view(ws, 1:r, c), tab.inversemod(ws[r, c], d), d)
        for jj in (c + 1):m
            beta = ws[r, jj]
            beta == 0 && continue
            # jj > c, so the source and destination occupy disjoint columns.
            submul_mod!(view(ws, 1:r, jj), view(ws, 1:r, c), beta, d)
        end
        profile[cld(r, 2) + 1] += 1
        c += 1
    end

    contained = 0
    for l in 1:n
        contained += profile[l + 1]
        profile[l + 1] = l - contained
    end
    return profile
end

function rank_subsystem_cols!(
    tab::AbstractTableau,
    subsystem::AbstractVector,
    mcols::Int;
    complement::Bool=false,
)
    stab = tab.stab
    ws = tab.workspace
    n = tab.n

    if complement
        N = n - length(subsystem)
        N == 0 && return 0

        # Use c_workspace as a temporary membership mask (0/1).
        mask = tab.c_workspace
        @turbo for i in 1:n
            mask[i] = 0
        end
        @inbounds for i in eachindex(subsystem)
            mask[subsystem[i]] = 1
        end

        row = 1
        @inbounds for qudit in 1:n
            if mask[qudit] == 0
                @turbo for j in 1:mcols
                    ws[row, j] = stab[qudit, j]
                    ws[row+N, j] = stab[qudit+n, j]
                end
                row += 1
            end
        end
        return rank_fp_cols!(
            view(ws, 1:2*N, 1:mcols),
            tab.d,
            tab.inversemod,
        )
    end

    N = length(subsystem)
    N == 0 && return 0
    @turbo for j in 1:mcols, i in eachindex(subsystem)
        qudit = subsystem[i]
        ws[i, j] = stab[qudit, j]
        ws[i+N, j] = stab[qudit+n, j]
    end

    return rank_fp_cols!(
        view(ws, 1:2*N, 1:mcols),
        tab.d,
        tab.inversemod,
    )
end

# Column-wise Gaussian elimination over F_d; returns the number of pivots.
#
# Because only that count is returned, the elimination runs forwards only. `A`
# is therefore left in column echelon form and **not** in RCEF: pivot rows keep
# whatever entries they had in earlier columns. Do not read a decomposition off
# it. (`_canonicalize_tableau!` does need the reduced form, since
# `coeffs_from_generators!` reads coefficients from pivot rows -- that is a
# separate implementation and this shortcut must not be carried over to it.)
#
# Pivots are taken from rows `1:pivot_rows` only, while the column operations
# act on whole columns. With `piv` pivots found, columns `piv+1:m` are then
# zero on rows `1:pivot_rows`: a row without a pivot was zero in every column
# from the current one on, and only those columns are combined afterwards. As
# the operations are invertible, those columns are a basis of the combinations
# of the input columns that vanish on `1:pivot_rows`.
#
# `{Int}` is the real contract, not a narrowing for convenience: the inner
# loops go through `src/modular.jl`, whose primitives take `AbstractVector{Int}`
# and whose tiers assume Int arithmetic. Declaring it here makes a wrong
# element type a signature mismatch at the call, rather than a MethodError
# raised from inside `scale_mod!`.
function rank_fp_cols!(
    A::AbstractMatrix{Int},
    d::Int,
    inversemod::InverseMod,
    pivot_rows::Int=size(A, 1),
)
    m = size(A, 2)

    r = 1
    c = 1
    piv = 0
    while r <= pivot_rows && c <= m
        # find a pivot in row r
        j = c
        while j <= m && A[r, j] == 0
            j += 1
        end
        if j > m
            r += 1
            continue
        end

        if j != c
            @turbo for i in axes(A, 1)
                t = A[i, c]
                A[i, c] = A[i, j]
                A[i, j] = t
            end
        end

        # scale pivot column so A[r,c] = 1
        α = inversemod(A[r, c], d)
        scale_mod!(view(A, axes(A, 1), c), α, d)

        # Eliminate row r from the columns still to be processed. That is what
        # keeps the pivot count honest; earlier columns are never read again,
        # so reducing them too would only cost time.
        for jj in (c+1):m
            β = A[r, jj]
            β == 0 && continue
            # jj > c, so these are disjoint columns.
            submul_mod!(view(A, axes(A, 1), jj), view(A, axes(A, 1), c), β, d)
        end

        piv += 1
        r += 1
        c += 1
    end
    return piv
end
