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

    # The check and the complement rank both write c_workspace, and the rank
    # writes a rank workspace, while reading `subsystem`, so a `subsystem`
    # stored in any of them is copied first.
    subsystem = _unalias_rank_workspaces(tab, Base.unalias(tab.c_workspace, subsystem))
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
`O(n*m^2 + n)` time (at most cubic in `n`), using one of the tableau's scratch
workspaces and allocating only the result vector. The elimination runs on
bit-packed words for `d = 2` and on one byte per entry for other `d ≤ 255`.
Stabilizers, destabilizers, phases and canonicalization caches are preserved.
Phase storage is not required. Use [`entanglement_profile!`](@ref) to reuse an
existing output vector.

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
not that workspace. An `order` that shares memory with `tab.c_workspace` or
a rank workspace is copied before either is written, which allocates.

The method reuses the rank workspace for `d` (`tab.bit_workspace` for `d = 2`,
`tab.narrow_workspace` for other `d ≤ 255`, else `tab.workspace`), and
`tab.c_workspace` to check `order`. It preserves generators and
canonicalization caches, and allocates nothing once compiled for ordinary
vectors and views of independent arrays.

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
        # The check writes c_workspace and the copy below writes a rank
        # workspace while both read `order`, so an `order` stored in any of
        # them is copied.
        order = _unalias_rank_workspaces(tab, Base.unalias(tab.c_workspace, order))
        _check_site_order!(tab, order)
    end
    if m == 0
        for l in 0:n
            profile[l + 1] = l
        end
        return profile
    end
    # Each reduced generator has a distinct bottommost nonzero (pivot) row.
    # In any nontrivial combination, the largest pivot row involved cannot
    # cancel. Thus a combination is supported on the first l sites of the
    # order exactly when all its participating pivots are at rows <= 2l, so
    # counting those pivots gives the number of independent stabilizers
    # contained in the prefix. The counts go into the result storage first.
    sites = order === nothing ? Base.OneTo(n) : order
    if d == 2
        _profile_pivots_gf2!(profile, tab, sites)
    elseif _uses_narrow_rank(d)
        _profile_pivots!(profile, tab, tab.narrow_workspace, sites)
    else
        _profile_pivots!(profile, tab, tab.workspace, sites)
    end

    contained = 0
    for l in 1:n
        contained += profile[l + 1]
        profile[l + 1] = l - contained
    end
    return profile
end

# Copy rows interleaved by site: rows 2k-1 and 2k hold the X and Z exponents
# of `sites[k]`, so that a bottom-up pivot identifies the rightmost support of
# a generator. Only the active columns are copied, without the phase row.
function _copy_sites_interleaved!(ws::AbstractMatrix{Int}, b::_GMBlock,
                                  m::Int, sites::AbstractVector)
    n = _nq(b)
    X, Z = b.x, b.z
    @turbo for j in 1:m, k in 1:n
        q = sites[k]
        ws[2k - 1, j] = X[q, j]
        ws[2k, j] = Z[q, j]
    end
    return ws
end

function _copy_sites_interleaved!(ws::AbstractMatrix{UInt8}, b::_GMBlock,
                                  m::Int, sites::AbstractVector)
    n = _nq(b)
    X, Z = b.x, b.z
    @inbounds for j in 1:m
        @simd for k in 1:n
            q = sites[k]
            ws[2k - 1, j] = X[q, j] % UInt8
            ws[2k, j] = Z[q, j] % UInt8
        end
    end
    return ws
end

# Bottom-up column elimination on the interleaved rows; `profile[l + 1]`
# receives the number of pivots on the rows of site l. `ws` is the Int or the
# UInt8 workspace.
function _profile_pivots!(
    profile::AbstractVector{Int},
    tab::AbstractTableau,
    ws::AbstractMatrix,
    sites::AbstractVector,
)
    n, m, d = tab.n, tab.m, tab.d
    _copy_sites_interleaved!(ws, _stab(tab), m, sites)
    fill!(profile, 0)
    c = 1
    @inbounds for r in (2n):-1:1
        c > m && break
        j = c
        while j <= m && iszero(ws[r, j])
            j += 1
        end
        j > m && continue

        # Unprocessed columns are zero below r. Only rows 1:r need updating.
        rows = 1:r
        j != c && _swap_cols!(ws, rows, c, j)
        # Eliminating with multiples of the unscaled pivot column changes the
        # other columns exactly as normalizing it first would.
        alpha = tab.inversemod(Int(ws[r, c]), d)
        src = view(ws, rows, c)
        for jj in (c + 1):m
            beta = Int(ws[r, jj])
            beta == 0 && continue
            # jj > c, so the source and destination occupy disjoint columns.
            _submul_col!(view(ws, rows, jj), src, mul_mod(beta, alpha, d), d)
        end
        profile[cld(r, 2) + 1] += 1
        c += 1
    end
    return profile
end

# `_profile_pivots!` for d = 2 on bit-packed columns, where row r is bit r - 1
# of the column. Bits above r in the word holding r are rows already
# processed, so they are zero in the remaining columns and the update can run
# over whole words 1:w.
function _profile_pivots_gf2!(
    profile::AbstractVector{Int},
    tab::AbstractTableau,
    sites::AbstractVector,
)
    n, m = tab.n, tab.m
    W = tab.bit_workspace
    _pack_sites_gf2!(W, _stab(tab), m, sites, n)
    fill!(profile, 0)
    c = 1
    @inbounds for r in (2n):-1:1
        c > m && break
        w = (r - 1) >> 6 + 1
        bit = one(UInt64) << ((r - 1) & 63)
        j = c
        while j <= m && iszero(W[w, j] & bit)
            j += 1
        end
        j > m && continue
        j != c && _swap_cols!(W, 1:w, c, j)
        for jj in (c + 1):m
            iszero(W[w, jj] & bit) && continue
            @simd for l in 1:w
                W[l, jj] = xor(W[l, jj], W[l, c])
            end
        end
        profile[cld(r, 2) + 1] += 1
        c += 1
    end
    return profile
end

# Rank of the generator columns 1:mcols restricted to the sites of
# `subsystem`, or to the sites outside it with `complement=true`. The rows go
# into the workspace that `d` selects (see src/rank_workspaces.jl), and the
# tableau itself is not changed.
function rank_subsystem_cols!(
    tab::AbstractTableau,
    subsystem::AbstractVector,
    mcols::Int;
    complement::Bool=false,
)
    n = tab.n
    if complement
        N = n - length(subsystem)
        N == 0 && return 0
        # Mark the subsystem in c_workspace, then compact the unmarked sites
        # into its front in increasing order. Entry k is written only after
        # entries 1:k have been read, so the compaction can run in place.
        mask = tab.c_workspace
        @turbo for i in 1:n
            mask[i] = 0
        end
        @inbounds for i in eachindex(subsystem)
            mask[subsystem[i]] = 1
        end
        k = 0
        @inbounds for q in 1:n
            if mask[q] == 0
                k += 1
                mask[k] = q
            end
        end
        return _rank_on_sites!(tab, view(mask, 1:N), N, mcols)
    end
    N = length(subsystem)
    N == 0 && return 0
    return _rank_on_sites!(tab, subsystem, N, mcols)
end

# Rank of the generator columns 1:mcols restricted to `sites[1:N]`.
function _rank_on_sites!(tab::AbstractTableau, sites::AbstractVector, N::Int, mcols::Int)
    d = tab.d
    mcols == 0 && return 0
    if d == 2
        W = tab.bit_workspace
        nw = _pack_sites_gf2!(W, _stab(tab), mcols, sites, N)
        return _rank_gf2_cols!(view(W, 1:nw, 1:mcols), 2N)
    elseif _uses_narrow_rank(d)
        ws = tab.narrow_workspace
        _copy_sites!(ws, _stab(tab), mcols, sites, N, 0)
        return rank_fp_cols!(view(ws, 1:(2N), 1:mcols), d, tab.inversemod)
    else
        ws = tab.workspace
        _copy_sites!(ws, _stab(tab), mcols, sites, N, 0)
        return rank_fp_cols!(view(ws, 1:(2N), 1:mcols), d, tab.inversemod)
    end
end

# Column-wise Gaussian elimination over F_d; returns the number of pivots.
#
# Because only that count is returned, the elimination runs forwards only. `A`
# is therefore left in column echelon form and **not** in RCEF: pivot rows keep
# whatever entries they had in earlier columns, and the pivot columns are not
# normalized. Do not read a decomposition off it. (`_canonicalize_tableau!`
# does need the reduced form, since `coeffs_from_generators!` reads
# coefficients from pivot rows -- that is a separate implementation and this
# shortcut must not be carried over to it.)
#
# Pivots are taken from rows `1:pivot_rows` only, while the column operations
# act on whole columns. With `piv` pivots found, columns `piv+1:m` are then
# zero on rows `1:pivot_rows`: a row without a pivot was zero in every column
# from the current one on, and only those columns are combined afterwards. As
# the operations are invertible, those columns are a basis of the combinations
# of the input columns that vanish on `1:pivot_rows`. Their entries do not
# depend on whether a pivot column is normalized first: subtracting
# `beta * inv(p)` times the pivot column is the same update.
#
# The same invariant means that, at pivot row `r`, every column from the
# current one on is already zero on rows `1:r-1`, so the swap and the updates
# run over rows `r:end` only.
#
# The element type is the contract: `Int` entries go through the primitives in
# src/modular.jl, and `UInt8` entries (for `d <= 255`) through `_submul_u8!`.
# Any other element type is a MethodError at the call.
function rank_fp_cols!(
    A::AbstractMatrix{T},
    d::Int,
    inversemod::InverseMod,
    pivot_rows::Int=size(A, 1),
) where {T<:Union{Int,UInt8}}
    T === UInt8 && !_uses_narrow_rank(d) && throw(ArgumentError(
        "UInt8 elimination needs d <= $_NARROW_RANK_MAX_D; got d = $d."))
    nr, m = size(A)
    0 <= pivot_rows <= nr || throw(ArgumentError(
        "pivot_rows must lie in 0:$nr; got $pivot_rows."))
    Base.require_one_based_indexing(A)

    r = 1
    c = 1
    piv = 0
    @inbounds while r <= pivot_rows && c <= m
        # find a pivot in row r
        j = c
        while j <= m && iszero(A[r, j])
            j += 1
        end
        if j > m
            r += 1
            continue
        end

        rows = r:nr
        j != c && _swap_cols!(A, rows, c, j)

        # Eliminate row r from the columns still to be processed. That is what
        # keeps the pivot count honest; earlier columns are never read again,
        # so reducing them too would only cost time.
        alpha = inversemod(Int(A[r, c]), d)
        src = view(A, rows, c)
        for jj in (c+1):m
            beta = Int(A[r, jj])
            beta == 0 && continue
            # jj > c, so these are disjoint columns.
            _submul_col!(view(A, rows, jj), src, mul_mod(beta, alpha, d), d)
        end

        piv += 1
        r += 1
        c += 1
    end
    return piv
end
