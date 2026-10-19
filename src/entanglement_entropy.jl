"""
    entanglement_entropy(tab::AbstractTableau, subsystem::AbstractVector)

Compute the stabilizer entanglement entropy of a subsystem.

# Arguments
- `tab::AbstractTableau`: Destabilizer or stabilizer tableau describing the
  (possibly mixed) state.
- `subsystem::AbstractVector{<:Integer}`: 1-based qudit indices defining subsystem `A`.

# Returns
- An integer `S(A)` in log-`d` units (so `S=1` means entropy `log(d)`).

# Examples
```julia
tab = DestabilizerTableau(2, 3; state=:ghz)
S1 = entanglement_entropy(tab, [1])      # 1
S2 = entanglement_entropy(tab, [1, 2])   # 1
```

# Notes
- For pure stabilizer states (`m == n`), the function uses `S(A) = S(B)` and computes the rank on the smaller side.
- For mixed stabilizer states (`m < n`), the function traces out the complement and uses the mixed-state stabilizer formula.
- `subsystem` should contain distinct indices; repeated indices lead to undefined results.
"""
function entanglement_entropy(
    tab::AbstractTableau,
    subsystem::T
) where T<:AbstractVector

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

Return the entropies of all prefixes of the qudit chain in log-`d` units.
The result has length `tab.n + 1`: entry `l + 1` is the entropy of sites
`1:l`, including the empty prefix at `l = 0` and the whole state at `l = n`.
It agrees with `[entanglement_entropy(tab, 1:l) for l in 0:tab.n]` for both
pure and mixed stabilizer states. The first entry is zero and the last is
`tab.n - tab.m`.

A single right-endpoint elimination computes the whole profile in
`O(n*m^2 + n)` time (at most cubic in `n`), using the tableau's scratch
`workspace` and allocating only the result vector. Stabilizers, destabilizers,
phases and canonicalization caches are preserved. Phase storage is not required.
Use [`entanglement_profile!`](@ref) to reuse an existing output vector.

For a pure state, each prefix has the same entropy as its complementary
suffix; the prefix profile itself need not be symmetric. For a mixed state,
these are subsystem von Neumann entropies, not an entanglement measure.

# Examples
```julia
entanglement_profile(StabilizerTableau(3, 4; state=:ghz)) # [0, 1, 1, 1, 0]
entanglement_profile(StabilizerTableau(3, 3))             # [0, 1, 2, 3]
```
"""
function entanglement_profile(tab::AbstractTableau)
    return entanglement_profile!(Vector{Int}(undef, tab.n + 1), tab)
end

"""
    entanglement_profile!(dest::AbstractVector{Int}, tab::AbstractTableau)

Write all prefix entropies into `dest` and return `dest`. The values and units
are the same as [`entanglement_profile`](@ref). Every entry is overwritten,
so the vector can be reused after the tableau's state or generator count changes.

`dest` must be writable, use one-based indexing, and have exactly `tab.n + 1`
entries. It must not alias any of the tableau's storage. Length and indexing
are checked before writing; the nonaliasing requirement is a caller precondition.
Ordinary `Vector{Int}` buffers and views into independent arrays are supported.

The method reuses `tab.workspace`, preserves generators and canonicalization
caches, and allocates nothing once compiled for ordinary vectors and views.

# Examples
```julia
tab = StabilizerTableau(3, 4; state=:ghz)
dest = Vector{Int}(undef, tab.n + 1)
entanglement_profile!(dest, tab) # dest is now [0, 1, 1, 1, 0]
reset!(tab; state=:mixed)
entanglement_profile!(dest, tab) # the same dest is now [0, 1, 2, 3, 4]
```
"""
function entanglement_profile!(profile::AbstractVector{Int}, tab::AbstractTableau)
    n, m, d = tab.n, tab.m, tab.d
    Base.require_one_based_indexing(profile)
    length(profile) == n + 1 || throw(DimensionMismatch(
        "The entropy profile destination must have length $(n + 1); got $(length(profile))."))
    if m == 0
        for l in 0:n
            profile[l + 1] = l
        end
        return profile
    end
    ws = tab.workspace
    stab = tab.stab
    # Site-local row order makes a bottom-up pivot identify the rightmost
    # support of a generator. Copy only active columns and omit the phase row.
    @turbo for j in 1:m, q in 1:n
        ws[2q - 1, j] = stab[q, j]
        ws[2q, j] = stab[n + q, j]
    end

    # Initially count generators by right endpoint in the result storage.
    # Each reduced generator has a distinct bottommost nonzero (pivot) row.
    # In any nontrivial combination, the largest pivot row involved cannot
    # cancel. Thus a combination is supported in 1:l exactly when all its
    # participating pivots are at rows <= 2l, so counting those pivots gives
    # the number of independent stabilizers contained in the prefix.
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
# `{Int}` is the real contract, not a narrowing for convenience: the inner
# loops go through `src/modular.jl`, whose primitives take `AbstractVector{Int}`
# and whose tiers assume Int arithmetic. Declaring it here makes a wrong
# element type a signature mismatch at the call, rather than a MethodError
# raised from inside `scale_mod!`.
function rank_fp_cols!(A::AbstractMatrix{Int}, d::Int, inversemod::InverseMod)
    n, m = size(A)

    r = 1
    c = 1
    piv = 0
    while r <= n && c <= m
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
