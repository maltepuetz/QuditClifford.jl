"""
    entanglement_negativity(tab::AbstractTableau, A::AbstractVector, B::AbstractVector)
    entanglement_negativity(tab::AbstractTableau, A::AbstractVector)

Return the logarithmic negativity `ℰ(A:B) = log_d ‖ρ_ABᵀᴮ‖₁` between the
disjoint subsystems `A` and `B`, in log-`d` units (so `ℰ = 1` means `log(d)`).
`ρ_AB` is the state of `A ∪ B` with every other qudit traced out, and `ᵀᴮ` is
the partial transpose on `B`. With only `A` given, `B` is its complement.

Unlike the entropy of a mixed state, the negativity vanishes on classical
correlations: it measures the entanglement between `A` and `B` alone. For a
pure state and `B` the complement of `A` it equals
[`entanglement_entropy`](@ref)`(tab, A)`. It is symmetric in `A` and `B`, and
it is zero when either is empty.

# Arguments
- `tab::AbstractTableau`: Destabilizer or stabilizer tableau describing the
  (possibly mixed) state. Phase storage is not required.
- `A`, `B`: 1-based qudit indices. Each must contain distinct sites in
  `1:tab.n`, and no site may be in both; otherwise an `ArgumentError` is
  thrown. The check is always performed and costs `O(n)`.

# Returns
- An `Int`. The stabilizer state `ρ_AB` has a negativity spectrum that is flat,
  so `ℰ(A:B)` is always a whole number of `log(d)` units.

# Method
The generators are reduced to a basis of the stabilizer group of `ρ_AB` (the
elements supported on `A ∪ B`), and `ℰ(A:B)` is half the rank over `F_d` of
the matrix of their pairwise symplectic products restricted to `A`. Since the
elements commute, that matrix is minus the one restricted to `B`, so the
smaller side is used. This takes at most `O(n*m^2)` time,
cubic in `n`, reads only the X and Z rows, and leaves the generators, phases,
destabilizers and canonicalization caches unchanged. It overwrites the
scratch arrays `tab.workspace`, `tab.c_workspace` and `tab.zacc_workspace`,
and allocates nothing unless `A` or `B` shares memory with
`tab.c_workspace`, in which case it is copied first.

# References
- S. Sang, Y. Li, T. Zhou, X. Chen, T. H. Hsieh and M. P. A. Fisher,
  *Entanglement Negativity at Measurement-Induced Criticality*,
  PRX Quantum **2**, 030313 (2021), arXiv:2012.00031. Derives
  `ℰ = rank(K)/2` for qubit stabilizer states, with `K` the anticommutation
  matrix over `F_2`.
- D. Fattal, T. S. Cubitt, Y. Yamamoto, S. Bravyi and I. L. Chuang,
  *Entanglement in the stabilizer formalism*, arXiv:quant-ph/0406168 (2004).
  The local reduction of bipartite stabilizer states to EPR pairs, which
  underlies the formula.

The qudit form used here, with the symplectic product `x_i·z_j - z_i·x_j`
over `F_d`, is the same construction for prime `d`. The package tests check
it against a dense partial transpose for `d = 2, 3, 5, 7`.

# Examples
```jldoctest
julia> bell = StabilizerTableau(3, 2; state = :ghz);

julia> entanglement_negativity(bell, [1], [2])
1

julia> ghz = StabilizerTableau(3, 3; state = :ghz);

julia> entanglement_negativity(ghz, [1], [2])   # tracing out qudit 3 leaves only classical correlations
0

julia> entanglement_negativity(ghz, [1]) == entanglement_entropy(ghz, [1]) == 1
true
```
"""
function entanglement_negativity(
    tab::AbstractTableau,
    A::AbstractVector,
    B::AbstractVector,
)
    # Marking writes c_workspace while reading A and B, so a subsystem stored
    # there is copied first. Neither is read after marking.
    A = Base.unalias(tab.c_workspace, A)
    B = Base.unalias(tab.c_workspace, B)
    fill!(tab.c_workspace, 0)
    _mark_subsystem!(tab, A, 1, "subsystem A")
    _mark_subsystem!(tab, B, 2, "subsystem B")
    return _negativity_from_mask!(tab, length(A), length(B))
end

function entanglement_negativity(tab::AbstractTableau, A::AbstractVector)
    A = Base.unalias(tab.c_workspace, A)
    _mark_with_complement!(tab, A)
    return _negativity_from_mask!(tab, length(A), tab.n - length(A))
end

# Validate `A` and label it 1 in `c_workspace`, and every other site 2.
function _mark_with_complement!(tab::AbstractTableau, A::AbstractVector)
    _check_subsystem!(tab, A)
    mask = tab.c_workspace
    @inbounds for q in 1:tab.n
        mask[q] = mask[q] == 1 ? 1 : 2
    end
    return nothing
end

# Group the sites by their label in `c_workspace`, given `counts[l + 1]`
# sites with each label `l` in 0:3: write them to `tab.zacc_workspace`, label
# by label and increasing within a label, and return the index at which each
# label's block starts. Copying rows through this list goes column by column,
# like the copy in `rank_subsystem_cols!`, instead of one strided row at a time.
function _group_sites_by_label!(tab::AbstractTableau, counts::NTuple{4,Int})
    mask = tab.c_workspace
    sites = tab.zacc_workspace
    first = (1, 1 + counts[1], 1 + counts[1] + counts[2],
             1 + counts[1] + counts[2] + counts[3])
    p0, p1, p2, p3 = first
    @inbounds for q in 1:tab.n
        label = mask[q]
        if label == 0
            sites[p0] = q
            p0 += 1
        elseif label == 1
            sites[p1] = q
            p1 += 1
        elseif label == 2
            sites[p2] = q
            p2 += 1
        else
            sites[p3] = q
            p3 += 1
        end
    end
    return first
end

# Copy the X and Z rows of the `N` sites listed in `tab.zacc_workspace` from
# index `first` on into `tab.workspace` after row `offset`: X exponents to rows
# offset+1:offset+N and Z exponents to offset+N+1:offset+2N, columns 1:m.
# Returns the last row written, offset + 2N.
function _copy_site_rows!(tab::AbstractTableau, first::Int, N::Int, offset::Int)
    n, m = tab.n, tab.m
    stab = tab.stab
    ws = tab.workspace
    sites = tab.zacc_workspace
    @turbo for j in 1:m, r in 1:N
        q = sites[first + r - 1]
        ws[offset + r, j] = stab[q, j]
        ws[offset + N + r, j] = stab[n + q, j]
    end
    return offset + 2N
end

# `c_workspace[q]` labels site `q`: 1 for A, 2 for B and 0 for a traced-out
# site. `NA` and `NB` are the sizes of A and B.
function _negativity_from_mask!(tab::AbstractTableau, NA::Int, NB::Int)
    n, m, d = tab.n, tab.m, tab.d
    (NA == 0 || NB == 0 || m == 0) && return 0
    ws = tab.workspace

    # ℰ(A:B) = ℰ(B:A), so the generators are restricted to the smaller side S.
    side = NA <= NB ? 1 : 2
    NS = min(NA, NB)
    NC = n - NA - NB

    # Rows 1:2NC hold the X and Z exponents on the traced-out sites, rows
    # R0+1:R0+NS and R0+NS+1:R starting at R0 = 2NC those on S.
    first = _group_sites_by_label!(tab, (NC, NA, NB, 0))
    R0 = _copy_site_rows!(tab, first[1], NC, 0)
    R = _copy_site_rows!(tab, first[side + 1], NS, R0)

    # Pivoting on the traced-out rows alone leaves columns piv+1:m zero there,
    # and those columns are a basis of the stabilizer group of ρ_AB.
    piv = rank_fp_cols!(view(ws, 1:R, 1:m), d, tab.inversemod, R0)
    k = m - piv
    k <= 1 && return 0

    # K[i, j] is the symplectic product on S of basis elements i and j. It is
    # alternating, so its rank is even. K occupies rows R+1:R+k and columns
    # 1:k, disjoint from the rows 1:R it is computed from. It fits: ρ_AB has
    # at most NA + NB independent stabilizers, so k <= NA + NB, and
    # 2NS <= NA + NB, so R + k <= 2NC + 2(NA + NB) = 2n.
    xS = R0
    zS = R0 + NS
    @inbounds for j in 1:k
        cj = piv + j
        ws[R + j, j] = 0
        for i in 1:(j - 1)
            ci = piv + i
            xz = 0
            zx = 0
            # Each sum has NS <= n terms below (d-1)^2, inside the package's
            # arithmetic envelope (see `max_safe_dimension`).
            @turbo for t in 1:NS
                xz += ws[xS + t, ci] * ws[zS + t, cj]
                zx += ws[zS + t, ci] * ws[xS + t, cj]
            end
            v = rem(xz, d) - rem(zx, d)
            v < 0 && (v += d)
            ws[R + i, j] = v
            ws[R + j, i] = v == 0 ? 0 : d - v
        end
    end

    return rank_fp_cols!(view(ws, (R + 1):(R + k), 1:k), d, tab.inversemod) ÷ 2
end
