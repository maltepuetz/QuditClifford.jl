"""
    mutual_information(tab::AbstractTableau, A::AbstractVector, B::AbstractVector)
    mutual_information(tab::AbstractTableau, A::AbstractVector)

Return the mutual information `I(A:B) = S(A) + S(B) - S(AB)` between the
disjoint subsystems `A` and `B`, in log-`d` units (so `I = 1` means
`log(d)`), with every other qudit traced out. With only `A` given, `B` is its
complement, and for a pure state `I(A:B) = 2 S(A)`.

The result equals the sum of three [`entanglement_entropy`](@ref) calls. For a
mixed state it shares their elimination and is cheaper; for a pure state it
costs the same. Mutual information counts
classical correlations as well as entanglement; for entanglement alone use
[`entanglement_negativity`](@ref).

# Arguments
- `tab::AbstractTableau`: Destabilizer or stabilizer tableau describing the
  (possibly mixed) state. Phase storage is not required.
- `A`, `B`: 1-based qudit indices. Each must contain distinct sites in
  `1:tab.n`, and no site may be in both; otherwise an `ArgumentError` is
  thrown. The check is always performed and costs `O(n)`.

# Returns
- A nonnegative `Int`. It is zero when either subsystem is empty.

# Method
The generators are reduced on the traced-out qudits to a basis of the `k`
stabilizers supported on `A ∪ B`, and `I(A:B) = rank(K_A) + rank(K_B) - k`,
where `K_X` is that basis restricted to `X`. For a pure state each of the
three entropies is instead computed on the smaller side of its cut, as
[`entanglement_entropy`](@ref) does. This takes at most `O(n*m^2)`
time, cubic in `n`, reads only the X and Z rows, and leaves the generators,
phases, destabilizers and canonicalization caches unchanged. It overwrites the
scratch arrays `tab.workspace`, `tab.c_workspace` and `tab.zacc_workspace`,
and allocates nothing unless `A` or `B` shares memory with
`tab.c_workspace`, in which case it is copied first.

# Examples
```julia
ghz = StabilizerTableau(3, 3; state=:ghz)
mutual_information(ghz, [1], [2])     # 1: classical correlation
entanglement_negativity(ghz, [1], [2]) # 0
mutual_information(ghz, [1])          # 2, twice entanglement_entropy(ghz, [1])
```
"""
function mutual_information(tab::AbstractTableau, A::AbstractVector, B::AbstractVector)
    # Marking writes c_workspace while reading A and B, so a subsystem stored
    # there is copied first. Neither is read after marking.
    A = Base.unalias(tab.c_workspace, A)
    B = Base.unalias(tab.c_workspace, B)
    fill!(tab.c_workspace, 0)
    _mark_subsystem!(tab, A, 1, "subsystem A")
    _mark_subsystem!(tab, B, 2, "subsystem B")
    return _mutual_information_from_mask!(tab, length(A), length(B))
end

function mutual_information(tab::AbstractTableau, A::AbstractVector)
    A = Base.unalias(tab.c_workspace, A)
    _mark_with_complement!(tab, A)
    return _mutual_information_from_mask!(tab, length(A), tab.n - length(A))
end

"""
    tripartite_information(tab::AbstractTableau, A::AbstractVector, B::AbstractVector, C::AbstractVector)

Return the tripartite mutual information

    I₃(A:B:C) = S(A) + S(B) + S(C) - S(AB) - S(AC) - S(BC) + S(ABC)
              = I(A:B) + I(A:C) - I(A:BC)

of the disjoint subsystems `A`, `B` and `C`, in log-`d` units, with every
other qudit traced out. It is symmetric in its three arguments and can have
either sign: a GHZ state has `I₃ = +1` on three single qudits, while
scrambled states, as in the volume-law phase of monitored circuits, have
`I₃ < 0`. It is zero when any subsystem is empty.

The result equals the sum of seven [`entanglement_entropy`](@ref) calls. For a
mixed state the eliminations are shared, which makes it cheaper; for a pure
state it costs the same.

# Arguments
- `tab::AbstractTableau`: Destabilizer or stabilizer tableau describing the
  (possibly mixed) state. Phase storage is not required.
- `A`, `B`, `C`: 1-based qudit indices. Each must contain distinct sites in
  `1:tab.n`, and no site may be in two of them; otherwise an `ArgumentError`
  is thrown. The check is always performed and costs `O(n)`.

# Returns
- An `Int`.

# Method
The generators are reduced on the traced-out qudits to a basis of the `k`
stabilizers supported on `A ∪ B ∪ C`. With `K_X` that basis restricted to
`X`,
`I₃ = rank(K_AB) + rank(K_BC) + rank(K_CA) - rank(K_A) - rank(K_B) - rank(K_C) - k`,
and each of three eliminations yields two of the ranks. For a pure state the
seven entropies are instead computed on the smaller side of each cut, as
[`entanglement_entropy`](@ref) does. This takes at most
`O(n*m^2)` time, cubic in `n`, reads only the X and Z rows, and leaves the
generators, phases, destabilizers and canonicalization caches unchanged. It
overwrites the scratch arrays `tab.workspace`, `tab.c_workspace` and
`tab.zacc_workspace`, and allocates nothing unless a subsystem shares memory
with `tab.c_workspace`, in which case it is copied first.

# Examples
```julia
ghz = StabilizerTableau(3, 4; state=:ghz)
tripartite_information(ghz, [1], [2], [3])  # 1

# |i, j, i+j, i+2j⟩ summed over i and j: every pair of qutrits is maximally
# entangled with the other pair.
ame = StabilizerTableau(3, 4; state=:product)
foreach(q -> apply!(ame, Fourier(q)), (1, 2))
foreach(g -> apply!(ame, g), (SUM(1, 3), SUM(2, 3), SUM(1, 4), SUM(2, 4, 2)))
tripartite_information(ame, [1], [2], [3])  # -2
```
"""
function tripartite_information(
    tab::AbstractTableau,
    A::AbstractVector,
    B::AbstractVector,
    C::AbstractVector,
)
    A = Base.unalias(tab.c_workspace, A)
    B = Base.unalias(tab.c_workspace, B)
    C = Base.unalias(tab.c_workspace, C)
    fill!(tab.c_workspace, 0)
    _mark_subsystem!(tab, A, 1, "subsystem A")
    _mark_subsystem!(tab, B, 2, "subsystem B")
    _mark_subsystem!(tab, C, 3, "subsystem C")
    NA, NB, NC = length(A), length(B), length(C)
    (NA == 0 || NB == 0 || NC == 0 || tab.m == 0) && return 0
    counts = (tab.n - NA - NB - NC, NA, NB, NC)
    first = _group_sites_by_label!(tab, counts)
    if tab.m == tab.n
        # S(ABC) = S(D) for a pure state.
        return _pure_labelled_entropy(tab, 0b0010, first, counts) +
               _pure_labelled_entropy(tab, 0b0100, first, counts) +
               _pure_labelled_entropy(tab, 0b1000, first, counts) -
               _pure_labelled_entropy(tab, 0b0110, first, counts) -
               _pure_labelled_entropy(tab, 0b1010, first, counts) -
               _pure_labelled_entropy(tab, 0b1100, first, counts) +
               _pure_labelled_entropy(tab, 0b0001, first, counts)
    end
    rD, rA, rAB = _traced_rank_pair!(tab, first, counts, 1, 2)
    _, rB, rBC = _traced_rank_pair!(tab, first, counts, 2, 3)
    _, rC, rCA = _traced_rank_pair!(tab, first, counts, 3, 1)
    k = tab.m - rD
    return rAB + rBC + rCA - rA - rB - rC - k
end

# `c_workspace[q]` labels site `q`: 1 for A, 2 for B and 0 for a traced-out
# site. `NA` and `NB` are the sizes of A and B.
function _mutual_information_from_mask!(tab::AbstractTableau, NA::Int, NB::Int)
    n, m, d = tab.n, tab.m, tab.d
    (NA == 0 || NB == 0 || m == 0) && return 0
    counts = (n - NA - NB, NA, NB, 0)
    first = _group_sites_by_label!(tab, counts)
    if m == n
        # S(AB) = S(D) for a pure state.
        return _pure_labelled_entropy(tab, 0b0010, first, counts) +
               _pure_labelled_entropy(tab, 0b0100, first, counts) -
               _pure_labelled_entropy(tab, 0b0001, first, counts)
    end
    ws = tab.workspace
    rows_D = _copy_site_rows!(tab, first[1], counts[1], 0)
    rows_A = _copy_site_rows!(tab, first[2], NA, rows_D)
    rows = _copy_site_rows!(tab, first[3], NB, rows_A)
    # Pivoting on the traced-out rows alone leaves columns piv+1:m zero there,
    # and those columns are a basis of the k stabilizers supported on A ∪ B.
    piv = rank_fp_cols!(view(ws, 1:rows, 1:m), d, tab.inversemod, rows_D)
    k = m - piv
    k == 0 && return 0
    # The two restrictions occupy disjoint rows, so each elimination leaves
    # the other's entries intact.
    rank_A = rank_fp_cols!(view(ws, (rows_D + 1):rows_A, (piv + 1):m), d, tab.inversemod)
    rank_B = rank_fp_cols!(view(ws, (rows_A + 1):rows, (piv + 1):m), d, tab.inversemod)
    return rank_A + rank_B - k
end

# With the sites labelled 0 traced out, return `(rank_D, rank_X, rank_XY)`:
# the rank of the generators restricted to those sites, and the ranks on X and
# on X ∪ Y of the basis of stabilizers that vanish there. X and Y are the sites
# labelled `x` and `y`; `first` and `counts` describe the site list built by
# `_group_sites_by_label!`.
#
# The rows are stacked as [D; X; Y] and eliminated in that order. Each call
# to `rank_fp_cols!` leaves its remaining columns zero on the rows it pivoted
# on, so the next call continues on those columns and the rows below, and the
# pivot counts add up to the ranks of the leading row blocks.
function _traced_rank_pair!(
    tab::AbstractTableau,
    first::NTuple{4,Int},
    counts::NTuple{4,Int},
    x::Int,
    y::Int,
)
    m, d = tab.m, tab.d
    ws = tab.workspace
    rows_D = _copy_site_rows!(tab, first[1], counts[1], 0)
    rows_X = _copy_site_rows!(tab, first[x + 1], counts[x + 1], rows_D)
    rows = _copy_site_rows!(tab, first[y + 1], counts[y + 1], rows_X)
    rank_D = rank_fp_cols!(view(ws, 1:rows, 1:m), d, tab.inversemod, rows_D)
    c = rank_D
    rank_X = rank_fp_cols!(view(ws, (rows_D + 1):rows, (c + 1):m), d, tab.inversemod,
                           rows_X - rows_D)
    c += rank_X
    rank_Y = rank_fp_cols!(view(ws, (rows_X + 1):rows, (c + 1):m), d, tab.inversemod)
    return rank_D, rank_X, rank_X + rank_Y
end

# Entropy of a pure state on the union X of the label classes selected by the
# bit set `labels`: bit `l` selects the sites labelled `l`, for labels 0 to 3,
# and `first` and `counts` describe the site list built by
# `_group_sites_by_label!`. Since S(X) = S(complement of X), the rank is taken
# on whichever side is smaller, as `entanglement_entropy` does, and
# S = rank - size on that side.
function _pure_labelled_entropy(
    tab::AbstractTableau,
    labels::Integer,
    first::NTuple{4,Int},
    counts::NTuple{4,Int},
)
    labels = Int(labels)
    N = 0
    for l in 0:3
        (labels >> l) & 1 == 1 && (N += counts[l + 1])
    end
    if 2N > tab.n
        labels = ~labels & 0b1111
        N = tab.n - N
    end
    N == 0 && return 0
    rows = 0
    for l in 0:3
        (labels >> l) & 1 == 1 || continue
        rows = _copy_site_rows!(tab, first[l + 1], counts[l + 1], rows)
    end
    return rank_fp_cols!(view(tab.workspace, 1:rows, 1:tab.m), tab.d, tab.inversemod) - N
end
