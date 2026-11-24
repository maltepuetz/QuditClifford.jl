"""
    conditional_mutual_information(tab::AbstractTableau, A::AbstractVector, B::AbstractVector, C::AbstractVector)

Return the conditional mutual information

    I(A:B|C) = S(AC) + S(BC) - S(C) - S(ABC)

of the disjoint subsystems `A` and `B` given `C`, in log-`d` units (so
`I = 1` means `log(d)`), with every other qudit traced out. It is symmetric
in `A` and `B`, nonnegative by strong subadditivity, and zero exactly when
`ρ_ABC` is a quantum Markov chain `A - C - B`. With `C` empty it is
[`mutual_information`](@ref)`(tab, A, B)`. It satisfies the chain rule
`I(A:BC) = I(A:C) + I(A:B|C)`, and `I₃(A:B:C) = I(A:B) - I(A:B|C)` for the
[`tripartite_information`](@ref).

The result equals the sum of four [`entanglement_entropy`](@ref) calls but
takes a single elimination: on mixed states it is several times faster, and
on pure states it is at least as fast.

# Arguments
- `tab::AbstractTableau`: Destabilizer or stabilizer tableau describing the
  (possibly mixed) state. Phase storage is not required.
- `A`, `B`, `C`: 1-based qudit indices. Each must contain distinct sites in
  `1:tab.n`, and no site may be in two of them; otherwise an `ArgumentError`
  is thrown. The check is always performed and costs `O(n)`.

# Returns
- A nonnegative `Int`. It is zero when `A` or `B` is empty.

# Method
The generators are reduced on the traced-out qudits to a basis of the `k`
stabilizers supported on `A ∪ B ∪ C`. With `K_X` that basis restricted to
`X`, each entropy above is `|X| - k + rank(K_Y)` with `Y` the rest of
`A ∪ B ∪ C`, and the sizes and `k` cancel:
`I(A:B|C) = rank(K_A) + rank(K_B) - rank(K_AB)`. The rows of `C` are never
read. For a pure state `I(A:B|C) = I(A:B|D)`, with `D` the traced-out qudits,
so whichever of `C` and `D` is smaller is traced out. This takes at most
`O(n*m^2)` time, cubic in `n`, reads only the X and Z rows, and leaves the
generators, phases, destabilizers and canonicalization caches unchanged. It
overwrites the scratch arrays `tab.c_workspace` and `tab.zacc_workspace` and
the rank workspace for `d` (`tab.bit_workspace` for `d = 2`, else
`tab.rank_workspace`), and allocates nothing unless a subsystem shares memory
with `tab.c_workspace`, in which case it is copied first.

# Examples
```jldoctest
julia> ghz = StabilizerTableau(3, 4; state = :ghz);

julia> conditional_mutual_information(ghz, [1], [2], [3])  # qudit 3 accounts for the correlation
0

julia> mutual_information(ghz, [1], [2])
1

julia> bell = StabilizerTableau(3, 3; state = :product, basis = [:X, :Z, :Z]);

julia> apply!(bell, SUM(1, 3));   # a Bell pair on qudits 1 and 3, with qudit 2 in |0⟩

julia> conditional_mutual_information(bell, [1], [3], [2])
2
```
"""
function conditional_mutual_information(
    tab::AbstractTableau,
    A::AbstractVector,
    B::AbstractVector,
    C::AbstractVector,
)
    NA, NB, NC = _mark_subsystems!(tab, (A, B, C))
    (NA == 0 || NB == 0 || tab.m == 0) && return 0
    counts = (tab.n - NA - NB - NC, NA, NB, NC)
    first = _group_sites_by_label!(tab, counts)
    if tab.m == tab.n && counts[4] < counts[1]
        # A pure state on A ∪ B ∪ C ∪ D has S(X) = S(complement of X), which
        # turns I(A:B|C) into I(A:B|D). Tracing out C is then cheaper.
        first = (first[4], first[2], first[3], first[1])
        counts = (counts[4], counts[2], counts[3], counts[1])
    end

    # Rows are stacked as [D; X; Y] with Y the smaller of A and B, since Y is
    # eliminated twice.
    x, y = NA >= NB ? (1, 2) : (2, 1)
    return _cmi_from_groups!(tab, _layout_marker(tab), _rank_workspace(tab), first, counts, x, y)
end

function _cmi_from_groups!(
    tab::AbstractTableau,
    L::TableauLayout,
    ws::AbstractMatrix,
    first::NTuple{4,Int},
    counts::NTuple{4,Int},
    x::Int,
    y::Int,
)
    rank_D, rank_X, rank_XY = _traced_rank_pair!(tab, L, ws, first, counts, x, y)
    # Every elimination after the one on D combines only the columns
    # rank_D+1:m, the basis of the stabilizers that vanish on D, so the rows
    # of Y in those columns still have the rank of K_Y.
    rows_X = _block_end(ws, _block_end(ws, 0, counts[1]), counts[x + 1])
    rows = _block_end(ws, rows_X, counts[y + 1])
    rank_Y = _block_rank!(ws, rows_X, rows, (rank_D + 1):tab.m, tab.d, tab.inversemod)
    return rank_X + rank_Y - rank_XY
end
