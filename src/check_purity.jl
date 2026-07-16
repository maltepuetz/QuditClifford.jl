"""
    is_pure(tab::AbstractTableau) -> Bool

Return `true` when the active generators define a pure stabilizer state.

Purity requires exactly one independent, mutually commuting stabilizer
generator per qudit. The check does not change the represented state and does
not emit diagnostic output.
"""
function is_pure(tab::AbstractTableau)
    tab.m == tab.n || return false
    is_commuting(tab) || return false
    is_independent(tab) || return false
    return true
end

# Check if all active generators commute.
function is_commuting(tab::AbstractTableau)
    m = tab.m
    for j in 1:m
        for i in (j+1):m
            mod(commutation_colcol(tab.stab, j, i), tab.d) != 0 && return false
        end
    end
    return true
end

"""Check if the active stabilizer generators are linearly independent (rank m)."""
function is_independent(tab::AbstractTableau)
    n = tab.n
    m = tab.m
    d = tab.d
    m == 0 && return true
    ws = tab.workspace
    @turbo for j in 1:m, i in 1:(2n)
        ws[i, j] = tab.stab[i, j]
    end
    rank = rank_fp_cols!(view(ws, 1:2n, 1:m), d, tab.inversemod)
    return rank == m
end
