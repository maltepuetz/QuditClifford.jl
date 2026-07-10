### check purity
function is_pure(tab::AbstractTableau)
    is_commuting(tab) || return false
    tab.m == tab.n || return false
    is_independent(tab) || return false
    return true
end

### check if all generators commute (explicitly)
function is_commuting(tab::AbstractTableau)
    m = tab.m
    for j in 1:m
        for i in (j+1):m
            if mod(commutation_colcol(tab.stab, j, i), tab.d) != 0
                @info "Generators $j and $i do not commute."
                return false
            end
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
