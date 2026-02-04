### check purity
function is_pure(stabtab::StabilizerTableau)
    is_commuting(stabtab) || return false
    stabtab.m == stabtab.n || return false
    is_independent(stabtab) || return false
    return true
end

### check if all generators commute (explicitly)
function is_commuting(stabtab::StabilizerTableau)
    m = stabtab.m
    for j in 1:m
        for i in (j+1):m
            if mod(commutation_colcol(stabtab.tableau, j, i), stabtab.d) != 0
                @info "Generators $j and $i do not commute."
                return false
            end
        end
    end
    return true
end

### check if all generators commute (using matrix multiplication)
function is_commuting_matrix(stabtab::StabilizerTableau)
    n = stabtab.n
    d = stabtab.d
    m = stabtab.m
    X = view(stabtab.tableau, 1:n, 1:m)
    Z = view(stabtab.tableau, n+1:2n, 1:m)
    comm_matrix = mod.(X' * Z - Z' * X, d)
    return all(comm_matrix .== 0)
end

"""Check if the active stabilizer generators are linearly independent (rank m)."""
function is_independent(stabtab::StabilizerTableau)
    n = stabtab.n
    m = stabtab.m
    d = stabtab.d
    m == 0 && return true
    rank = rank_fp_cols!(view(stabtab.tableau, 1:2n, 1:m), d, stabtab.inversemod)
    return rank == m
end
