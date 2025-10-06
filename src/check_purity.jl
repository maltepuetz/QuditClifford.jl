### check purity (not allocation free)
function is_pure(stabtab::StabilizerTableau)
    is_commuting(stabtab) || return false
    is_full_rank(stabtab) || return false
    return true
end


### check if all generators commute (explicitly)
function is_commuting(stabtab::StabilizerTableau)
    for j in axes(stabtab.tableau, 2)
        gen_j = Generator(stabtab, j)
        for i in j+1:size(stabtab.tableau, 2)
            gen_i = Generator(stabtab, i)
            if mod(commutation(gen_j, gen_i), stabtab.d) != 0
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
    X = view(stabtab.tableau, 1:n, :)
    Z = view(stabtab.tableau, n+1:2n, :)
    comm_matrix = mod.(X' * Z - Z' * X, d)
    return all(comm_matrix .== 0)
end

### check if the rank of the stabilizer generators is n
function is_full_rank(stabtab::StabilizerTableau)
    n = stabtab.n
    d = stabtab.d
    rank = rank_fp_cols!(stabtab.tableau[1:2n, 1:n], d, stabtab.inversemod)
    return rank == n
end
