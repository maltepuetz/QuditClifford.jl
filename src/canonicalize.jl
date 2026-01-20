# Helper: dot product x_i ⋅ z_j for a given column i and j, mod d.
# useful when updating phases while multiplying stabilizer generators
@inline function dot_xz_col(tab, n::Int, i::Int, j::Int, d::Int)
    s = zero(eltype(tab))
    @turbo for q in 1:n
        s += tab[q, i] * tab[n+q, j]
    end
    return mod(s, d)
end
@inline dot_xz_col(tab, n::Int, j::Int, d::Int) = dot_xz_col(tab, n, j, j, d)

# Helper: C(t,2) mod d = t*(t-1)/2 mod d for prime d (needs inv2 = inv(2) mod d).
@inline function binom2_mod(t::Int, d::Int, inv2::Int)
    return mod(mod(t * (t - 1), d) * inv2, d)
end

"""
    canonicalize!(stabtab::StabilizerTableau)

RCEF (column Gauss–Jordan) on stabtab.tableau, with correct phase updates if
stabtab.storephase=true. Updates stabtab.pivcol_of_row internally such that 
pivcol_of_row[r] = pivot column index for row r, or 0 if no pivot.
"""
function canonicalize!(
    stabtab::StabilizerTableau,
)
    tab = stabtab.tableau
    n = stabtab.n
    d = stabtab.d
    inv = stabtab.inversemod
    pivcol_of_row = stabtab.pivcol_of_row
    @turbo for i in eachindex(pivcol_of_row)
        pivcol_of_row[i] = 0
    end

    hasphase = stabtab.storephase
    nrows_block = 2n

    if hasphase
        @assert size(tab, 1) == 2n + 1
        phase_row = 2n + 1
        inv2 = inv(2, d)
    else
        @assert size(tab, 1) == 2n
        phase_row = 0
        inv2 = 0
    end
    @assert size(tab, 2) == n

    r = 1
    c = 1
    while r <= nrows_block && c <= n
        # find pivot in row r among columns c..n
        j = c
        while j <= n && tab[r, j] == 0
            j += 1
        end
        if j > n
            r += 1
            continue
        end

        # swap columns j <-> c (swap full stored rows)
        if j != c
            @turbo for i in axes(tab, 1)
                t = tab[i, c]
                tab[i, c] = tab[i, j]
                tab[i, j] = t
            end
        end

        # scale pivot column so tab[r,c]=1  (power by α)
        α = inv(tab[r, c], d)

        if hasphase
            xdotz = dot_xz_col(tab, n, c, d)
            k_old = tab[phase_row, c]
            tab[phase_row, c] = mod(α * k_old + binom2_mod(α, d, inv2) * xdotz, d)
        end

        @turbo for i in 1:nrows_block
            tab[i, c] = mod(tab[i, c] * α, d)
        end
        
        if hasphase
            xdotz = mod(mod(α^2, d) * xdotz, d)
        end

        # eliminate pivot row r from all other columns
        for jj in 1:n
            jj == c && continue
            β = tab[r, jj]
            β == 0 && continue

            if hasphase
                # g_jj -> g_jj * (g_c)^{t}, where t = -β mod d
                texp = mod(-β, d)

                kc = tab[phase_row, c]
                kc_power_t = mod(texp * kc + binom2_mod(texp, d, inv2) * xdotz, d)
                cross = mod(texp * dot_xz_col(tab, n, c, jj, d), d)

                tab[phase_row, jj] = mod(tab[phase_row, jj] + kc_power_t + cross, d)
            end

            @inbounds @simd for i in 1:nrows_block
                tab[i, jj] = mod(tab[i, jj] - mod(β * tab[i, c], d), d)
            end
        end

        pivcol_of_row[r] = c
        r += 1
        c += 1
    end

    return nothing
end
