"""
    canonicalize!(stabtab::StabilizerTableau)

RCEF (column Gauss–Jordan) on stabtab.tableau, with correct phase updates if
stabtab.storephase=true. Updates stabtab.pivcol_of_row internally such that 
pivcol_of_row[r] = pivot column index for row r, or 0 if no pivot.
"""
function canonicalize!(stabtab::StabilizerTableau)
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
        d_phase = phase_modulus(d)
        inv2 = (d == 2) ? 0 : inv(2, d)  # only needed for odd d
    else
        @assert size(tab, 1) == 2n
        phase_row = 0
        d_phase = 0
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
        # For d=2, α is always 1 when pivot is nonzero; for odd d, normal inverse.
        α = inv(tab[r, c], d)

        # Compute x·z for pivot column when needed
        xdotz = hasphase ? dot_xz_col(tab, n, c, d) : 0

        if hasphase
            k_old = tab[phase_row, c]
            if d == 2
                # α=1 always, and we store phase mod 4 as i^k, so scaling does nothing
                tab[phase_row, c] = mod(k_old, d_phase)
            else
                tab[phase_row, c] = mod(α * k_old + binom2_mod_oddprime(α, d, inv2) * xdotz, d_phase)
            end
        end

        @turbo for i in 1:nrows_block
            tab[i, c] = mod(tab[i, c] * α, d)
        end

        if hasphase && d != 2
            # update x·z for scaled generator: (αx)·(αz) = α^2 (x·z)
            xdotz = mod(mod(α * α, d) * xdotz, d)
        end

        # eliminate pivot row r from all other columns
        for jj in 1:n
            jj == c && continue
            β = tab[r, jj]
            β == 0 && continue

            if hasphase
                texp = mod(-β, d)

                kc = tab[phase_row, c]  # phase exponent of pivot generator (mod d_phase)

                if d == 2
                    # Multiply g_jj <- g_jj * g_c, with phase mod 4.
                    # (g_c)^texp is g_c (texp=1), so no binom term needed.
                    # Cross-term lifts to mod 4 with factor 2: add 2*(x_c·z_jj)
                    cross = dot_xz_col(tab, n, c, jj, d) # mod 2
                    tab[phase_row, jj] = mod(tab[phase_row, jj] + kc + 2 * cross, d_phase)
                else
                    # Multiply g_jj <- g_jj * (g_c)^texp, with phase mod d.
                    # Binomial term needed for (g_c)^texp.
                    kc_power_t = mod(texp * kc + binom2_mod_oddprime(texp, d, inv2) * xdotz, d_phase)
                    cross = mod(texp * dot_xz_col(tab, n, c, jj, d), d)
                    tab[phase_row, jj] = mod(tab[phase_row, jj] + kc_power_t + cross, d_phase)
                end
            end

            @inbounds @simd for i in 1:nrows_block
                tab[i, jj] = mod(tab[i, jj] - mod(β * tab[i, c], d), d)
            end
        end

        pivcol_of_row[r] = c
        r += 1
        c += 1
    end

    # compute xdotz_cache after canonicalization
    xdotz_cache = stabtab.xdotz_cache
    @inbounds for j in eachindex(xdotz_cache)
        xdotz_cache[j] = dot_xz_col(tab, n, j, d)
    end

    return nothing
end
