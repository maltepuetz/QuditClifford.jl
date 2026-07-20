@inline function _canonicalize_tableau!(
    tab::AbstractMatrix{Int},
    n::Int,
    m::Int,
    d::Int,
    storephase::Bool,
    inv::InverseMod,
    pivcol_of_row::Vector{Int},
    xdotz_cache::Vector{Int},
)
    @turbo for i in eachindex(pivcol_of_row)
        pivcol_of_row[i] = 0
    end

    nrows_block = 2n

    if storephase
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
    @assert size(tab, 2) == n  # capacity is n; only 1:m are active
    (0 <= m <= n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ n"))

    r = 1
    c = 1
    while r <= nrows_block && c <= m
        # find pivot in row r among columns c..m
        j = c
        while j <= m && tab[r, j] == 0
            j += 1
        end
        if j > m
            r += 1
            continue
        end

        # swap columns j <-> c
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
        xdotz = storephase ? dot_xz_col(tab, n, c, d) : 0

        if storephase
            k_old = tab[phase_row, c]
            if d == 2 # α=1 always, and we store phase mod 4 as i^k, so scaling does nothing
                tab[phase_row, c] = mod(k_old, d_phase)
            else
                tab[phase_row, c] = mod(α * k_old + binom2_mod_oddprime(α, d, inv2) * xdotz, d_phase)
            end
        end

        @turbo for i in 1:nrows_block
            tab[i, c] = mod(tab[i, c] * α, d)
        end

        if storephase && d != 2
            # update x·z for scaled generator: (αx)·(αz) = α^2 (x·z)
            xdotz = mod(mod(α * α, d) * xdotz, d)
        end

        # eliminate pivot row r from all other (active) columns
        for jj in 1:m
            jj == c && continue
            β = tab[r, jj]
            β == 0 && continue

            if storephase
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

    # compute xdotz_cache after canonicalization (valid entries 1:m)
    @inbounds for j in 1:m
        xdotz_cache[j] = dot_xz_col(tab, n, j, d)
    end
    @inbounds for j in (m+1):n
        xdotz_cache[j] = 0
    end

    return nothing
end

@inline function _canonicalize_tableau_with_destab!(
    tab::AbstractMatrix{Int},
    destab::AbstractMatrix{Int},
    n::Int,
    m::Int,
    d::Int,
    storephase::Bool,
    inv::InverseMod,
    pivcol_of_row::Vector{Int},
    xdotz_cache::Vector{Int},
)
    @turbo for i in eachindex(pivcol_of_row)
        pivcol_of_row[i] = 0
    end

    nrows_block = 2n

    if storephase
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
    @assert size(tab, 2) == n  # capacity is n; only 1:m are active
    @assert size(destab, 1) == 2n
    @assert size(destab, 2) == n
    (0 <= m <= n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ n"))

    r = 1
    c = 1
    while r <= nrows_block && c <= m
        # find pivot in row r among columns c..m
        j = c
        while j <= m && tab[r, j] == 0
            j += 1
        end
        if j > m
            r += 1
            continue
        end

        # swap columns j <-> c
        if j != c
            @turbo for i in axes(tab, 1)
                t = tab[i, c]
                tab[i, c] = tab[i, j]
                tab[i, j] = t
            end
            @turbo for i in axes(destab, 1)
                t = destab[i, c]
                destab[i, c] = destab[i, j]
                destab[i, j] = t
            end
        end

        # scale pivot column so tab[r,c]=1  (power by α)
        # For d=2, α is always 1 when pivot is nonzero; for odd d, normal inverse.
        pivot = tab[r, c]
        α = inv(pivot, d)

        # Compute x·z for pivot column when needed
        xdotz = storephase ? dot_xz_col(tab, n, c, d) : 0

        if storephase
            k_old = tab[phase_row, c]
            if d == 2 # α=1 always, and we store phase mod 4 as i^k, so scaling does nothing
                tab[phase_row, c] = mod(k_old, d_phase)
            else
                tab[phase_row, c] = mod(α * k_old + binom2_mod_oddprime(α, d, inv2) * xdotz, d_phase)
            end
        end

        @turbo for i in 1:nrows_block
            tab[i, c] = mod(tab[i, c] * α, d)
        end

        # Update destabilizer column by α^{-1} (inverse transpose of the column scaling).
        αinv = mod(pivot, d)
        @inbounds @simd for i in 1:nrows_block
            destab[i, c] = mod(destab[i, c] * αinv, d)
        end

        if storephase && d != 2
            # update x·z for scaled generator: (αx)·(αz) = α^2 (x·z)
            xdotz = mod(mod(α * α, d) * xdotz, d)
        end

        # eliminate pivot row r from all other (active) columns
        for jj in 1:m
            jj == c && continue
            β = tab[r, jj]
            β == 0 && continue

            if storephase
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

            # Update inverse-transpose dual basis for S_jj <- S_jj - β*S_c.
            # All eliminated columns contribute to the pivot dual D_c.
            @inbounds @simd for i in 1:nrows_block
                destab[i, c] = mod(destab[i, c] + mod(β * destab[i, jj], d), d)
            end
        end

        pivcol_of_row[r] = c
        r += 1
        c += 1
    end

    # compute xdotz_cache after canonicalization (valid entries 1:m)
    @inbounds for j in 1:m
        xdotz_cache[j] = dot_xz_col(tab, n, j, d)
    end
    @inbounds for j in (m+1):n
        xdotz_cache[j] = 0
    end

    return nothing
end

"""
    canonicalize!(tab::StabilizerTableau)
    canonicalize!(tab::DestabilizerTableau)

Put the active generator columns of `tab` into column-reduced echelon form (RCEF),
updating phases consistently and caching pivot metadata.

# Arguments
- `tab::AbstractTableau`: Tableau to canonicalize in-place.

# Returns
- `nothing`. Mutates `tab`.

# Examples
```julia
tab = DestabilizerTableau(2, 3; state=:ghz)
canonicalize!(tab)
```

# Notes
- Operates on columns `1:m` (active generators). Columns `m+1:n` are unused capacity.
- If `storephase=true`, the phase row is updated so the represented stabilizer subgroup is unchanged.
- Updates `tab.pivcol_of_row`, `tab.xdotz_cache`, and sets `tab.iscanonical = true`.
- For `DestabilizerTableau`, canonicalization also updates `tab.destab` to preserve
  the stabilizer/destabilizer duality.
"""
function canonicalize!(tab::StabilizerTableau)
    _canonicalize_tableau!(
        tab.stab,
        tab.n,
        tab.m,
        tab.d,
        tab.storephase,
        tab.inversemod,
        tab.pivcol_of_row,
        tab.xdotz_cache,
    )
    tab.iscanonical = true
    return nothing
end

function canonicalize!(tab::DestabilizerTableau)
    _canonicalize_tableau_with_destab!(
        tab.stab,
        tab.destab,
        tab.n,
        tab.m,
        tab.d,
        tab.storephase,
        tab.inversemod,
        tab.pivcol_of_row,
        tab.xdotz_cache,
    )
    tab.iscanonical = true
    return nothing
end
