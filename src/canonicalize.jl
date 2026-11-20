# Column-reduce the active generators `1:m` of `b` in place. `phase` holds the
# phase exponents when `storephase` and may be empty otherwise; it is never
# read or written without `storephase`. The pivot scan walks the combined rows
# `1:2n` in order, X rows first, so a column's pivot row decides whether its X
# block is zero.
@inline function _canonicalize_tableau!(
    b::XZBlock,
    phase::AbstractVector{Int},
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

    n = _nq(b)
    nrows_block = 2n

    if storephase
        @assert length(phase) == n
        d_phase = phase_modulus(d)
        inv2 = (d == 2) ? 0 : inv(2, d)  # only needed for odd d
    else
        d_phase = 0
        inv2 = 0
    end
    @assert _ncap(b) == n  # capacity is n; only 1:m are active
    (0 <= m <= n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ n"))

    r = 1
    c = 1
    while r <= nrows_block && c <= m
        # find pivot in row r among columns c..m
        j = c
        while j <= m && _rget(b, r, j) == 0
            j += 1
        end
        if j > m
            r += 1
            continue
        end

        # swap columns j <-> c
        if j != c
            _swap_gens!(b, c, j)
            if storephase
                t = phase[c]
                phase[c] = phase[j]
                phase[j] = t
            end
        end

        # scale pivot column so entry (r, c) = 1  (power by α)
        # For d=2, α is always 1 when pivot is nonzero; for odd d, normal inverse.
        α = inv(_rget(b, r, c), d)

        # Compute x·z for pivot column when needed
        xdotz = storephase ? dot_xz_col(b, c, d) : 0

        if storephase
            k_old = phase[c]
            if d == 2 # α=1 always, and we store phase mod 4 as i^k, so scaling does nothing
                phase[c] = mod(k_old, d_phase)
            else
                phase[c] = mod(α * k_old + binom2_mod_oddprime(α, d, inv2) * xdotz, d_phase)
            end
        end

        _gen_scale!(b, c, α, d)

        if storephase && d != 2
            # update x·z for scaled generator: (αx)·(αz) = α^2 (x·z)
            xdotz = mod(mod(α * α, d) * xdotz, d)
        end

        # eliminate pivot row r from all other (active) columns
        for jj in 1:m
            jj == c && continue
            β = _rget(b, r, jj)
            β == 0 && continue

            if storephase
                texp = mod(-β, d)

                kc = phase[c]  # phase exponent of pivot generator (mod d_phase)

                if d == 2
                    # Multiply g_jj <- g_jj * g_c, with phase mod 4.
                    # (g_c)^texp is g_c (texp=1), so no binom term needed.
                    # Cross-term lifts to mod 4 with factor 2: add 2*(x_c·z_jj)
                    cross = dot_xz_col(b, c, jj, d) # mod 2
                    phase[jj] = mod(phase[jj] + kc + 2 * cross, d_phase)
                else
                    # Multiply g_jj <- g_jj * (g_c)^texp, with phase mod d.
                    # Binomial term needed for (g_c)^texp.
                    kc_power_t = mod(texp * kc + binom2_mod_oddprime(texp, d, inv2) * xdotz, d_phase)
                    cross = mod(texp * dot_xz_col(b, c, jj, d), d)
                    phase[jj] = mod(phase[jj] + kc_power_t + cross, d_phase)
                end
            end

            # jj != c: distinct generators of one block, so dst and src cannot alias.
            _gen_submul!(b, jj, b, c, β, d)
        end

        pivcol_of_row[r] = c
        r += 1
        c += 1
    end

    # compute xdotz_cache after canonicalization (valid entries 1:m)
    @inbounds for j in 1:m
        xdotz_cache[j] = dot_xz_col(b, j, d)
    end
    @inbounds for j in (m+1):n
        xdotz_cache[j] = 0
    end

    return nothing
end

# `_canonicalize_tableau!` that also carries the destabilizer block `db` through
# the same column operations, keeping `⟨D_j, S_k⟩ = δ_jk`.
@inline function _canonicalize_tableau_with_destab!(
    b::XZBlock,
    phase::AbstractVector{Int},
    db::XZBlock,
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

    n = _nq(b)
    nrows_block = 2n

    if storephase
        @assert length(phase) == n
        d_phase = phase_modulus(d)
        inv2 = (d == 2) ? 0 : inv(2, d)  # only needed for odd d
    else
        d_phase = 0
        inv2 = 0
    end
    @assert _ncap(b) == n  # capacity is n; only 1:m are active
    @assert _nq(db) == n
    @assert _ncap(db) == n
    (0 <= m <= n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ n"))

    r = 1
    c = 1
    while r <= nrows_block && c <= m
        # find pivot in row r among columns c..m
        j = c
        while j <= m && _rget(b, r, j) == 0
            j += 1
        end
        if j > m
            r += 1
            continue
        end

        # swap columns j <-> c
        if j != c
            _swap_gens!(b, c, j)
            if storephase
                t = phase[c]
                phase[c] = phase[j]
                phase[j] = t
            end
            _swap_gens!(db, c, j)
        end

        # scale pivot column so entry (r, c) = 1  (power by α)
        # For d=2, α is always 1 when pivot is nonzero; for odd d, normal inverse.
        pivot = _rget(b, r, c)
        α = inv(pivot, d)

        # Compute x·z for pivot column when needed
        xdotz = storephase ? dot_xz_col(b, c, d) : 0

        if storephase
            k_old = phase[c]
            if d == 2 # α=1 always, and we store phase mod 4 as i^k, so scaling does nothing
                phase[c] = mod(k_old, d_phase)
            else
                phase[c] = mod(α * k_old + binom2_mod_oddprime(α, d, inv2) * xdotz, d_phase)
            end
        end

        _gen_scale!(b, c, α, d)

        # Update destabilizer column by α^{-1} (inverse transpose of the column scaling).
        # At d = 2 the pivot is 1, so this is a no-op that scale_mod! skips entirely.
        αinv = mod(pivot, d)
        _gen_scale!(db, c, αinv, d)

        if storephase && d != 2
            # update x·z for scaled generator: (αx)·(αz) = α^2 (x·z)
            xdotz = mod(mod(α * α, d) * xdotz, d)
        end

        # eliminate pivot row r from all other (active) columns
        for jj in 1:m
            jj == c && continue
            β = _rget(b, r, jj)
            β == 0 && continue

            if storephase
                texp = mod(-β, d)

                kc = phase[c]  # phase exponent of pivot generator (mod d_phase)

                if d == 2
                    # Multiply g_jj <- g_jj * g_c, with phase mod 4.
                    # (g_c)^texp is g_c (texp=1), so no binom term needed.
                    # Cross-term lifts to mod 4 with factor 2: add 2*(x_c·z_jj)
                    cross = dot_xz_col(b, c, jj, d) # mod 2
                    phase[jj] = mod(phase[jj] + kc + 2 * cross, d_phase)
                else
                    # Multiply g_jj <- g_jj * (g_c)^texp, with phase mod d.
                    # Binomial term needed for (g_c)^texp.
                    kc_power_t = mod(texp * kc + binom2_mod_oddprime(texp, d, inv2) * xdotz, d_phase)
                    cross = mod(texp * dot_xz_col(b, c, jj, d), d)
                    phase[jj] = mod(phase[jj] + kc_power_t + cross, d_phase)
                end
            end

            # jj != c, so these are disjoint generators of one block.
            _gen_submul!(b, jj, b, c, β, d)

            # Update inverse-transpose dual basis for S_jj <- S_jj - β*S_c.
            # All eliminated columns contribute to the pivot dual D_c.
            # jj != c, so these are disjoint columns of destab.
            _gen_addmul!(db, c, db, jj, β, d)
        end

        pivcol_of_row[r] = c
        r += 1
        c += 1
    end

    # compute xdotz_cache after canonicalization (valid entries 1:m)
    @inbounds for j in 1:m
        xdotz_cache[j] = dot_xz_col(b, j, d)
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
```jldoctest
julia> tab = DestabilizerTableau(2, 3; state = :ghz);

julia> canonicalize!(tab)

julia> tab == DestabilizerTableau(2, 3; state = :ghz)   # the same state
true
```

# Notes
- Operates on columns `1:m` (active generators). Columns `m+1:n` are unused capacity.
- If `storephase=true`, the phases are updated so the represented stabilizer subgroup is unchanged.
- Updates `tab.pivcol_of_row`, `tab.xdotz_cache`, and sets `tab.iscanonical = true`.
- For `DestabilizerTableau`, canonicalization also updates the destabilizers to
  preserve the stabilizer/destabilizer duality.
"""
function canonicalize!(tab::StabilizerTableau)
    _canonicalize!(tab, _layout_marker(tab))
    return nothing
end

function canonicalize!(tab::DestabilizerTableau)
    _canonicalize!(tab, _layout_marker(tab))
    return nothing
end

function _canonicalize!(tab::StabilizerTableau, L::TableauLayout)
    _canonicalize_tableau!(
        _stab(tab, L),
        _phase(tab),
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

function _canonicalize!(tab::DestabilizerTableau, L::TableauLayout)
    _canonicalize_tableau_with_destab!(
        _stab(tab, L),
        _phase(tab),
        _destab(tab, L),
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
