# The phase exponent of the pivot generator after it is raised to the power `α`
# (`xdotz` is its x·z before the scaling). At d = 2, α = 1 and the phase mod 4
# stays.
@inline function _scaled_pivot_phase(k_old::Int, α::Int, xdotz::Int, d::Int, d_phase::Int,
                                     inv2::Int)
    d == 2 && return mod(k_old, d_phase)
    return mod(α * k_old + binom2_mod_oddprime(α, d, inv2) * xdotz, d_phase)
end

# The phase exponent of generator g_jj after g_jj <- g_jj · g_c^(-β), from its
# phase `kj`, the pivot's phase `kc`, `cross = x_c·z_jj mod d` and the pivot's
# x·z `xdotz`.
@inline function _eliminated_phase(kj::Int, kc::Int, β::Int, cross::Int, xdotz::Int, d::Int,
                                   d_phase::Int, inv2::Int)
    if d == 2
        # g_jj <- g_jj · g_c, phase mod 4: the cross term lifts with a factor 2.
        return mod(kj + kc + 2 * cross, d_phase)
    end
    # g_jj <- g_jj · (g_c)^texp, phase mod d, with the binomial term of the power.
    texp = mod(-β, d)
    kc_power_t = mod(texp * kc + binom2_mod_oddprime(texp, d, inv2) * xdotz, d_phase)
    return mod(kj + kc_power_t + mod(texp * cross, d), d_phase)
end

# How the destabilizers `db` follow the column operations on the stabilizers:
# not at all for `nothing` (a `StabilizerTableau`), and for a block by the
# inverse transpose of each operation, which keeps `⟨D_j, S_k⟩ = δ_jk`.
@inline _dual_swap!(::Nothing, ::Int, ::Int) = nothing
@inline _dual_swap!(db::_AnyBlock, c::Int, j::Int) = (_swap_gens!(db, c, j); nothing)
# The pivot generator was scaled by α = 1/pivot, so its dual is scaled by
# pivot. At d = 2 the pivot is 1 and `scale_mod!` returns at once.
@inline _dual_scale!(::Nothing, ::Int, ::Int, ::Int) = nothing
@inline _dual_scale!(db::_AnyBlock, c::Int, pivot::Int, d::Int) =
    (_gen_scale!(db, c, mod(pivot, d), d); nothing)
# S_jj <- S_jj - β S_c adds β D_jj into the pivot dual D_c; jj != c, so these
# are disjoint generators of one block.
@inline _dual_addmul!(::Nothing, ::Int, ::Int, ::Int, ::Int) = nothing
@inline _dual_addmul!(db::_AnyBlock, c::Int, jj::Int, β::Int, d::Int) =
    (_gen_addmul!(db, c, db, jj, β, d); nothing)

# Column-reduce the active generators `1:m` of `b` in place, carrying the
# destabilizers `db` (`nothing` for a `StabilizerTableau`) through the same
# column operations. `phase` holds the phase exponents when `storephase` and
# may be empty otherwise; it is never read or written without `storephase`.
# The pivot scan walks the combined rows `1:2n` in order, X rows first, so a
# column's pivot row decides whether its X block is zero.
@inline function _canonicalize_tableau!(
    b::_AnyBlock,
    phase::AbstractVector{Int},
    db::Union{Nothing,_AnyBlock},
    m::Int,
    d::Int,
    storephase::Bool,
    inv::InverseMod,
    pivcol_of_row::Vector{Int},
    xdotz_cache::AbstractVector{Int},
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
    if db !== nothing
        @assert _nq(db) == n
        @assert _ncap(db) == n
    end
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
            _dual_swap!(db, c, j)
        end

        # scale pivot column so entry (r, c) = 1  (power by α)
        # For d=2, α is always 1 when pivot is nonzero; for odd d, normal inverse.
        pivot = _rget(b, r, c)
        α = inv(pivot, d)

        # Compute x·z for pivot column when needed
        xdotz = storephase ? dot_xz_col(b, c, d) : 0

        if storephase
            phase[c] = _scaled_pivot_phase(phase[c], α, xdotz, d, d_phase, inv2)
        end

        _gen_scale!(b, c, α, d)
        _dual_scale!(db, c, pivot, d)

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
                phase[jj] = _eliminated_phase(phase[jj], phase[c], β, dot_xz_col(b, c, jj, d),
                                              xdotz, d, d_phase, inv2)
            end

            # jj != c: distinct generators of one block, so dst and src cannot alias.
            _gen_submul!(b, jj, b, c, β, d)
            _dual_addmul!(db, c, jj, β, d)
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
function canonicalize!(tab::AbstractTableau)
    _canonicalize!(tab, _layout_marker(tab))
    return nothing
end

_canonicalize!(tab::AbstractTableau, L::TableauLayout) = _canonicalize_generic!(tab, L)

# The destabilizer block `canonicalize!` carries along: none for a
# `StabilizerTableau`.
@inline _canon_destab(::StabilizerTableau, ::TableauLayout) = nothing
@inline _canon_destab(tab::DestabilizerTableau, L::TableauLayout) = _destab(tab, L)

# The per-generator `canonicalize!`, for every layout and storage.
function _canonicalize_generic!(tab::AbstractTableau, L::TableauLayout)
    _canonicalize_tableau!(
        _stab(tab, L),
        _phase(tab),
        _canon_destab(tab, L),
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
