###########################################
# Commutation / symplectic inner products #
###########################################

@inline function commutation(s::AbstractVector{<:Integer}, v::AbstractVector{<:Integer})
    N = length(s) ÷ 2
    comm = 0

    @turbo for i in 1:N
        comm += s[i] * v[i+N]
    end
    @turbo for i in 1:N
        comm -= s[i+N] * v[i]
    end
    comm
end

@inline commutation_vec_op(v::AbstractVector{<:Integer}, op::AbstractVector{<:Integer}) = commutation(v, op)
@inline commutation_vec_op(v::AbstractVector{<:Integer}, op::GeneralPauli) = commutation(v, op.xz)
@inline function commutation_vec_op(v::AbstractVector{<:Integer}, op::SinglePauli)
    N = length(v) ÷ 2
    return v[op.qudit] * op.z - v[op.qudit+N] * op.x
end
@inline function commutation_vec_op(v::AbstractVector{<:Integer}, op::DoublePauli)
    N = length(v) ÷ 2
    comm = 0
    comm += v[op.qudit1] * op.z1 - v[op.qudit1+N] * op.x1
    comm += v[op.qudit2] * op.z2 - v[op.qudit2+N] * op.x2
    return comm
end
@inline function commutation_vec_op(v::AbstractVector{<:Integer}, op::TriplePauli)
    N = length(v) ÷ 2
    comm = 0
    comm += v[op.qudit1] * op.z1 - v[op.qudit1+N] * op.x1
    comm += v[op.qudit2] * op.z2 - v[op.qudit2+N] * op.x2
    comm += v[op.qudit3] * op.z3 - v[op.qudit3+N] * op.x3
    return comm
end
@inline function commutation_vec_op(v::AbstractVector{<:Integer}, op::NPauli{D}) where D
    N = length(v) ÷ 2
    comm = 0
    @turbo for i in 1:D
        q = op.qudits[i]
        comm += v[q] * op.zs[i] - v[q+N] * op.xs[i]
    end
    return comm
end

@inline function commutation_col(b::XZBlock, col::Int, s::AbstractVector{<:Integer})
    N = _nq(b)
    xg = _xgen(b, col)
    zg = _zgen(b, col)
    comm = 0

    @turbo for i in 1:N
        comm += Int(xg[i]) * s[i+N]
    end
    @turbo for i in 1:N
        comm -= Int(zg[i]) * s[i]
    end
    comm
end
# Packed, the set bits of generator `col` select the operand entries they pair
# with.
@inline function commutation_col(b::BitBlock, col::Int, s::AbstractVector{<:Integer})
    N = _nq(b)
    comm = 0
    @inbounds for q in 1:N
        comm += _xget(b, q, col) * s[q+N] - _zget(b, q, col) * s[q]
    end
    comm
end
@inline function commutation_col(b::_AnyBlock, col::Int, op::GeneralPauli)
    return commutation_col(b, col, op.xz)
end

@inline function commutation_col(b::_AnyBlock, col::Int, op::SinglePauli)
    comm = _xget(b, op.qudit, col) * op.z - _zget(b, op.qudit, col) * op.x
    comm
end
@inline function commutation_col(b::_AnyBlock, col::Int, op::DoublePauli)
    comm = 0
    comm += _xget(b, op.qudit1, col) * op.z1 - _zget(b, op.qudit1, col) * op.x1
    comm += _xget(b, op.qudit2, col) * op.z2 - _zget(b, op.qudit2, col) * op.x2
    comm
end
@inline function commutation_col(b::_AnyBlock, col::Int, op::TriplePauli)
    comm = 0
    comm += _xget(b, op.qudit1, col) * op.z1 - _zget(b, op.qudit1, col) * op.x1
    comm += _xget(b, op.qudit2, col) * op.z2 - _zget(b, op.qudit2, col) * op.x2
    comm += _xget(b, op.qudit3, col) * op.z3 - _zget(b, op.qudit3, col) * op.x3
    comm
end
# The measurement scan calls this once per generator column, and in the
# generator-major layout each entry it reads sits in a different cache line. A
# zero exponent skips its load, which halves the misses of a pure Z or X
# operand. Callers validate the qudit indices before the scan.
@inline function commutation_col(b::_AnyBlock, col::Int, op::NPauli{D}) where D
    comm = 0
    @inbounds for i in 1:D
        q = op.qudits[i]
        z = op.zs[i]
        x = op.xs[i]
        z == 0 || (comm += _xget(b, q, col) * z)
        x == 0 || (comm -= _zget(b, q, col) * x)
    end
    comm
end

# Reduce exponents once before dot products, keeping local operators sparse.
@inline function _commutation_operand!(tab::AbstractTableau, op::_BuiltinFewQuditPauli)
    qudits, xs, zs, _ = _few_qudit_pauli_data(op)
    d = tab.d
    return NPauli(qudits, map(x -> mod(x, d), xs), map(z -> mod(z, d), zs))
end

@inline function _commutation_operand!(tab::AbstractTableau, op::AbstractPauli)
    set_operator!(tab.res_workspace, tab, op)
    return tab.res_workspace
end

# Whether `op` commutes with every active generator of `tab`.
function _commutes_with_generators(tab::AbstractTableau, L::TableauLayout, op::AbstractPauli)
    normalized = _commutation_operand!(tab, op)
    d = tab.d
    stab = _stab(tab, L)
    for i in 1:tab.m
        mod(commutation_col(stab, i, normalized), d) == 0 || return false
    end
    return true
end

#########################################################
# Column update primitive used in noncommuting branch   #
#########################################################

"""
Multiply a tableau column by (pivot_workspace)^a on the RIGHT, with correct phase updates.

This is used in the noncommuting update of `dephase!`:
    g_i <- g_i * (g_pivot)^a
where the X and Z exponents of g_pivot are stored in pivot_workspace and its
phase exponent is `kws` (0 without a phase row).

- XZ rows are updated linearly mod d.
- If storephase=false: no phase bookkeeping.
- If storephase=true:
    odd prime d:
        (ω^k X^x Z^z)(ω^k' X^x' Z^z') = ω^(k+k' + x'·z) X^(x+x') Z^(z+z')
    d=2:
        use phase mod 4 as i^k and cross-term contributes 2*(x'·z) mod 4.
"""
function mul_col_by_workspace_power!(
    tab::AbstractTableau,
    L::TableauLayout,
    col::Int,
    a::Int,
    kws::Int,
)
    stab = _stab(tab, L)
    genws = tab.pivot_workspace
    n = tab.n
    d = tab.d

    a == 0 && return nothing

    # If we store phase, compute cross term using the OLD target Z (before XZ update)
    if tab.storephase
        phase = _phase(tab)
        d_phase = phase_modulus(d)

        if d == 2
            # a ∈ {0,1} and a != 0, so a == 1
            # cross = x_ws · z_col_old   (mod 2)
            cross = dot_xz_ws_vs_col(genws, stab, col, d)  # uses old tab Z entries
            phase[col] = mod(phase[col] + kws + 2 * cross, d_phase)
        else
            inv2 = tab.inversemod(2, d)

            # Phase of (ws)^a:
            # k_power = a*k_ws + C(a,2)*(x_ws · z_ws)  (mod d)
            xdotz_ws = dot_xz_ws(genws, n, d)
            k_power = mod(a * kws + binom2_mod_oddprime(a, d, inv2) * xdotz_ws, d_phase)

            # cross = a*(x_ws · z_col_old) (mod d)
            cross_base = dot_xz_ws_vs_col(genws, stab, col, d)  # x_ws · z_col_old
            cross = mod(a * cross_base, d)

            phase[col] = mod(phase[col] + k_power + cross, d_phase)
        end
    end

    # Now update XZ: col <- col + a*ws  (mod d)
    # genws is a separate array from the block of the same element type, and its
    # contents were copied out of a tableau column, so they are reduced.
    _gen_addmul!(stab, col, genws, a, d)

    return nothing
end

##############################################################
# Noncommuting-branch update, restricted to the pivot support #
##############################################################

# In the noncommuting branch every later anticommuting generator absorbs a
# power of the pivot generator g_p, whose X and Z exponents are held in
# pivot_workspace, in the storage element type of the tableau:
#     g_i <- g_i * g_p^a,   a = -c_i / c_p (mod d).
# Everything that depends on g_p alone -- its phase, x_p·z_p, -1/c_p, 1/2 -- is
# fixed for the whole measurement, so `_PivotUpdate` holds it and each
# generator pays only for its own terms. Every O(n) update of a generator is a
# vector primitive between two arrays of the storage element type.
#
# g_p vanishes outside its qudit support, so the XZ update of g_i, its cross
# term x_p·z_i and (for a `DestabilizerTableau`) the change of x_i·z_i all live
# on that support. In monitored circuits the pivot is local, a handful of sites
# while dozens of generators anticommute, and the restricted update turns
# O(k·n) into O(n + k·|supp|). A pivot with a large support (a random state)
# keeps the dense SIMD loops, which win once the gathered loop would visit more
# than about half of the `n` sites.
struct _PivotUpdate
    nsupp::Int      # sites of g_p, stored in tab.zacc_workspace[1:nsupp]
    sparse::Bool    # whether updates run over those sites or over all rows
    ninv::Int       # -1/c_p mod d
    kws::Int        # phase exponent of g_p (0 without a phase row)
    xzws::Int       # x_p·z_p mod d
    inv2::Int       # 1/2 mod d for odd d, 0 for d = 2
end

# Dense loops stream 2n entries per generator with SIMD; the gathered loop
# costs more per visited site but skips the rest. Measured on an M3 Pro, the
# gathered update is 1.7-3.4x faster with a pivot on a quarter of the sites and
# breaks even at about half of them for d = 2 and 3 (still 1.9x at d = 31). A
# random state's pivot covers about three quarters and stays dense.
@inline _pivot_support_is_sparse(nsupp::Int, n::Int) = 2 * nsupp <= n

# Copy the X and Z exponents of generator `pivot` into pivot_workspace, its
# phase into `kws`, and record its support in zacc_workspace, which nothing else
# touches during a measurement. Must run before the pivot column is
# overwritten.
function _prepare_pivot_update(tab::AbstractTableau, L::TableauLayout, pivot::Int, comm0::Int)
    stab = _stab(tab, L)
    genws = tab.pivot_workspace
    n = tab.n
    d = tab.d
    _copy_gen!(genws, stab, pivot)
    sites = tab.zacc_workspace
    nsupp = _column_support!(sites, stab, pivot)
    sparse = _pivot_support_is_sparse(nsupp, n)
    if sparse
        s = 0
        @inbounds for k in 1:nsupp
            q = sites[k]
            s += Int(genws[q]) * Int(genws[n+q])
        end
        xzws = mod(s, d)
    else
        xzws = dot_xz_ws(genws, n, d)
    end
    kws = tab.storephase ? _phase(tab)[pivot] : 0
    inv2 = d == 2 ? 0 : tab.inversemod(2, d)
    ninv = mod(-tab.inversemod(comm0, d), d)
    return _PivotUpdate(nsupp, sparse, ninv, kws, xzws, inv2)
end

# The phase of g_col * g_p^a and, for a `DestabilizerTableau`, its
# `xdotz_cache` entry, from the cross terms cb = x_p·z_col (reduced) and
# cz = x_col·z_p of the old generator `col`.
@inline function _absorb_pivot_scalars!(tab::AbstractTableau, col::Int, a::Int, cb::Int,
                                        cz::Int, pu::_PivotUpdate)
    d = tab.d
    if tab.storephase
        phase = _phase(tab)
        if d == 2
            # a == 1; the cross term enters i^k as 2·(x_p·z_col).
            phase[col] = mod(phase[col] + pu.kws + 2 * cb, 4)
        else
            k_power = mod(a * pu.kws + binom2_mod_oddprime(a, d, pu.inv2) * pu.xzws, d)
            phase[col] = mod(phase[col] + k_power + mod(a * cb, d), d)
        end
    end
    # Only a DestabilizerTableau keeps `xdotz_cache` valid, so only it patches it.
    if tab isa DestabilizerTableau
        cache = tab.xdotz_cache
        lin = mod(a * mod(cb + mod(cz, d), d), d)
        quad = mod(mod(a * a, d) * pu.xzws, d)
        cache[col] = mod(cache[col] + lin + quad, d)
    end
    return nothing
end

# g_col <- g_col * g_p^a for the prepared pivot. Same phase arithmetic as
# `mul_col_by_workspace_power!`, with the pivot terms read from `pu`; for a
# `DestabilizerTableau` it also patches `xdotz_cache[col]` from
#     (x + a x_p)·(z + a z_p) = x·z + a (x_p·z + x·z_p) + a² x_p·z_p.
@inline function _absorb_pivot_power!(tab::AbstractTableau, L::TableauLayout, col::Int, a::Int,
                                      pu::_PivotUpdate)
    a == 0 && return nothing
    stab = _stab(tab, L)
    xg = _xgen(stab, col)
    zg = _zgen(stab, col)
    genws = tab.pivot_workspace
    n = tab.n
    d = tab.d
    # Only a DestabilizerTableau keeps `xdotz_cache` valid, so only it patches it.
    track = tab isa DestabilizerTableau
    sites = tab.zacc_workspace

    # cb = x_p·z_col and cz = x_col·z_p, both on the old column, with every
    # entry widened to `Int` before it multiplies.
    cb = 0
    cz = 0
    if pu.sparse
        if track
            @inbounds for k in 1:pu.nsupp
                q = sites[k]
                cb += Int(genws[q]) * Int(zg[q])
                cz += Int(xg[q]) * Int(genws[n+q])
            end
        elseif tab.storephase
            @inbounds for k in 1:pu.nsupp
                q = sites[k]
                cb += Int(genws[q]) * Int(zg[q])
            end
        end
    else
        if track
            @turbo for q in 1:n
                cb += Int(genws[q]) * Int(zg[q])
                cz += Int(xg[q]) * Int(genws[n+q])
            end
        elseif tab.storephase
            @turbo for q in 1:n
                cb += Int(genws[q]) * Int(zg[q])
            end
        end
    end
    _absorb_pivot_scalars!(tab, col, a, mod(cb, d), cz, pu)

    # genws is a separate array from the block of the same element type, and its
    # contents were copied out of a tableau column, so they are reduced. Outside
    # the pivot support genws is zero, so the site-restricted update gives the
    # dense result.
    if pu.sparse
        _gen_addmul_sites!(stab, col, genws, sites, pu.nsupp, a, d)
    else
        _gen_addmul!(stab, col, genws, a, d)
    end
    return nothing
end

# Packed storage: the same cross terms through the leaves, and the update
# through the bit composites.
@inline function _absorb_pivot_power!(tab::_BitTableau, L::TableauLayout, col::Int, a::Int,
                                      pu::_PivotUpdate)
    a == 0 && return nothing
    stab = _stab(tab, L)
    genws = tab.pivot_workspace
    n = tab.n
    d = tab.d
    track = tab isa DestabilizerTableau
    sites = tab.zacc_workspace
    cb = 0
    cz = 0
    if track || tab.storephase
        if pu.sparse
            @inbounds for k in 1:pu.nsupp
                q = sites[k]
                cb += Int(genws[q]) * _zget(stab, q, col)
                cz += _xget(stab, q, col) * Int(genws[n+q])
            end
        else
            @inbounds for q in 1:n
                cb += Int(genws[q]) * _zget(stab, q, col)
                cz += _xget(stab, q, col) * Int(genws[n+q])
            end
        end
    end
    _absorb_pivot_scalars!(tab, col, a, mod(cb, d), cz, pu)
    if pu.sparse
        _gen_addmul_sites!(stab, col, genws, sites, pu.nsupp, a, d)
    else
        _gen_addmul!(stab, col, genws, a, d)
    end
    return nothing
end

############################################
# Projective measurement (mixed-state)     #
############################################

@inline function _effective_measurement_phase(op::AbstractPauli, d::Int, phase_policy::Int)::Int
    (0 <= phase_policy <= 2) || throw(ArgumentError("phase_policy must be 0, 1, or 2."))
    d != 2 && return op.phase

    valid = _is_valid_measurement(op, d)
    valid && return op.phase

    if phase_policy == 0
        @warn "Measuring a non-Hermitian Pauli for d=2; results may be unphysical. Set phase_policy=2 to silence or phase_policy=1 to auto-fix."
        return op.phase
    elseif phase_policy == 1
        parity = _xdotz_parity(op)
        return _fix_qubit_phase(op.phase, parity)
    else
        return op.phase
    end
end

#############################################################
# Type-specific hooks for generic projective measurement    #
#############################################################

@inline _on_noncommuting_col_updated!(tab::AbstractTableau, ::TableauLayout, _col::Int) = nothing
@inline _on_noncommuting_col_updated!(tab::DestabilizerTableau, L::TableauLayout, col::Int) =
    _update_xdotz_cache!(tab, L, col)

@inline function _after_noncommuting_measurement!(
    tab::AbstractTableau,
    ::TableauLayout,
    _pivot::Int,
    _comm0::Int,
    _m::Int,
    _pu::_PivotUpdate,
)
    return nothing
end

@inline function _after_noncommuting_measurement!(
    tab::DestabilizerTableau,
    L::TableauLayout,
    pivot::Int,
    comm0::Int,
    m::Int,
    pu::_PivotUpdate,
)
    d = tab.d
    stab = _stab(tab, L)
    destab = _destab(tab, L)

    # noncommuting branch: update destabilizers
    inv_comm0 = tab.inversemod(comm0, d)
    _gen_mulcopy!(destab, pivot, tab.pivot_workspace, inv_comm0, d)

    # Every dual must stay orthogonal to the newly written generator, which now
    # holds the measured operator. Pairing each dual against it over all 2n rows
    # would cost O(n*m) per measurement. But `set_operator!` zeroed the column
    # before writing it, so every term outside the operator's own qudit support
    # vanishes: collect that support once in O(n) and each pairing drops to
    # O(nsupp) -- a handful of terms for a sparse Pauli. A dense operator has
    # full support and pays the full O(n*m); only zero terms go unvisited.
    supp = tab.support_workspace
    nsupp = _column_support!(supp, stab, pivot)

    # The new pivot dual is a multiple of the old pivot generator, so it
    # vanishes outside that generator's sites, which `pu` recorded in
    # zacc_workspace.
    _orthogonalize_duals!(tab, L, pivot, m, supp, nsupp, tab.zacc_workspace, pu.nsupp, pu.sparse)
    return nothing
end

# Make the duals `1:m` other than `col` orthogonal to generator `col`:
#     D_j <- D_j - t_j D_col,   t_j = <D_j, S_col>.
# `supp[1:nsupp]` covers the support of generator `col` and `sites[1:nsites]`
# that of dual `col`; `sparse` says whether the update runs over those sites or
# over all rows. D_col is fixed throughout, so each t_j is the same whether it
# is read before or after the other duals change.
_orthogonalize_duals!(tab::DestabilizerTableau, L::TableauLayout, col::Int, m::Int,
                      supp::Vector{Int}, nsupp::Int, sites::Vector{Int}, nsites::Int,
                      sparse::Bool) =
    _orthogonalize_duals_generic!(tab, L, col, m, supp, nsupp, sites, nsites, sparse)

_orthogonalize_duals!(tab::_BitDestab, L::QuditMajor, col::Int, m::Int,
                      supp::Vector{Int}, nsupp::Int, sites::Vector{Int}, nsites::Int,
                      sparse::Bool) =
    _orthogonalize_duals_generic!(tab, L, col, m, supp, nsupp, sites, nsites, sparse)

function _orthogonalize_duals_generic!(tab::DestabilizerTableau, L::TableauLayout, col::Int,
                                       m::Int, supp::Vector{Int}, nsupp::Int,
                                       sites::Vector{Int}, nsites::Int, sparse::Bool)
    d = tab.d
    stab = _stab(tab, L)
    destab = _destab(tab, L)
    @inbounds for j in 1:m
        j == col && continue
        traw = _symplectic_col_col_support(destab, j, stab, col, supp, nsupp)
        traw == 0 && continue
        t = mod(traw, d)
        t == 0 && continue
        # j != col (guarded above), so these are disjoint columns of destab.
        if sparse
            _gen_submul_sites!(destab, j, destab, col, sites, nsites, t, d)
        else
            _gen_submul!(destab, j, destab, col, t, d)
        end
    end
    return nothing
end

@inline function _after_append_measurement!(
    tab::AbstractTableau,
    ::TableauLayout,
    _newcol::Int,
    _res::AbstractVector{Int},
)
    return nothing
end

@inline function _after_append_measurement!(
    tab::DestabilizerTableau,
    L::TableauLayout,
    newcol::Int,
    res::AbstractVector{Int},
)
    n = tab.n
    d = tab.d
    m = newcol - 1

    # Construct a new destabilizer without elimination.
    # Use residual to pick a qudit.
    q = 0
    @inbounds for i in 1:n
        if res[i] != 0 || res[n+i] != 0
            q = i
            break
        end
    end
    q == 0 && throw(ArgumentError("Failed to construct destabilizer: residual is zero."))

    stab = _stab(tab, L)
    destab = _destab(tab, L)
    # The new destabilizer v is built in pivot_workspace, in the storage element
    # type, so the projection below runs on same-type vector primitives. The
    # append branch has no pivot generator, so pivot_workspace holds nothing
    # live here.
    genws = tab.pivot_workspace
    fill!(genws, zero(eltype(genws)))
    zq = res[q] != 0
    if zq
        genws[n+q] = 1  # Z_q
    else
        genws[q] = 1    # X_q
    end

    # Project v = e - Σ_j t_j D_j to commute with the old stabilizers, where e
    # is the unit vector above and t_j = <v, S_j> at the time D_j is
    # subtracted. Duality makes <D_i, S_j> = δ_ij, so subtracting earlier
    # duals leaves <v, S_j> = <e, S_j>: one entry of S_j, -x_q for e = Z_q and
    # z_q for e = X_q. Reading it costs one load per generator instead of a
    # 2n-term pairing.
    @inbounds for j in 1:m
        traw = zq ? _xget(stab, q, j) : _zget(stab, q, j)
        traw == 0 && continue
        t = zq ? d - traw : traw
        # genws is a separate array from the destabilizer block, of the same
        # element type.
        _vec_submul!(genws, destab, j, t, d)
    end

    # The new column holds op with reduced exponents. Every pairing against it
    # vanishes outside op's qudit support, so collect that support once.
    supp = tab.support_workspace
    nsupp = _column_support!(supp, stab, newcol)

    # β = <v, op>
    β = mod(_symplectic_vec_col_support(genws, stab, newcol, supp, nsupp), d)
    β == 0 && throw(ArgumentError("Failed to construct destabilizer: β = 0."))
    invβ = tab.inversemod(β, d)

    # D_new = invβ * v
    _gen_mulcopy!(destab, newcol, genws, invβ, d)

    # Orthogonalize old destabilizers to the new stabilizer. D_new vanishes
    # wherever v does, so when it is local the update runs over its sites,
    # collected in zacc_workspace, which the append branch does not otherwise
    # use; a dense D_new keeps the SIMD loop (see `_pivot_support_is_sparse`).
    sites = tab.zacc_workspace
    ndsupp = _column_support!(sites, destab, newcol)
    dsparse = _pivot_support_is_sparse(ndsupp, n)
    # m == newcol - 1, so every j in 1:m is a dual other than newcol.
    _orthogonalize_duals!(tab, L, newcol, m, supp, nsupp, sites, ndsupp, dsparse)

    _update_xdotz_cache!(tab, L, newcol)
    return nothing
end

"""
    measure!([rng::AbstractRNG,] tab::AbstractTableau, op;
        outcome::Union{Int,Nothing}=nothing,
        phase_policy::Int=0
    )

Projectively measure a Pauli operator `op` and update `tab` in-place.

# Arguments
- `rng::AbstractRNG`: Source of the outcome when the measurement is random. The
  form without `rng` uses `Random.default_rng()`.
- `tab::AbstractTableau`: Tableau to update (`DestabilizerTableau` or `StabilizerTableau`).
- `op`: Pauli operator specified as `AbstractPauli` (e.g. `SinglePauli`, `DoublePauli`,
  `TriplePauli`, `NPauli`, `GeneralPauli`), or an `AbstractVector{<:Integer}` of length
  `2n` or `2n+1` (wrapped into `GeneralPauli` internally).

# Keyword Arguments
- `outcome::Union{Int,Nothing}=nothing`: Outcome used when the measurement is random.
  With `nothing`, it is drawn uniformly from `0:(d-1)` with `rng`, and only when the
  measurement turns out to be random: a deterministic measurement draws nothing, so
  it leaves `rng` untouched. An explicit `outcome` replaces the draw; it must lie in
  `0:(d-1)` and uses the encoding of the returned value.
- `phase_policy::Int=0`: How to handle non-Hermitian Paulis when `d=2`.
  - `0`: warn (default), keep phase as-is.
  - `1`: auto-fix phase to a Hermitian one, no warning.
  - `2`: ignore and keep phase as-is, no warning.

# Returns
- Integer outcome `t` in `0:(d-1)`, in every branch: the measured eigenvalue is `ω^t`
  for odd prime `d` and `(-1)^t` for `d=2`.
- A random branch returns the drawn or given `outcome`; a deterministic one returns the
  outcome fixed by the state, so repeating a measurement returns the same `t`.
- For `d=2` this is half the exponent [`expect_int!`](@ref) reports: `⟨P⟩ = i^(2t)`.
- If `storephase=false`, a deterministic measurement returns `0` by convention, whatever
  the eigenvalue.

# Throws
- `ArgumentError` if `op` is not a Pauli on the register: a sparse Pauli with a qudit
  index outside `1:n` or with a repeated index, a [`GeneralPauli`](@ref) whose vector
  does not have length `2n`, or a plain vector whose length is neither `2n` nor `2n+1`.
- `ArgumentError` if `outcome` is outside `0:(d-1)`.
- `ArgumentError` if `phase_policy` is not `0`, `1` or `2`. This is checked for every
  `d`, but only when `storephase=true`; without a phase row the keyword is not read.
- `ArgumentError` if a random measurement would append a generator to a tableau that
  already has `m == n`, or if a [`DestabilizerTableau`](@ref) cannot build the
  destabilizer of an appended generator. Neither occurs for a tableau whose generators
  commute and are independent, which the raw-matrix constructors verify unless
  `check=false`.
- `ArgumentError` for `d=2` with `storephase=true` if the measurement is deterministic
  with eigenvalue `±i`. This needs a non-Hermitian `op` kept by `phase_policy` `0` or
  `2`, or a non-Hermitian stored generator. `tab` still represents the same state.

# Examples
```jldoctest
julia> using Random

julia> tab = DestabilizerTableau(2, 2; state = :product, basis = :Z);

julia> t = measure!(Xoshiro(1), tab, SinglePauli(1, 1, 0));   # X on qudit 1: random outcome

julia> t in 0:1
true

julia> measure!(tab, SinglePauli(1, 1, 0)) == t               # repeating it is deterministic
true

julia> measure!(tab, SinglePauli(2, 1, 0); outcome = 1)       # a chosen outcome for a random branch
1

julia> ghz = DestabilizerTableau(3, 2; state = :ghz);

julia> measure!(ghz, DoublePauli(1, 0, 1, 2, 0, 2))           # Z₁ Z₂², fixed by the state
0
```

# Notes
- If `op` does not commute with all generators, the outcome is random and the first
  non-commuting generator is replaced; other non-commuting generators are adjusted
  to restore commutation.
- If `op` commutes with all generators, the outcome is deterministic when `op` lies
  in the stabilizer span; otherwise the outcome is random and a new generator is
  appended (if `m < n`).
- The outcome is drawn before the tableau changes, so an exception from `rng` leaves
  `tab` representing the same state.
- One seeded `rng` passed to every `measure!`, [`random_clifford!`](@ref) and
  [`random_state!`](@ref) call reproduces a whole monitored-circuit trajectory.
- For a pure tableau (`m == n`) with `storephase=false`, a commuting `op` is settled by
  the commutation check alone and the tableau, including its generator basis, is left
  as it was. This costs `O(km)` for a sparse Pauli on `k` qudits, or `O(nm)` for a
  dense Pauli. Otherwise a commuting `op` needs its decomposition over the generators.
  A [`DestabilizerTableau`](@ref) reads it off its dual basis in `O(km)`. A canonical
  `StabilizerTableau` reads it off its pivot rows in `O(n)`; one that is not canonical
  solves for it by elimination on a copy of its generators, one bit per entry for
  `d=2` and one entry of its storage element type otherwise, at up to `O(nm²)`
  operations on those. Neither changes the
  generator basis or `iscanonical`: a fixed outcome leaves the tableau as it was, and
  an appended generator goes after the existing ones. Prefer a `DestabilizerTableau`
  for measurement-heavy circuits with phases, and call [`canonicalize!`](@ref) first
  when measuring many commuting operators on a `StabilizerTableau` whose generators
  do not change in between.
"""
function measure!(rng::Random.AbstractRNG, tab::AbstractTableau,
    op::AbstractVector{<:Integer};
    outcome::Union{Int,Nothing}=nothing,
    phase_policy::Int=0,
)
    gop = GeneralPauli(tab.n, tab.d, op)
    return measure!(rng, tab, gop; outcome=outcome, phase_policy=phase_policy)
end

function measure!(tab::AbstractTableau, op::Union{AbstractPauli,AbstractVector{<:Integer}};
    outcome::Union{Int,Nothing}=nothing,
    phase_policy::Int=0,
)
    return measure!(Random.default_rng(), tab, op; outcome=outcome, phase_policy=phase_policy)
end

# The outcome of a random measurement branch: the caller's `outcome`, or a
# fresh draw from `rng`.
@inline _random_outcome(rng::Random.AbstractRNG, outcome::Union{Int,Nothing}, d::Int) =
    outcome === nothing ? _draw(rng, d) : outcome

@inline _check_outcome(::Nothing, ::Int) = nothing
@inline function _check_outcome(outcome::Int, d::Int)
    (0 <= outcome < d) || throw(ArgumentError(
        "outcome must be in 0:$(d - 1) for d = $d, got $outcome."))
    return nothing
end

# Phase exponent of the generator that stabilizes the post-measurement state
# after outcome `t`, for a measured operator with reduced phase exponent `kop`.
# If ω^{kop} P_xz |ψ'> = ω^t |ψ'> then (ω^{-t + kop} P_xz) |ψ'> = |ψ'>.
# For d=2: (-1)^b = i^(2b), so the required stabilizer phase exponent is 2*b (mod 4).
@inline function _measured_generator_phase(t::Int, kop::Int, d::Int)
    d == 2 && return mod(2 * t + kop, phase_modulus(d))
    return mod(-t + kop, d)  # phase modulus is d for odd primes
end

function measure!(rng::Random.AbstractRNG, tab::AbstractTableau, op::AbstractPauli;
    outcome::Union{Int,Nothing}=nothing, # outcome used if random
    phase_policy::Int=0,
)
    return _measure!(rng, tab, _layout_marker(tab), op, outcome, phase_policy)
end

# The argument checks shared by every layout's `measure!`, and the phase
# exponent of the measured operator under `phase_policy`, reduced mod
# `phase_modulus(d)` so that the outcome arithmetic cannot overflow.
function _measurement_phase(tab::AbstractTableau, op::AbstractPauli,
                            outcome::Union{Int,Nothing}, phase_policy::Int)
    _check_pauli_support(tab, op)
    d = tab.d
    _check_outcome(outcome, d)
    kop = tab.storephase ? _effective_measurement_phase(op, d, phase_policy) : op.phase
    return mod(kop, phase_modulus(d))
end

_measure!(rng::Random.AbstractRNG, tab::AbstractTableau, L::TableauLayout, op::AbstractPauli,
          outcome::Union{Int,Nothing}, phase_policy::Int) =
    _measure_generic!(rng, tab, L, op, outcome, phase_policy)

# Packed storage runs the per-generator algorithm in both layouts.
_measure!(rng::Random.AbstractRNG, tab::_BitTableau, L::QuditMajor, op::AbstractPauli,
          outcome::Union{Int,Nothing}, phase_policy::Int) =
    _measure_generic!(rng, tab, L, op, outcome, phase_policy)

# The per-generator `measure!`, for every layout and storage.
function _measure_generic!(rng::Random.AbstractRNG, tab::AbstractTableau, L::TableauLayout,
                           op::AbstractPauli, outcome::Union{Int,Nothing}, phase_policy::Int)
    kop = _measurement_phase(tab, op, outcome, phase_policy)
    d = tab.d
    n = tab.n
    m = tab.m

    # Outcome of a random branch, drawn on reaching one.
    t = 0

    comm0 = 0  # commutation with the first non-commuting generator
    pivot = 0
    pu = _PivotUpdate(0, false, 0, 0, 0, 0)

    # Dot products with unreduced exponents can overflow. A dense operand lives
    # in res_workspace, which nothing in this loop writes; the commuting branch
    # below overwrites it from `op` again.
    opc = _commutation_operand!(tab, op)
    stab = _stab(tab, L)

    for i in 1:m

        # commutator of generator i with the operator; most generators of a
        # local state miss the operator's support entirely, and a zero dot
        # needs no reduction
        craw = commutation_col(stab, i, opc)
        craw == 0 && continue
        commutator = mod(craw, d)

        ### case a: if they commute we continue
        commutator == 0 && continue

        ### case b:
        # If the current generator is the first one that does not commute, we replace it
        # by the measured operator (with phase chosen to match the sampled outcome).
        # Else we multiply the generator with the generator that we replaced to some power.
        # The power is chosen, such that the new generator commutes with the operator.
        if comm0 == 0 # first non-commuting generator becomes pivot
            # The outcome is random; draw it before anything in tab changes.
            t = _random_outcome(rng, outcome, d)

            # store generator i in workspace, with everything the later
            # generators need from it, and then replace it by the operator
            pu = _prepare_pivot_update(tab, L, i, commutator)

            # replace generator i by op (phase-aware if storephase=true)
            set_operator!(tab, L, i, op)

            # IMPORTANT for mixed-state measurement:
            # we override the phase so that the new generator stabilizes the post-measurement state,
            # dependent on the sampled outcome, and the phase of the input operator.
            if tab.storephase
                _phase(tab)[i] = _measured_generator_phase(t, kop, d)
            end

            comm0 = commutator
            pivot = i
            _on_noncommuting_col_updated!(tab, L, i)
        else
            # multiply generator i by the (generator that we replaced)^(a)
            # where a = mod(-commutator * invmod(comm0, d), d)
            # this ensures that the new generator commutes with the operator;
            # the update also keeps xdotz_cache[i] current.
            a = mod(commutator * pu.ninv, d)
            _absorb_pivot_power!(tab, L, i, a, pu)
        end
    end

    ######################################################
    # If comm0 != 0, we were in the noncommuting branch. #
    # Measurement was random, tableau has been updated,  #
    # and `t` is the returned measurement result.        #
    ######################################################
    if comm0 != 0
        _after_noncommuting_measurement!(tab, L, pivot, comm0, m, pu)
        tab.iscanonical = false
        return t
    end
    return _measure_commuting!(rng, tab, L, op, kop, outcome)
end

# The branch of `measure!` for an `op` that commutes with every generator,
# with `kop` the reduced phase exponent of `op`. It overwrites res_workspace.
function _measure_commuting!(rng::Random.AbstractRNG, tab::AbstractTableau, L::TableauLayout,
                             op::AbstractPauli, kop::Int, outcome::Union{Int,Nothing})
    d = tab.d
    n = tab.n
    m = tab.m

    # A pure state's n independent generators span a Lagrangian subspace (d
    # prime), so a Pauli commuting with all of them lies in the span: the outcome
    # is deterministic, and without phases it is `0` by convention.
    tab.m == n && !tab.storephase && return 0

    #########################################################
    # Commuting branch (comm0 == 0): mixed-state semantics. #
    # Need to check if op is in the stabilizer span or not. #
    #########################################################

    # check whether op is in the stabilizer span and set coeffs in c_workspace;
    # the generator basis stays as it is
    _span_coeffs_keep_basis!(tab, L, op)
    in_span = _in_span_from_coeffs!(tab, L, op)
    res = tab.res_workspace

    if in_span
        # deterministic outcome, state unchanged
        if tab.storephase
            kacc = phase_exponent_from_coeffs!(tab, L)
            if d == 2
                # The eigenvalue is i^k; a Hermitian operator on Hermitian
                # generators has even k, and (-1)^t = i^(2t).
                k = mod(-kacc + kop, phase_modulus(d))
                isodd(k) && throw(ArgumentError(
                    "the measured operator has eigenvalue $(k == 1 ? "i" : "-i") on this state, " *
                    "so it has no outcome in 0:1: the operator or a stored generator is not " *
                    "Hermitian. Pass phase_policy=1 to measure the Hermitian operator, or use " *
                    "expect_int! for the phase exponent."))
                return k >> 1
            else
                return mod(-kacc + kop, d)
            end
        else
            return 0
        end
    end
    # commuting but not in span: random outcome, append a generator
    (m < n) || throw(ArgumentError("Cannot append generator: tableau at full capacity (m==n)."))
    t = _random_outcome(rng, outcome, d)

    newcol = m + 1
    tab.m = newcol
    set_operator!(tab, L, newcol, op)
    if tab.storephase
        _phase(tab)[newcol] = _measured_generator_phase(t, kop, d)
    end
    _after_append_measurement!(tab, L, newcol, res)

    tab.iscanonical = false
    return t
end
