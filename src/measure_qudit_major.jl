########################################
# Qudit-major measurement kernel       #
########################################
#
# In the qudit-major layout the X (Z) exponent of qudit `q` over all generators
# is one contiguous column, so the random branch of `measure!` works column by
# column over the generators instead of generator by generator:
#
# 1. the commutators of generators `1:m` with `op`, over `op`'s support columns;
# 2. the pivot, the first generator with a nonzero commutator, and its
#    replacement by `op` (the generic pivot preparation, a strided row read).
#    With no pivot, the generic commuting branch takes over;
# 3. the multipliers `α_i = -c_i / c_p`, the cross terms `x_p·z_i` and
#    `x_i·z_p` over the pivot's support columns, the phases and caches of the
#    anticommuting generators, and the rank-1 update `col_q += α · g_p[q]` of
#    every pivot-support column.
#
# When few generators anticommute, step 3 instead updates only their rows
# through the generic site-restricted `_absorb_pivot_power!`. Both give the
# results of the generator-major branch bit for bit: the same multipliers, the
# same reduced cross terms and the same phase formulas (`_absorb_pivot_scalars!`).
#
# `coeff_workspace` (n × 3, element type `E`) holds the per-generator columns,
# all reduced mod `d`: column 1 the commutators, then the multipliers `α`, and
# in the destabilizer hooks the pairings `t_j`; column 2 `x_p·z_i`; column 3
# `x_i·z_p`. `apply!` borrows the same columns, and the two never nest.

# Use the sparse update when at most `m ÷ _QM_SPARSE_UPDATE_DIV` generators
# besides the pivot anticommute (here and in the qudit-major `dephase!`).
# Measured on an M3 Pro (random-branch Z measurements, UInt8, both tableau
# types, d = 2, 3, 31, n = 1024 and 2048, monitored states at p = 0.05 and 0.15
# and random states, mean of two runs), 32 has the smallest worst case against
# the best divisor of 4, 8, 16, 32 and 64 on each state: 1.08x (d = 31,
# p = 0.15), against 1.12x for 16 and 64 and 2-3.5x for 8 and 4. Random states
# do not depend on it.
const _QM_SPARSE_UPDATE_DIV = 32

# Rows `1:m` of the X and Z columns of qudit `q`.
@inline _qm_xcol(b::_QMBlock, q::Int, m::Int) = view(b.x, 1:m, q)
@inline _qm_zcol(b::_QMBlock, q::Int, m::Int) = view(b.z, 1:m, q)

# acc[j] += x_j[q]·z - z_j[q]·x (mod d) for the rows `j` of `acc`: the term of
# qudit `q` in the symplectic pairing of every generator of `b` with a Pauli
# whose exponents on `q` are `(x, z)`, both reduced. `acc` is a separate array
# from the blocks.
@inline function _qm_pair_add!(acc::AbstractVector{E}, b::_QMBlock, q::Int, x::Int, z::Int,
                               d::Int) where {E}
    m = length(acc)
    addmul_mod!(acc, _qm_xcol(b, q, m), z, d)
    submul_mod!(acc, _qm_zcol(b, q, m), x, d)
    return acc
end

# comm[i] = <g_i, op> mod d for the generators `1:length(comm)` of `b`. `op` is
# the reduced operand of `_commutation_operand!`.
function _qm_commutators!(comm::AbstractVector{E}, b::_QMBlock, op::NPauli{D},
                          d::Int) where {E,D}
    fill!(comm, zero(E))
    @inbounds for k in 1:D
        _qm_pair_add!(comm, b, op.qudits[k], op.xs[k], op.zs[k], d)
    end
    return comm
end
function _qm_commutators!(comm::AbstractVector{E}, b::_QMBlock, op::AbstractVector{Int},
                          d::Int) where {E}
    fill!(comm, zero(E))
    n = _nq(b)
    @inbounds for q in 1:n
        x = op[q]
        z = op[n+q]
        (x == 0 && z == 0) && continue
        _qm_pair_add!(comm, b, q, x, z, d)
    end
    return comm
end

function _measure!(rng::Random.AbstractRNG, tab::AbstractTableau, L::QuditMajor,
                   op::AbstractPauli, outcome::Union{Int,Nothing}, phase_policy::Int)
    return _measure_qm!(rng, tab, L, op, outcome, phase_policy, _QM_SPARSE_UPDATE_DIV)
end

# `measure!` on a qudit-major tableau. `div` sets the sparse-update cutoff
# (`_QM_SPARSE_UPDATE_DIV`); both choices give the same result.
function _measure_qm!(rng::Random.AbstractRNG, tab::AbstractTableau, L::QuditMajor,
                      op::AbstractPauli, outcome::Union{Int,Nothing}, phase_policy::Int,
                      div::Int)
    kop = _measurement_phase(tab, op, outcome, phase_policy)
    d = tab.d
    m = tab.m

    # A dense operand lives in res_workspace, which the commutators only read.
    opc = _commutation_operand!(tab, op)
    stab = _stab(tab, L)
    comm = view(tab.coeff_workspace, 1:m, 1)
    _qm_commutators!(comm, stab, opc, d)

    pivot = 0
    @inbounds for i in 1:m
        if comm[i] != 0
            pivot = i
            break
        end
    end
    pivot == 0 && return _measure_commuting!(rng, tab, L, op, kop, outcome)
    comm0 = Int(comm[pivot])

    # The outcome is random; draw it before anything in tab changes.
    t = _random_outcome(rng, outcome, d)

    # Copy the pivot generator into pivot_workspace and its support into
    # zacc_workspace, then replace it by the measured operator.
    pu = _prepare_pivot_update(tab, L, pivot, comm0)
    set_operator!(tab, L, pivot, op)
    if tab.storephase
        _phase(tab)[pivot] = _measured_generator_phase(t, kop, d)
    end
    _on_noncommuting_col_updated!(tab, L, pivot)

    _qm_absorb_pivot!(tab, L, pivot, pu, div)

    _after_noncommuting_measurement!(tab, L, pivot, comm0, m, pu)
    tab.iscanonical = false
    return t
end

# Multiply every generator `i ≠ pivot` with commutator `c_i` (column 1 of
# coeff_workspace) by g_p^{α_i}, α_i = -c_i / c_p, where g_p is the old pivot
# generator held in pivot_workspace with its support in zacc_workspace.
function _qm_absorb_pivot!(tab::AbstractTableau, L::QuditMajor, pivot::Int,
                           pu::_PivotUpdate, div::Int)
    d = tab.d
    n = tab.n
    m = tab.m
    stab = _stab(tab, L)
    genws = tab.pivot_workspace
    sites = tab.zacc_workspace
    ws = tab.coeff_workspace

    # α in column 1, over the commutators. c_workspace holds the indices of the
    # anticommuting generators other than the pivot; nothing else uses it
    # during the random branch.
    alpha = view(ws, 1:m, 1)
    scale_mod!(alpha, pu.ninv, d)
    @inbounds alpha[pivot] = zero(eltype(alpha))
    idx = tab.c_workspace
    k = 0
    @inbounds for i in 1:m
        if alpha[i] != 0
            k += 1
            idx[k] = i
        end
    end
    k == 0 && return nothing

    if k <= m ÷ div
        # The pivot's sites cover its whole support, so the site-restricted
        # update is exact whatever `pu.sparse` says.
        pus = _PivotUpdate(pu.nsupp, true, pu.ninv, pu.kws, pu.xzws, pu.inv2)
        @inbounds for kk in 1:k
            i = idx[kk]
            _absorb_pivot_power!(tab, L, i, Int(alpha[i]), pus)
        end
        return nothing
    end

    # Cross terms x_p·z_i in column 2 and, for a DestabilizerTableau,
    # x_i·z_p in column 3, over the pivot's support columns. They read the old
    # z and x of every generator i ≠ pivot, so they run before the rank-1
    # update below.
    track = tab isa DestabilizerTableau
    cb = view(ws, 1:m, 2)
    cz = view(ws, 1:m, 3)
    if track || tab.storephase
        fill!(cb, zero(eltype(cb)))
        @inbounds for s in 1:pu.nsupp
            q = sites[s]
            addmul_mod!(cb, _qm_zcol(stab, q, m), Int(genws[q]), d)
        end
    end
    if track
        fill!(cz, zero(eltype(cz)))
        @inbounds for s in 1:pu.nsupp
            q = sites[s]
            addmul_mod!(cz, _qm_xcol(stab, q, m), Int(genws[n+q]), d)
        end
    end

    # Phases and caches, one anticommuting generator at a time.
    if track || tab.storephase
        @inbounds for kk in 1:k
            i = idx[kk]
            _absorb_pivot_scalars!(tab, i, Int(alpha[i]), Int(cb[i]), track ? Int(cz[i]) : 0, pu)
        end
    end

    # Rank-1 update of the pivot-support columns: col_q += α · g_p[q]. α is
    # zero at the pivot and at every commuting generator, so those rows keep
    # their entries. α lives in coeff_workspace, apart from the blocks.
    @inbounds for s in 1:pu.nsupp
        q = sites[s]
        addmul_mod!(_qm_xcol(stab, q, m), alpha, Int(genws[q]), d)
        addmul_mod!(_qm_zcol(stab, q, m), alpha, Int(genws[n+q]), d)
    end
    return nothing
end

# The qudit-major form of the duals' orthogonalization: the pairings t_j go
# into column 1 of coeff_workspace (free once the generators are updated; the
# append hook of the commuting branch borrows it the same way) over the support
# columns of generator `col`, then every dual support column of `col` is
# updated at once. When at most `m ÷ _QM_SPARSE_UPDATE_DIV` pairings are
# nonzero (a local measurement), only those rows of each column are updated,
# with their indices in c_workspace. `sparse` is not needed: `sites[1:nsites]`
# always lists the whole support of dual `col`. `div` sets the cutoff
# (`_QM_SPARSE_UPDATE_DIV`); every choice gives the same result.
_orthogonalize_duals!(tab::DestabilizerTableau, L::QuditMajor, col::Int, m::Int,
                      supp::Vector{Int}, nsupp::Int, sites::Vector{Int}, nsites::Int, ::Bool) =
    _qm_orthogonalize_duals!(tab, L, col, m, supp, nsupp, sites, nsites, _QM_SPARSE_UPDATE_DIV)

function _qm_orthogonalize_duals!(tab::DestabilizerTableau, L::QuditMajor, col::Int, m::Int,
                                  supp::Vector{Int}, nsupp::Int,
                                  sites::Vector{Int}, nsites::Int, div::Int)
    d = tab.d
    stab = _stab(tab, L)
    destab = _destab(tab, L)
    tj = view(tab.coeff_workspace, 1:m, 1)
    fill!(tj, zero(eltype(tj)))
    @inbounds for s in 1:nsupp
        q = supp[s]
        _qm_pair_add!(tj, destab, q, _xget(stab, q, col), _zget(stab, q, col), d)
    end
    col <= m && (@inbounds tj[col] = zero(eltype(tj)))
    idx = tab.c_workspace
    k = 0
    @inbounds for j in 1:m
        if tj[j] != 0
            k += 1
            idx[k] = j
        end
    end
    k == 0 && return nothing
    # The multipliers are entries of dual `col`, whose row t_j leaves alone.
    if k <= m ÷ div
        x = destab.x
        z = destab.z
        E = eltype(x)
        @inbounds for s in 1:nsites
            q = sites[s]
            a = _xget(destab, q, col)
            if a != 0
                for t in 1:k
                    j = idx[t]
                    x[j, q] = mod(Int(x[j, q]) - Int(tj[j]) * a, d) % E
                end
            end
            c = _zget(destab, q, col)
            if c != 0
                for t in 1:k
                    j = idx[t]
                    z[j, q] = mod(Int(z[j, q]) - Int(tj[j]) * c, d) % E
                end
            end
        end
        return nothing
    end
    @inbounds for s in 1:nsites
        q = sites[s]
        submul_mod!(_qm_xcol(destab, q, m), tj, _xget(destab, q, col), d)
        submul_mod!(_qm_zcol(destab, q, m), tj, _zget(destab, q, col), d)
    end
    return nothing
end

# `dephase!` on a qudit-major tableau: the commutators column by column, then
# every anticommuting generator other than the pivot absorbs the power
# α_i = -c_i / c_p of the pivot through `_qm_absorb_pivot!`, as in the random
# branch of `measure!`, and the pivot is dropped. The phases follow
# `mul_col_by_workspace_power!` and the destabilizer caches the patch of
# `_absorb_pivot_scalars!`, which equals their recomputation by
# `_update_xdotz_cache!`, so the result is the generic one bit for bit.
function _dephase!(tab::AbstractTableau, L::QuditMajor, op::AbstractPauli)
    _check_pauli_support(tab, op)
    d = tab.d
    m = tab.m

    # A dense operand lives in res_workspace, which nothing below writes.
    opc = _commutation_operand!(tab, op)
    comm = view(tab.coeff_workspace, 1:m, 1)
    _qm_commutators!(comm, _stab(tab, L), opc, d)

    pivot = 0
    @inbounds for i in 1:m
        if comm[i] != 0
            pivot = i
            break
        end
    end
    pivot == 0 && return tab

    pu = _prepare_pivot_update(tab, L, pivot, Int(comm[pivot]))
    _qm_absorb_pivot!(tab, L, pivot, pu, _QM_SPARSE_UPDATE_DIV)
    _drop_generator!(tab, L, pivot)
    tab.iscanonical = false
    return tab
end
