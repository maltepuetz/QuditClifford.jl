using QuditClifford
using Test
using Random

# Differential tests of `measure!` against a plain dense reference for its
# noncommuting branch: every generator update over all 2n rows, x·z of the
# pivot recomputed per generator, `xdotz_cache` recomputed per updated column,
# and the destabilizer re-orthogonalization over all 2n rows. The package's
# support-restricted kernel must reproduce it bit for bit -- tableau, destab,
# cache, outcome, raised errors and RNG consumption -- across both tableau
# types, every Pauli type, all three branches, both phase settings and every
# phase policy. The commuting branches call the package's own span
# decomposition, which both sides share.

const MK = QuditClifford

function mkref_mul_col!(tab, col::Int, a::Int)
    stab = stabilizer_matrix(tab)
    genws = tab.generator_workspace
    n = tab.n
    d = tab.d
    a == 0 && return nothing
    if tab.storephase
        prow = 2n + 1
        if d == 2
            cross = MK.dot_xz_ws_vs_col(genws, MK._stacked_views(stab, n, false)[1], col, d)
            stab[prow, col] = mod(stab[prow, col] + genws[prow] + 2 * cross, 4)
        else
            inv2 = tab.inversemod(2, d)
            xdotz_ws = MK.dot_xz_ws(genws, n, d)
            k_power = mod(a * genws[prow] + MK.binom2_mod_oddprime(a, d, inv2) * xdotz_ws, d)
            cross = mod(a * MK.dot_xz_ws_vs_col(genws, MK._stacked_views(stab, n, false)[1], col, d), d)
            stab[prow, col] = mod(stab[prow, col] + k_power + cross, d)
        end
    end
    for r in 1:(2n)
        stab[r, col] = mod(stab[r, col] + a * genws[r], d)
    end
    QuditClifford._load_stabilizer_matrix!(tab, stab)
    return nothing
end

mkref_col_updated!(::StabilizerTableau, ::Int) = nothing
mkref_col_updated!(tab::DestabilizerTableau, col::Int) =
    (tab.xdotz_cache[col] = MK.dot_xz_col(MK._stab(tab), col, tab.d); nothing)

mkref_after_noncommuting!(::StabilizerTableau, ::Int, ::Int, ::Int) = nothing
function mkref_after_noncommuting!(tab::DestabilizerTableau, pivot::Int, comm0::Int, m::Int)
    n = tab.n
    d = tab.d
    inv_comm0 = invmod(comm0, d)
    S = stabilizer_matrix(tab)
    D = destabilizer_matrix(tab)
    for r in 1:(2n)
        D[r, pivot] = mod(inv_comm0 * tab.generator_workspace[r], d)
    end
    for j in 1:m
        j == pivot && continue
        t = 0
        for q in 1:n
            t += D[q, j] * S[n+q, pivot] - D[n+q, j] * S[q, pivot]
        end
        t = mod(t, d)
        t == 0 && continue
        for r in 1:(2n)
            D[r, j] = mod(D[r, j] - t * D[r, pivot], d)
        end
    end
    QuditClifford._load_destabilizer_matrix!(tab, D)
    return nothing
end

# `measure!` with the dense reference kernel in its noncommuting branch.
function mkref_measure!(rng, tab, op::QuditClifford.AbstractPauli; outcome = nothing, phase_policy = 0)
    MK._check_pauli_support(tab, op)
    d = tab.d
    n = tab.n
    m = tab.m
    MK._check_outcome(outcome, d)
    kop = tab.storephase ? MK._effective_measurement_phase(op, d, phase_policy) : op.phase
    kop = mod(kop, MK.phase_modulus(d))
    t = 0
    comm0 = 0
    pivot = 0
    opc = MK._commutation_operand!(tab, op)
    for i in 1:m
        commutator = mod(MK.commutation_col(MK._stab(tab), i, opc), d)
        commutator == 0 && continue
        if comm0 == 0
            t = MK._random_outcome(rng, outcome, d)
            for j in eachindex(tab.generator_workspace)
                tab.generator_workspace[j] = stabilizer_matrix(tab)[j, i]
            end
            MK.set_operator!(tab, i, op)
            if tab.storephase
                S = stabilizer_matrix(tab)
                S[2n+1, i] = MK._measured_generator_phase(t, kop, d)
                QuditClifford._load_stabilizer_matrix!(tab, S)
            end
            comm0 = commutator
            pivot = i
            mkref_col_updated!(tab, i)
        else
            a = mod(-commutator * invmod(comm0, d), d)
            mkref_mul_col!(tab, i, a)
            mkref_col_updated!(tab, i)
        end
    end
    if comm0 != 0
        mkref_after_noncommuting!(tab, pivot, comm0, m)
        tab.iscanonical = false
        return t
    end
    tab.m == n && !tab.storephase && return 0
    MK._span_coeffs_keep_basis!(tab, op)
    in_span = MK._in_span_from_coeffs!(tab, op)
    res = tab.res_workspace
    if in_span
        tab.storephase || return 0
        kacc = MK.phase_exponent_from_coeffs!(tab)
        if d == 2
            k = mod(-kacc + kop, 4)
            isodd(k) && throw(ArgumentError("eigenvalue ±i"))
            return k >> 1
        end
        return mod(-kacc + kop, d)
    end
    (m < n) || throw(ArgumentError("Cannot append generator: tableau at full capacity (m==n)."))
    t = MK._random_outcome(rng, outcome, d)
    newcol = m + 1
    tab.m = newcol
    MK.set_operator!(tab, newcol, op)
    if tab.storephase
        S = stabilizer_matrix(tab)
        S[2n+1, newcol] = MK._measured_generator_phase(t, kop, d)
        QuditClifford._load_stabilizer_matrix!(tab, S)
    end
    MK._after_append_measurement!(tab, newcol, res)
    tab.iscanonical = false
    return t
end

function mkref_same_state(a, b)
    a.m == b.m || return false
    stabilizer_matrix(a) == stabilizer_matrix(b) || return false
    a.iscanonical == b.iscanonical || return false
    a.xdotz_cache == b.xdotz_cache || return false
    a isa DestabilizerTableau && (destabilizer_matrix(a) == destabilizer_matrix(b) || return false)
    return true
end

mkref_exp(rng, d) = rand(rng, -2d:2d)

function mkref_random_op(rng, d, n)
    kind = rand(rng, 1:5)
    ph = rand(rng, -5:5)
    if kind == 2 && n >= 2
        q1, q2 = randperm(rng, n)[1:2]
        rand(rng) < 0.5 && (q2 = mod1(q1 + 1, n))
        q1 == q2 && (q2 = mod1(q1 + 1, n))
        return DoublePauli(q1, mkref_exp(rng, d), mkref_exp(rng, d),
                           q2, mkref_exp(rng, d), mkref_exp(rng, d), ph)
    elseif kind == 3 && n >= 3
        q = randperm(rng, n)[1:3]
        return TriplePauli(q[1], mkref_exp(rng, d), mkref_exp(rng, d),
                           q[2], mkref_exp(rng, d), mkref_exp(rng, d),
                           q[3], mkref_exp(rng, d), mkref_exp(rng, d), ph)
    elseif kind == 4 && n >= 4
        q = Tuple(randperm(rng, n)[1:4])
        return NPauli(q, ntuple(_ -> mkref_exp(rng, d), 4), ntuple(_ -> mkref_exp(rng, d), 4), ph)
    elseif kind == 5
        xz = zeros(Int, 2n)
        # Local or dense support, so both the sparse and the dense pivot path see it.
        sites = rand(rng) < 0.5 ? randperm(rng, n)[1:min(n, 2)] : collect(1:n)
        for q in sites
            xz[q] = mkref_exp(rng, d)
            xz[n+q] = mkref_exp(rng, d)
        end
        return GeneralPauli(xz, ph)
    end
    q = rand(rng, 1:n)
    # Plain Z and X dominate in monitored circuits.
    r = rand(rng)
    r < 0.3 && return SinglePauli(q, 0, 1)
    r < 0.5 && return SinglePauli(q, 1, 0)
    return SinglePauli(q, mkref_exp(rng, d), mkref_exp(rng, d), ph)
end

# A state to measure on. `:local` stays near a product state, so pivots have
# small supports (the restricted path); `:random` has dense pivots.
function mkref_state(rng, TT, d, n, flavor, storephase)
    init = flavor === :mixed ? :mixed : :Z
    tab = TT(d, n; state = init, storephase = storephase)
    if flavor === :random
        random_state!(rng, tab)
    elseif flavor === :local
        for _ in 1:2n
            q = rand(rng, 1:n)
            sup = n >= 2 && rand(rng) < 0.7 ? [q, mod1(q + 1, n)] : [q]
            apply!(tab, random_clifford(rng, d, sup))
        end
    elseif flavor === :mixed
        for _ in 1:(n ÷ 2)
            measure!(rng, tab, mkref_random_op(rng, d, n); phase_policy = 1)
        end
    end
    return tab
end

# Run `nsteps` measurements through both kernels from identical states and
# RNGs; compare everything observable after each one, then continue from the
# package's state. Mixes in local gates so generators keep changing.
function mkref_fuzz(TT, d, n, flavor, storephase, nsteps, seed)
    rng = Xoshiro(seed)
    tab = mkref_state(rng, TT, d, n, flavor, storephase)
    ok = true
    sparse_hits = 0
    dense_hits = 0
    for step in 1:nsteps
        op = mkref_random_op(rng, d, n)
        policy = rand(rng, 0:2)
        out = rand(rng) < 0.5 ? nothing : rand(rng, 0:(d - 1))
        # Which pivot path a noncommuting update with a second anticommuting
        # generator takes (the pivot alone never reaches the update).
        if tab.m > 0 && all(q -> 1 <= q <= n, MK._few_qudit_pauli_data(op isa GeneralPauli ? SinglePauli(1, 0, 0) : op)[1])
            opc = MK._commutation_operand!(tab, op)
            anti = [i for i in 1:tab.m if mod(MK.commutation_col(MK._stab(tab), i, opc), d) != 0]
            if length(anti) >= 2
                p = anti[1]
                nsupp = count(q -> stabilizer_matrix(tab)[q, p] != 0 || stabilizer_matrix(tab)[n+q, p] != 0, 1:n)
                MK._pivot_support_is_sparse(nsupp, n) ? (sparse_hits += 1) : (dense_hits += 1)
            end
        end
        a = copy(tab)
        ra = Xoshiro(step + seed)
        rb = copy(ra)
        resa = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            try
                mkref_measure!(ra, a, op; outcome = out, phase_policy = policy)
            catch e
                e isa ArgumentError || rethrow()
                :threw
            end
        end
        resb = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            try
                measure!(rb, tab, op; outcome = out, phase_policy = policy)
            catch e
                e isa ArgumentError || rethrow()
                :threw
            end
        end
        same = resa == resb && mkref_same_state(a, tab) && rand(ra, UInt64) == rand(rb, UInt64)
        if !same
            @info "mismatch" TT d n flavor storephase seed step op policy out resa resb
            ok = false
            break
        end
        if rand(rng) < 0.3 && n >= 2
            q = rand(rng, 1:n)
            apply!(tab, random_clifford(rng, d, [q, mod1(q + 1, n)]))
        end
    end
    return ok, sparse_hits, dense_hits
end

@testset "measure! matches the dense reference kernel" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5, 31),
        storephase in (true, false)
        @testset "$(nameof(TT)) d=$d storephase=$storephase" begin
            sparse_total = 0
            dense_total = 0
            for (n, flavor) in ((1, :random), (2, :mixed), (3, :random), (5, :local),
                                (8, :mixed), (12, :random), (24, :local), (40, :local),
                                (40, :mixed), (33, :random))
                for seed in 1:(n <= 12 ? 6 : 2)
                    ok, s, dn = mkref_fuzz(TT, d, n, flavor, storephase, n <= 12 ? 30 : 60,
                                           1000 * d + 17 * n + seed)
                    @test ok
                    sparse_total += s
                    dense_total += dn
                end
            end
            # Both pivot paths ran.
            @test sparse_total > 0
            @test dense_total > 0
        end
    end
end

@testset "Site-restricted generator update equals the dense one" begin
    # Feed the same pivot and column through both paths of
    # `_absorb_pivot_power!` and compare phase, XZ rows and cache.
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5, 31, 131),
        storephase in (true, false)
        rng = Xoshiro(d + 7)
        n = 16
        tab = TT(d, n; state = :Z, storephase = storephase)
        random_state!(rng, tab)
        # A pivot with a local support, so both paths are valid.
        piv = 1
        S = stabilizer_matrix(tab)
        S[:, piv] .= 0
        S[3, piv] = rand(rng, 1:(d - 1))
        S[n+3, piv] = rand(rng, 0:(d - 1))
        S[n+7, piv] = rand(rng, 1:(d - 1))
        storephase && (S[2n+1, piv] = rand(rng, 0:(MK.phase_modulus(d) - 1)))
        QuditClifford._load_stabilizer_matrix!(tab, S)
        pu = MK._prepare_pivot_update(tab, piv, rand(rng, 1:(d - 1)))
        @test pu.nsupp == 2
        pd = MK._PivotUpdate(pu.nsupp, false, pu.ninv, pu.kws, pu.xzws, pu.inv2)
        ps = MK._PivotUpdate(pu.nsupp, true, pu.ninv, pu.kws, pu.xzws, pu.inv2)
        for col in 2:n, a in (d == 2 ? (1,) : (1, 2, d - 1))
            TT === DestabilizerTableau && (tab.xdotz_cache[col] = MK.dot_xz_col(MK._stab(tab), col, d))
            before = stabilizer_matrix(tab)[:, col]
            MK._absorb_pivot_power!(tab, col, a, pd)
            dense = stabilizer_matrix(tab)[:, col]
            cdense = tab.xdotz_cache[col]
            S = stabilizer_matrix(tab)
            S[:, col] .= before
            QuditClifford._load_stabilizer_matrix!(tab, S)
            TT === DestabilizerTableau && (tab.xdotz_cache[col] = MK.dot_xz_col(MK._stab(tab), col, d))
            MK._absorb_pivot_power!(tab, col, a, ps)
            @test stabilizer_matrix(tab)[:, col] == dense
            @test tab.xdotz_cache[col] == cdense
            if TT === DestabilizerTableau
                @test cdense == MK.dot_xz_col(MK._stab(tab), col, d)
            end
            S = stabilizer_matrix(tab)
            S[:, col] .= before
            QuditClifford._load_stabilizer_matrix!(tab, S)
        end
    end
end

function mkref_measure_allocs(rng, tab, op)
    warm = copy(tab)
    t = copy(tab)
    measure!(rng, warm, op; outcome = 0, phase_policy = 1)
    return @allocated measure!(rng, t, op; outcome = 0, phase_policy = 1)
end

@testset "measure! allocates nothing in the noncommuting branch" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31),
        E in (UInt8, UInt16, Int)
        n = 64
        tab = TT(d, n; state = :Z, eltype = E)
        rng = Xoshiro(3)
        for q in 1:2:(n - 1)
            apply!(tab, random_clifford(rng, d, [q, q + 1]))
        end
        ops = (SinglePauli(5, 1, 0), DoublePauli(9, 1, 0, 10, 1, 0),
               TriplePauli(1, 1, 0, 2, 1, 0, 3, 1, 1), NPauli((4, 6, 8, 11), (1, 1, 0, 1), (0, 1, 1, 0)))
        for op in ops
            @test mkref_measure_allocs(rng, tab, op) == 0
        end
        # A random state takes the dense pivot path.
        r = TT(d, n; state = :Z, eltype = E)
        random_state!(rng, r)
        @test mkref_measure_allocs(rng, r, SinglePauli(5, 1, 0)) == 0
    end
end
