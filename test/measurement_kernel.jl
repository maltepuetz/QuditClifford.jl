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
    (tab.xdotz_cache[col] = MK.dot_xz_col(MK._stab(tab, MK._layout_marker(tab)), col, tab.d); nothing)

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
        commutator = mod(MK.commutation_col(MK._stab(tab, MK._layout_marker(tab)), i, opc), d)
        commutator == 0 && continue
        if comm0 == 0
            t = MK._random_outcome(rng, outcome, d)
            for j in eachindex(tab.generator_workspace)
                tab.generator_workspace[j] = stabilizer_matrix(tab)[j, i]
            end
            MK.set_operator!(tab, MK._layout_marker(tab), i, op)
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
    MK._span_coeffs_keep_basis!(tab, MK._layout_marker(tab), op)
    in_span = MK._in_span_from_coeffs!(tab, MK._layout_marker(tab), op)
    res = tab.res_workspace
    if in_span
        tab.storephase || return 0
        kacc = MK.phase_exponent_from_coeffs!(tab, MK._layout_marker(tab))
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
    MK.set_operator!(tab, MK._layout_marker(tab), newcol, op)
    if tab.storephase
        S = stabilizer_matrix(tab)
        S[2n+1, newcol] = MK._measured_generator_phase(t, kop, d)
        QuditClifford._load_stabilizer_matrix!(tab, S)
    end
    MK._after_append_measurement!(tab, MK._layout_marker(tab), newcol, res)
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
            anti = [i for i in 1:tab.m if mod(MK.commutation_col(MK._stab(tab, MK._layout_marker(tab)), i, opc), d) != 0]
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
        # Unpacked storage: a packed tableau never reaches this function.
        tab = TT(d, n; state = :Z, storephase = storephase, eltype = UInt8)
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
        pu = MK._prepare_pivot_update(tab, MK._layout_marker(tab), piv, rand(rng, 1:(d - 1)))
        @test pu.nsupp == 2
        pd = MK._PivotUpdate(pu.nsupp, false, pu.ninv, pu.kws, pu.xzws, pu.inv2)
        ps = MK._PivotUpdate(pu.nsupp, true, pu.ninv, pu.kws, pu.xzws, pu.inv2)
        for col in 2:n, a in (d == 2 ? (1,) : (1, 2, d - 1))
            TT === DestabilizerTableau && (tab.xdotz_cache[col] = MK.dot_xz_col(MK._stab(tab, MK._layout_marker(tab)), col, d))
            before = stabilizer_matrix(tab)[:, col]
            MK._absorb_pivot_power!(tab, MK._layout_marker(tab), col, a, pd)
            dense = stabilizer_matrix(tab)[:, col]
            cdense = tab.xdotz_cache[col]
            S = stabilizer_matrix(tab)
            S[:, col] .= before
            QuditClifford._load_stabilizer_matrix!(tab, S)
            TT === DestabilizerTableau && (tab.xdotz_cache[col] = MK.dot_xz_col(MK._stab(tab, MK._layout_marker(tab)), col, d))
            MK._absorb_pivot_power!(tab, MK._layout_marker(tab), col, a, ps)
            @test stabilizer_matrix(tab)[:, col] == dense
            @test tab.xdotz_cache[col] == cdense
            if TT === DestabilizerTableau
                @test cdense == MK.dot_xz_col(MK._stab(tab, MK._layout_marker(tab)), col, d)
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
        E in filter(E -> QuditClifford._eltype_admissible(E, d), (UInt8, UInt16, Int, Bits)),
        layout in (:generator, :qudit)
        n = 64
        tab = TT(d, n; state = :Z, eltype = E, layout)
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
        r = TT(d, n; state = :Z, eltype = E, layout)
        random_state!(rng, r)
        @test mkref_measure_allocs(rng, r, SinglePauli(5, 1, 0)) == 0
        @test mkref_measure_allocs(rng, r, GeneralPauli(rand(rng, 0:(d - 1), 2n), 0)) == 0
    end
end

@testset "packed qudit-major measure! allocates nothing on the sparse path" begin
    # At n = 384 the mask of generators 1:n has six words, so one nonzero word
    # is below the default cutoff `(6 - 1) ÷ _BIT_QM_SPARSE_DIV`.
    n = 384
    @test (MK._nwords(n) - 1) ÷ MK._BIT_QM_SPARSE_DIV >= 1
    for TT in (StabilizerTableau, DestabilizerTableau)
        # X5 X6 anticommutes with Z5 (the pivot) and Z6 only: one mask word.
        tab = TT(2, n; state = :Z, eltype = Bits, layout = :qudit)
        op = DoublePauli(5, 1, 0, 6, 1, 0)
        @test mkref_measure_allocs(Xoshiro(1), tab, op) == 0
        a, b = copy(tab), copy(tab)
        L = MK.QuditMajor()
        @test MK._measure_bits_qm!(Xoshiro(2), a, L, op, nothing, 1, MK._BIT_QM_SPARSE_DIV) ==
              MK._measure_bits_qm!(Xoshiro(2), b, L, op, nothing, 1, typemax(Int))
        @test a == b
    end
end

@testset "measure! allocates nothing in the append branch of a DestabilizerTableau" begin
    for d in (2, 3, 31),
        E in filter(E -> QuditClifford._eltype_admissible(E, d), (UInt8, UInt16, Int, Bits)),
        layout in (:generator, :qudit)
        n = 64
        h = n ÷ 2
        tab = DestabilizerTableau(d, n; state = :mixed, eltype = E, layout)
        for q in 1:h
            measure!(tab, SinglePauli(q, 0, 1); outcome = 0)
        end
        # Generators and duals spread over the first half; Z_n commutes with
        # them and lies outside their span.
        apply!(tab, random_clifford(Xoshiro(d), d, collect(1:h)))
        @test tab.m == h
        @test mkref_measure_allocs(Xoshiro(1), tab, SinglePauli(n, 0, 1)) == 0
        c = copy(tab)
        measure!(c, SinglePauli(n, 0, 1); outcome = 0)
        @test c.m == h + 1
    end
end

# Random-branch fixtures: brickwork circuits with Z measurements at rate `p`
# on a pure state, and a random pure state, in the default element type for
# `d` or in `eltype`.
function mkqm_monitored(TT, d, n, p, storephase, seed; eltype = MK._default_eltype(d))
    rng = Xoshiro(seed)
    tab = TT(d, n; state = :Z, storephase, eltype)
    for l in 1:8
        for q in (isodd(l) ? 1 : 2):2:(n - 1)
            apply!(tab, random_clifford(rng, d, [q, q + 1]))
        end
        for q in 1:n
            rand(rng) < p && measure!(rng, tab, SinglePauli(q, 0, 1))
        end
    end
    return tab
end

function mkqm_random(TT, d, n, storephase, seed; eltype = MK._default_eltype(d))
    tab = TT(d, n; state = :Z, storephase, eltype)
    random_state!(Xoshiro(seed), tab)
    return tab
end

# Measure Z, X and a two-qudit Pauli on random qudits of a generator-major
# fixture and its qudit-major copy; compare after every measurement.
function mkqm_compare(g, nsteps, seed)
    q = qudit_major!(copy(g))
    rng = Xoshiro(seed)
    rg = Xoshiro(seed + 1)
    rq = Xoshiro(seed + 1)
    n, d = g.n, g.d
    for step in 1:nsteps
        a, b = randperm(rng, n)[1:2]
        r = rand(rng)
        op = r < 0.6 ? SinglePauli(a, 0, 1) : r < 0.8 ? SinglePauli(a, 1, 0) :
             DoublePauli(a, rand(rng, 0:(d - 1)), 1, b, 1, rand(rng, 0:(d - 1)))
        tg = measure!(rg, g, op; phase_policy = 1)
        tq = measure!(rq, q, op; phase_policy = 1)
        same = tg == tq && mkref_same_state(g, q) && g.phase == q.phase
        same || (@info "layouts differ" typeof(g) d n step op; return false)
    end
    return rand(rg, UInt64) == rand(rq, UInt64)
end

@testset "qudit-major random branch" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31), storephase in (false, true)
        n = 96
        for (k, p) in enumerate((0.05, 0.15))
            @test mkqm_compare(mkqm_monitored(TT, d, n, p, storephase, 10d + k), 60, d + k)
        end
        @test mkqm_compare(mkqm_random(TT, d, n, storephase, 20d), 60, d + 7)
    end
end

@testset "qudit-major sparse and dense updates agree" begin
    # Divisor 1 takes the sparse update whenever a generator anticommutes;
    # `typemax(Int)` takes the dense one whenever one does.
    L = MK.QuditMajor()
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31), storephase in (false, true)
        n = 48
        # The qudit-major kernel takes unpacked storage.
        E = d == 2 ? UInt8 : MK._default_eltype(d)
        for (k, g) in enumerate((mkqm_monitored(TT, d, n, 0.15, storephase, 30d; eltype = E),
                                 mkqm_random(TT, d, n, storephase, 40d; eltype = E)))
            a = qudit_major!(copy(g))
            b = copy(a)
            rng = Xoshiro(k)
            ok = true
            for step in 1:40
                q = rand(rng, 1:n)
                op = rand(rng) < 0.6 ? SinglePauli(q, 0, 1) : SinglePauli(q, 1, rand(rng, 0:(d - 1)))
                ta = MK._measure_qm!(Xoshiro(step), a, L, op, nothing, 1, 1)
                tb = MK._measure_qm!(Xoshiro(step), b, L, op, nothing, 1, typemax(Int))
                ok &= ta == tb && mkref_same_state(a, b) && a.phase == b.phase
                ok || break
            end
            @test ok
        end
    end
end

# `st_padding_zero` lives in test/storage_helpers.jl.
isdefined(Main, :st_padding_zero) || include(joinpath(@__DIR__, "storage_helpers.jl"))

# A random-branch fixture in `Bits` and in `UInt8`, both in `layout`; the
# fixtures evolve through the same operations and must agree before the test.
function mkbit_pair(fixture, layout)
    t = fixture(Bits)
    u = fixture(UInt8)
    sw = layout === :qudit ? qudit_major! : generator_major!
    return sw(t), sw(u)
end

# Measure Z, X and a two-qubit Pauli on random qudits of the packed `t` and the
# `UInt8` `u`; compare after every measurement.
function mkbit_compare(t, u, nsteps, seed)
    mkref_same_state(t, u) && t.phase == u.phase || return false
    rng = Xoshiro(seed)
    rt = Xoshiro(seed + 1)
    ru = Xoshiro(seed + 1)
    n = t.n
    for step in 1:nsteps
        a, b = randperm(rng, n)[1:2]
        r = rand(rng)
        op = r < 0.6 ? SinglePauli(a, 0, 1) : r < 0.8 ? SinglePauli(a, 1, 0) :
             DoublePauli(a, rand(rng, 0:1), 1, b, 1, rand(rng, 0:1))
        tt = measure!(rt, t, op; phase_policy = 1)
        tu = measure!(ru, u, op; phase_policy = 1)
        same = tt == tu && mkref_same_state(t, u) && t.phase == u.phase && st_padding_zero(t)
        same || (@info "Bits and UInt8 differ" typeof(t) tableau_layout(t) n step op; return false)
    end
    return rand(rt, UInt64) == rand(ru, UInt64)
end

@testset "bit random branch" begin
    for TT in (StabilizerTableau, DestabilizerTableau), storephase in (false, true),
        layout in (:generator, :qudit)
        n = 96
        for (k, p) in enumerate((0.05, 0.15))
            t, u = mkbit_pair(E -> mkqm_monitored(TT, 2, n, p, storephase, 20 + k; eltype = E), layout)
            @test mkbit_compare(t, u, 60, 2 + k)
        end
        t, u = mkbit_pair(E -> mkqm_random(TT, 2, n, storephase, 40; eltype = E), layout)
        @test mkbit_compare(t, u, 60, 9)
        for (TL, L) in ((:generator, MK.GeneratorMajor()), (:qudit, MK.QuditMajor()))
            s = TT(2, n; storephase, eltype = Bits, layout = TL)
            op = SinglePauli(1, 0, 1)
            m = which(MK._measure!, (Xoshiro, typeof(s), typeof(L), typeof(op), Nothing, Int))
            @test basename(String(m.file)) == "measure_bits.jl"
            m = which(MK._dephase!, (typeof(s), typeof(L), typeof(op)))
            @test basename(String(m.file)) == "measure_bits.jl"
        end
    end
end

@testset "bit qudit-major sparse and dense updates agree" begin
    # Divisor 0 takes the sparse update whenever a generator anticommutes;
    # `typemax(Int)` takes the dense one whenever one does.
    L = MK.QuditMajor()
    for TT in (StabilizerTableau, DestabilizerTableau), storephase in (false, true)
        n = 200
        for (k, fixture) in enumerate((E -> mkqm_monitored(TT, 2, n, 0.15, storephase, 60; eltype = E),
                                       E -> mkqm_random(TT, 2, n, storephase, 61; eltype = E)))
            a, u = mkbit_pair(fixture, :qudit)
            b = copy(a)
            rng = Xoshiro(k)
            ok = true
            for step in 1:40
                q = rand(rng, 1:n)
                op = rand(rng) < 0.6 ? SinglePauli(q, 0, 1) : SinglePauli(q, 1, rand(rng, 0:1))
                ta = MK._measure_bits_qm!(Xoshiro(step), a, L, op, nothing, 1, 0)
                tb = MK._measure_bits_qm!(Xoshiro(step), b, L, op, nothing, 1, typemax(Int))
                tu = MK._measure_qm!(Xoshiro(step), u, L, op, nothing, 1, MK._QM_SPARSE_UPDATE_DIV)
                ok &= ta == tb == tu && mkref_same_state(a, b) && mkref_same_state(a, u) &&
                      a.phase == b.phase == u.phase
                ok || break
            end
            @test ok
        end
    end
end
