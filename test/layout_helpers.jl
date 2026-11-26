# Helpers for the tests of the storage layouts, shared by test/layout.jl,
# test/gate_kernel_stabilizer.jl and test/gate_kernel_destabilizer.jl. Each of
# them includes this file unless another has done so.
using QuditClifford, Test, Random
include(joinpath(@__DIR__, "env_switch.jl"))

const ly_QC = QuditClifford

# The storage element types a tableau at dimension `d` admits.
ly_eltypes(d) = filter(E -> ly_QC._eltype_admissible(E, d), (UInt8, UInt16, Int))

ly_matrices(tab::StabilizerTableau) = (stabilizer_matrix(tab),)
ly_matrices(tab::DestabilizerTableau) = (stabilizer_matrix(tab), destabilizer_matrix(tab))

# Entry-wise equality of the generators, phases and caches of a generator-major
# and a qudit-major tableau, without building the stacked matrices.
function ly_same_state(g, q)
    G, Q = ly_QC.GeneratorMajor(), ly_QC.QuditMajor()
    g.m == q.m || return false
    g.phase == q.phase || return false
    g.xdotz_cache == q.xdotz_cache || return false
    pairs = g isa DestabilizerTableau ?
        ((ly_QC._stab(g, G), ly_QC._stab(q, Q)), (ly_QC._destab(g, G), ly_QC._destab(q, Q))) :
        ((ly_QC._stab(g, G), ly_QC._stab(q, Q)),)
    for (bg, bq) in pairs, j in 1:g.n, t in 1:g.n
        ly_QC._xget(bg, t, j) == ly_QC._xget(bq, t, j) || return false
        ly_QC._zget(bg, t, j) == ly_QC._zget(bq, t, j) || return false
    end
    return true
end

# Gate `i` of a seeded stream on `n` qudits at dimension `d`. Gates 8, 26, 44, …
# are stored operators on four qudits when `n ≥ 4`, gates 17, 35, 53, … on five
# when `n ≥ 5`, and gates 9, 27, 45, … on six (the dense backing) when `n ≥ 6`.
function ly_stream_gate(rng, d, n, i)
    kmax = min(n, 3)
    tg = randperm(rng, n)
    r = rand(rng, 1:d - 1)
    kind = mod1(i, 9)
    if n == 1 && kind in (1, 2, 3)
        kind = 4
    end
    kind == 1 && return SUM(tg[1], tg[2], r)
    kind == 2 && return CPhase(tg[1], tg[2], r)
    kind == 3 && return SWAP(tg[1], tg[2])
    kind == 4 && return Fourier(tg[1])
    kind == 5 && return Phase(tg[1])
    kind == 6 && return Multiplier(tg[1], r)
    kind == 7 && return PauliGate(tg[1], rand(rng, 0:d - 1), rand(rng, 0:d - 1))
    k = i % 18 == 9 && n >= 6 ? 6 :
        i % 18 == 17 && n >= 5 ? 5 :
        i % 18 == 8 && n >= 4 ? 4 : rand(rng, 1:kmax)
    return random_clifford(rng, d, tg[1:k])
end

function ly_check_gate_stream(TT, d, E, n, storephase, seed)
    rng = Xoshiro(seed)
    g = TT(d, n; eltype = E, storephase, layout = :generator)
    random_state!(rng, g; m = max(1, n - 2))
    q = qudit_major!(copy(g))
    ok = ly_same_state(g, q)
    for i in 1:200
        gate = ly_stream_gate(rng, d, n, i)
        if i % 25 == 0
            prep = ly_QC._prepare(gate, d, g.inversemod, storephase; force_safe = true)
            ly_QC._apply_prepared!(g, prep)
            ly_QC._apply_prepared!(q, prep)
        else
            apply!(g, gate)
            apply!(q, gate)
        end
        # The entry-wise comparison costs far more than the gate at large `n`,
        # so there it runs on every twentieth gate. Gates are bijections on
        # tableaux, so a difference persists until the next comparison; only
        # the index of the offending gate is lost.
        if n <= 64 || i % 20 == 0
            ok &= ly_same_state(g, q)
            ok || (@error "layouts differ" TT d E n storephase i gate; break)
        end
    end
    ok &= ly_matrices(g) == ly_matrices(q)
    return ok
end

# The cases `(TT, d, E, n, storephase, seed)` of the qudit-major gate-kernel
# tests, each with its own seed. The two tableau types run in separate files so
# that CI can put them in different shards.
function ly_gate_kernel_cases()
    # Lane boundaries of "lane selection" (test/layout.jl): the largest Int16 d
    # for S = 6, 4, 2 (47, 61, 89), the first Int32 d (53, 67, 97), Int32
    # lanes with `UInt16` storage (4093) and Int64 lanes (13399 for S = 6, 23189 for S = 2). Each
    # lane runs with its own signed constant (d = 3; 101 and 61 at S = 6 in
    # Int32) and with the unsigned one (d = 5 at S = 4, 6; 43-97, 251, 4093,
    # and 11777, which has no signed constant at S = 6). For the stored
    # operators on four and five qudits (S = 8, 10; qudit-major tuple backing
    # at n = 64 and 300, dense backing at n = 5): the largest Int16 d at S = 10
    # (37) and the first Int32 d (41), Int32 lanes with their own signed
    # constant (43 at S = 10, 61 and 109 at S = 8), an Int64 lane at S = 10
    # (10369, Int32 at S = 8) and a d without a lane at S = 8 (11587).
    ds = [2, 3, 5, 37, 41, 43, 47, 53, 61, 67, 89, 97, 101, 109, 251, 257, 4093,
          10369, 11587, 11777, 13399, 23189]
    Sys.WORD_SIZE == 64 && push!(ds, 65521)
    cases = Tuple{Type, Int, Type, Int, Bool, Int}[]
    seed = 1
    for TT in (StabilizerTableau, DestabilizerTableau), storephase in (true, false),
        d in ds, E in ly_eltypes(d), n in (5, 64, 300)
        seed += 1
        push!(cases, (TT, d, E, n, storephase, seed))
    end
    return cases
end
