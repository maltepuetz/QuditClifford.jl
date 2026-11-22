# Storage benchmark: times every public tableau operation for each layout and
# element type, so that runs on different machines decide the default layout per
# `d` and the element type at `d = 2`. Uses only QuditClifford and stdlibs.
#
#     julia --project=. benchmark/storage/run.jl out.csv
#
# Grid (environment variables, comma lists):
#   QC_BENCH_PROFILE  `default` (under 2 h on one core) or `quick` (under 10 min);
#                     sets the defaults of the variables below
#   QC_BENCH_LAYOUTS  `generator,qudit`
#   QC_BENCH_ELTYPES  `default,Bits`: the default element type of each `d`, and
#                     packed bits (used at `d = 2` only)
#   QC_BENCH_DS       `2,3,5,31`
#   QC_BENCH_NS       sizes, see `PROFILES`
#   QC_BENCH_TYPES    `Stab,Destab`
#   QC_BENCH_OPS      the row labels to run (all by default, see `ALL_OPS`)
#   QC_BENCH_BUDGET   seconds per row; sets the repetition count between the
#                     minimum and the maximum of the row
#
# Output: comment lines (`#`) describing the machine and the commit, then
# `label,layout,eltype,tableau,d,n,median_ns,reps`, one line per row: the median
# time per operation over `reps` repetitions, each on a fresh state (making the
# state is not timed). Summarize with `benchmark/storage/summarize.jl`.

using QuditClifford, Random, Printf, Dates, InteractiveUtils

# `randstate_nmax`: `random_state!` is O(n^3) (about 2 s at n = 1024) and the
# same for every layout and element type, so it runs only up to this size.
const PROFILES = Dict(
    "default" => (ns = "512,1024,2048,4096", budget = "1.0", randstate_nmax = 1100),
    "quick" => (ns = "256,1024", budget = "0.15", randstate_nmax = 300))

const PROFILE = get(ENV, "QC_BENCH_PROFILE", "default")
haskey(PROFILES, PROFILE) || error("QC_BENCH_PROFILE must be one of $(join(keys(PROFILES), ", "))")
const P = PROFILES[PROFILE]

envlist(k, default) = String.(split(get(ENV, k, default), ','; keepempty = false))

# Row labels. `switch` is a layout round trip; `ising` is the measurement-only
# Ising chain with `entanglement_profile` after each layer; `monitored_p05` and
# `monitored_p15` are brickwork layers of random two-qudit gates with Z
# measurements at rate p; `brickwork` is the same layers without measurements.
const ALL_OPS = ("gate1", "gate2", "apply_dense4", "brickwork", "monitored_p05", "monitored_p15",
                 "measZ_random", "measZ_det", "measZ_append", "measZZ_random", "measX_random", "ising",
                 "random_state", "switch", "canonicalize", "expect", "expect_int", "dephase",
                 "traceout", "reset_sites", "entropy_half", "entropy_profile", "mutual_info", "cmi",
                 "tmi", "negativity", "is_pure_verify", "copy!", "==", "stabilizer_matrix")

const LAYOUTS = Symbol.(envlist("QC_BENCH_LAYOUTS", "generator,qudit"))
const ELTYPES = envlist("QC_BENCH_ELTYPES", "default,Bits")
const DS = parse.(Int, envlist("QC_BENCH_DS", "2,3,5,31"))
const NS = parse.(Int, envlist("QC_BENCH_NS", P.ns))
const TYPES = filter(t -> t[2] in envlist("QC_BENCH_TYPES", "Stab,Destab"),
                     ((StabilizerTableau, "Stab"), (DestabilizerTableau, "Destab")))
const OPS = envlist("QC_BENCH_OPS", join(ALL_OPS, ','))
const BUDGET_NS = 1e9 * parse(Float64, get(ENV, "QC_BENCH_BUDGET", P.budget))
foreach(op -> op in ALL_OPS || error("unknown row $op; rows: $(join(ALL_OPS, ", "))"), OPS)
foreach(l -> l in (:generator, :qudit) || error("unknown layout $l"), LAYOUTS)
foreach(e -> e in ("default", "Bits") || error("QC_BENCH_ELTYPES takes default and Bits, got $e"), ELTYPES)

const OUT = isempty(ARGS) ? "storage_benchmark.csv" : ARGS[1]
const RANDSTATE_NMAX = P.randstate_nmax

function median!(v::Vector{Float64})
    sort!(v)
    k = length(v)
    isodd(k) ? v[(k + 1) ÷ 2] : (v[k ÷ 2] + v[k ÷ 2 + 1]) / 2
end

# Median ns per op over fresh states from `mk` (not timed), and the rep count:
# a first call (also a warm-up) sets it from the budget, within `minreps:maxreps`.
function timeit(f!, mk, nops; maxreps = 41, minreps = 5)
    s = mk()
    t0 = time_ns(); f!(s); t1 = time_ns() - t0
    reps = clamp(round(Int, BUDGET_NS / max(t1, 1)), t1 > 1e9 ? 3 : minreps, maxreps)
    ts = Float64[]
    for _ in 1:reps
        s = mk()
        GC.gc(false)
        t0 = time_ns(); f!(s); push!(ts, (time_ns() - t0) / nops)
    end
    median!(ts), reps
end

zop(q) = SinglePauli(q, 0, 1)
xop(q) = SinglePauli(q, 1, 0)

function bonds_for(rng, d, n)
    [random_clifford(rng, d, [q, q + 1]) for q in 1:(n - 1)]
end

function layers!(tab, bonds, rng, layers, p)
    n, d = tab.n, tab.d
    for l in 1:layers
        for q in (isodd(l) ? 1 : 2):2:(n - 1)
            apply!(tab, random_clifford!(rng, bonds[q]))
        end
        p > 0 || continue
        for q in 1:n
            rand(rng) < p || continue
            measure!(tab, zop(q); outcome = rand(rng, 0:(d - 1)))
        end
    end
    tab
end

# Z_i Z_{i+1}^† on every bond, then X_i on every site (outcomes from `outs` where
# the measurement is random), then the entanglement profile.
function ising_layers!(tab, outs, layers)
    n, d = tab.n, tab.d
    k = 0
    for _ in 1:layers
        for i in 1:(n - 1)
            measure!(tab, DoublePauli(i, 0, 1, i + 1, 0, d - 1); outcome = outs[k += 1])
        end
        for i in 1:n
            measure!(tab, xop(i); outcome = outs[k += 1])
        end
        entanglement_profile(tab)
    end
    tab
end

random_ops(canon, ops, K) = first([op for op in ops if iszero(expect!(canon, op))], K)

function run_case!(io, T, tn, layout, eltype, d, n)
    kw = eltype === nothing ? (; layout) : (; layout, eltype)
    want(op) = op in OPS
    probe = T(d, 2; kw...)
    ename = string(nameof(tableau_eltype(probe)))
    function rec!(op, (v, reps))
        println(io, op, ",", layout, ",", ename, ",", tn, ",", d, ",", n, ",", round(v; digits = 1), ",", reps)
        flush(io)
        @printf("  %-18s %-6s %-9s %-5s d=%-3d n=%-5d %14.1f ns  (%d reps)\n", op, tn, layout, ename, d, n, v, reps)
        flush(stdout)
    end
    rng = Xoshiro(1000 * d + n)

    # gate streams on a product state
    prod = T(d, n; state = :product, kw...)
    N = 512
    g1 = [random_clifford(rng, d, [rand(rng, 1:n)]) for _ in 1:N]
    g2 = [random_clifford(rng, d, randperm(rng, n)[1:2]) for _ in 1:N]
    want("gate1") && rec!("gate1", timeit(t -> foreach(U -> apply!(t, U), g1), () -> copy(prod), N; maxreps = 61))
    want("gate2") && rec!("gate2", timeit(t -> foreach(U -> apply!(t, U), g2), () -> copy(prod), N; maxreps = 61))
    if want("apply_dense4")
        g4 = [random_clifford(rng, d, randperm(rng, n)[1:4]) for _ in 1:64]
        rec!("apply_dense4", timeit(t -> foreach(U -> apply!(t, U), g4), () -> copy(prod), 64))
    end

    # brickwork layers: 8 warm-up layers once, then 8 timed layers on a copy
    # (with a copy of the random stream), per layer
    for (op, p) in (("brickwork", 0.0), ("monitored_p05", 0.05), ("monitored_p15", 0.15))
        want(op) || continue
        r = Xoshiro(1234)
        b = bonds_for(r, d, n)
        warm = layers!(T(d, n; state = :Z, kw...), b, r, 8, p)
        rec!(op, timeit(s -> layers!(s..., 8, p), () -> (copy(warm), b, copy(r)), 8; maxreps = 9))
    end
    # appending measurements: Z on half of the qudits once (m = n/2), then up to
    # 64 more Z measurements timed on a copy, per measurement
    if want("measZ_append")
        order = randperm(rng, n)
        h = n ÷ 2
        half = T(d, n; state = :mixed, kw...)
        foreach(q -> measure!(half, zop(q); outcome = 0), view(order, 1:h))
        timed = view(order, (h + 1):min(n, h + 64))
        rec!("measZ_append", timeit(t -> foreach(q -> measure!(t, zop(q); outcome = 0), timed),
                                    () -> copy(half), length(timed); maxreps = 21))
    end
    # two Ising layers once, then two timed layers on a copy, per layer
    if want("ising")
        outs = rand(Xoshiro(77), 0:(d - 1), 4 * (2n - 1))
        warm = ising_layers!(T(d, n; state = :Z, kw...), outs, 2)
        rec!("ising", timeit(t -> ising_layers!(t, outs, 2), () -> copy(warm), 2; maxreps = 9))
    end
    if want("random_state") && n <= RANDSTATE_NMAX
        rec!("random_state", timeit(t -> random_state!(Xoshiro(5), t), () -> copy(prod), 1))
    end

    # shared state: 64 monitored layers at p = 0.15
    r0 = Xoshiro(4321)
    tab0 = layers!(T(d, n; state = :Z, kw...), bonds_for(r0, d, n), r0, 64, 0.15)
    if want("switch")
        sw! = layout === :qudit ? (t -> (generator_major!(t); qudit_major!(t))) :
                                  (t -> (qudit_major!(t); generator_major!(t)))
        rec!("switch", timeit(sw!, () -> copy(tab0), 1))
    end

    canon = copy(tab0); canonicalize!(canon)
    isdet = [!iszero(expect!(canon, zop(q))) for q in 1:n]
    K = 64
    rseq = [(q, rand(rng, 0:(d - 1))) for q in first(shuffle(rng, findall(!, isdet)), K)]
    dseq = first(shuffle(rng, findall(isdet)), K)
    if want("measZ_random")
        length(rseq) >= 8 || error("too few random Z measurements at d = $d, n = $n")
        rec!("measZ_random", timeit(t -> foreach(((q, o),) -> measure!(t, zop(q); outcome = o), rseq),
                                    () -> copy(tab0), length(rseq)))
    end
    if want("measZ_det")
        length(dseq) >= 8 || error("too few deterministic Z measurements at d = $d, n = $n")
        rec!("measZ_det", timeit(t -> foreach(q -> measure!(t, zop(q); outcome = 0), dseq),
                                 () -> copy(tab0), length(dseq)))
    end
    if want("measZZ_random")
        zz = random_ops(canon, [DoublePauli(i, 0, 1, j, 0, d - 1)
                                for (i, j) in (sort(randperm(rng, n)[1:2]) for _ in 1:4K)], K)
        length(zz) >= 8 || error("too few random ZZ measurements at d = $d, n = $n")
        rec!("measZZ_random", timeit(t -> foreach(op -> measure!(t, op; outcome = 1), zz),
                                     () -> copy(tab0), length(zz)))
    end
    if want("measX_random")
        xs = random_ops(canon, [xop(q) for q in shuffle(rng, 1:n)], K)
        length(xs) >= 8 || error("too few random X measurements at d = $d, n = $n")
        rec!("measX_random", timeit(t -> foreach(op -> measure!(t, op; outcome = 1), xs),
                                    () -> copy(tab0), length(xs)))
    end

    want("canonicalize") && rec!("canonicalize", timeit(canonicalize!, () -> copy(tab0), 1; maxreps = 21))
    zzs = [DoublePauli(i, 0, 1, j, 0, 1) for (i, j) in (sort(randperm(rng, n)[1:2]) for _ in 1:32)]
    want("expect") && rec!("expect", timeit(t -> foreach(op -> expect!(t, op), zzs), () -> copy(canon), length(zzs)))
    # on a fresh copy of the non-canonical state: a StabilizerTableau canonicalizes in the first call
    want("expect_int") && rec!("expect_int", timeit(t -> foreach(op -> expect_int!(t, op), zzs),
                                                    () -> copy(tab0), length(zzs)))
    if want("dephase")
        dq = [xop(q) for q in first(shuffle(rng, 1:n), 16)]
        rec!("dephase", timeit(t -> foreach(op -> dephase!(t, op), dq), () -> copy(tab0), length(dq)))
    end
    sub8 = sort(first(shuffle(rng, 1:n), n ÷ 8))
    want("traceout") && rec!("traceout", timeit(t -> traceout!(t, sub8), () -> copy(tab0), 1))
    want("reset_sites") && rec!("reset_sites", timeit(t -> reset!(t, sub8), () -> copy(tab0), 1))

    q4 = n ÷ 4
    A, B, C = 1:q4, (q4 + 1):(2q4), (2q4 + 1):(3q4)
    want("entropy_half") && rec!("entropy_half", timeit(t -> entanglement_entropy(t, 1:(n ÷ 2)), () -> tab0, 1))
    want("entropy_profile") && rec!("entropy_profile", timeit(entanglement_profile, () -> tab0, 1))
    want("mutual_info") && rec!("mutual_info", timeit(t -> mutual_information(t, A, C), () -> tab0, 1))
    want("cmi") && rec!("cmi", timeit(t -> conditional_mutual_information(t, A, C, B), () -> tab0, 1))
    want("tmi") && rec!("tmi", timeit(t -> tripartite_information(t, A, B, C), () -> tab0, 1))
    want("negativity") && rec!("negativity", timeit(t -> entanglement_negativity(t, A, B), () -> tab0, 1))
    want("is_pure_verify") && rec!("is_pure_verify", timeit(t -> is_pure(t; verify = true), () -> tab0, 1))
    if want("copy!")
        dst = copy(tab0)
        rec!("copy!", timeit(t -> copy!(dst, t), () -> tab0, 1))
    end
    if want("==")
        other = copy(tab0)
        rec!("==", timeit(t -> t == other, () -> tab0, 1))
    end
    want("stabilizer_matrix") && rec!("stabilizer_matrix", timeit(stabilizer_matrix, () -> tab0, 1))
end

function git_commit()
    try
        dir = dirname(dirname(pathof(QuditClifford)))
        c = readchomp(pipeline(`git -C $dir rev-parse HEAD`; stderr = devnull))
        dirty = !isempty(readchomp(pipeline(`git -C $dir status --porcelain --untracked-files=no`; stderr = devnull)))
        dirty ? c * " (modified)" : c
    catch
        "unknown"
    end
end

function header(io)
    println(io, "# QuditClifford storage benchmark, profile ", PROFILE)
    println(io, "# timestamp: ", Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS"), "Z")
    println(io, "# commit: ", git_commit())
    println(io, "# package: ", pathof(QuditClifford))
    println(io, "# cpu_name: ", Sys.CPU_NAME)
    println(io, "# cpu_model: ", strip(Sys.cpu_info()[1].model))
    println(io, "# threads: ", Threads.nthreads(), " (cores visible: ", Sys.CPU_THREADS, ")")
    println(io, "# grid: layouts=", join(LAYOUTS, ' '), " eltypes=", join(ELTYPES, ' '), " ds=", join(DS, ' '),
            " ns=", join(NS, ' '), " budget_s=", BUDGET_NS / 1e9)
    for l in split(sprint(versioninfo), '\n'; keepempty = false)
        println(io, "# ", l)
    end
    println(io, "label,layout,eltype,tableau,d,n,median_ns,reps")
end

function main()
    t0 = time()
    open(OUT, "w") do io
        header(io)
        print(stdout, "writing ", OUT, "\n")
        for d in DS, n in NS, (T, tn) in TYPES, e in ELTYPES, layout in LAYOUTS
            e == "Bits" && d != 2 && continue
            eltype = e == "Bits" ? Bits : nothing
            @printf("%s d=%d n=%d %s %s  (t = %d s)\n", tn, d, n, layout, e, round(Int, time() - t0))
            flush(stdout)
            run_case!(io, T, tn, layout, eltype, d, n)
        end
    end
    println("done in ", round(Int, time() - t0), " s")
end

main()
