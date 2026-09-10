# AirspeedVelocity / benchpkg entry point. Must define `const SUITE::BenchmarkGroup`.
#
# The same copy of this file is run against BOTH revisions of an A/B comparison
# (the workflow passes `--script`, which pins it to the PR head), and both child
# processes inherit QC_BENCH_PROFILE from the workflow step. Never make a size
# or a seed depend on anything revision-specific, or the two columns stop being
# comparable.
#
# The CI environment is exactly QuditClifford + BenchmarkTools + stdlibs, so
# this file and workloads.jl may not `using` anything else. See benchmark/README.md.

using BenchmarkTools
using QuditClifford
using Random

include(joinpath(@__DIR__, "workloads.jl"))

const PROFILE = get(ENV, "QC_BENCH_PROFILE", "ci")

#   smoke  n = 8               correctness only; seconds to run
#   ci     n in (64, 256)      the PR job; ~60 s per revision
#   full   n up to 512         workflow_dispatch; adds d = 5, 7 and the circuits
const CONFIG = if PROFILE == "smoke"
    (ns = (8,), ds = (2, 3), probe_n = 8, circuits = false)
elseif PROFILE == "ci"
    (ns = (64, 256), ds = (2, 3), probe_n = 256, circuits = false)
elseif PROFILE == "full"
    (ns = (64, 256, 512), ds = (2, 3, 5, 7), probe_n = 512, circuits = true)
else
    error("Unknown QC_BENCH_PROFILE = $(repr(PROFILE)); expected \"smoke\", \"ci\" or \"full\".")
end

const TYPES = (StabilizerTableau, DestabilizerTableau)

const SUITE = BenchmarkGroup()

# ------------------------------------------------------------------- spine
# Eight operations x {Stab, Destab} x d x n.

for T in TYPES, d in CONFIG.ds, n in CONFIG.ns
    SUITE["micro"]["$(nameof(T))/d=$d/n=$n"] = micro_group(; d = d, n = n, T = T)
end

# ------------------------------------------------------------- mid-circuit
# The state-sensitive operations again, but on a state produced by a circuit
# rather than a constructor. Large n only: this is about the regime, not the
# scaling, and the spine already covers the scaling. See midcircuit_group.

for T in TYPES, d in CONFIG.ds
    n = last(CONFIG.ns)
    SUITE["midcircuit"]["$(nameof(T))/d=$d/n=$n"] = midcircuit_group(; d = d, n = n, T = T)
end

# ------------------------------------------------------------------ probes
# One representative configuration per secondary axis. Each probe is its own
# regression tracker; comparing probe rows against the matching spine row in
# the same table also answers "what does this axis cost".

let n = CONFIG.probe_n, d = 3, g = BenchmarkGroup()
    dense_xz = vcat(fill(1, n), fill(0, n))   # prod_i X_i: dense, and anticommutes
    s1, s2 = spread_sites(n)

    # storephase = false skips all phase bookkeeping. d = 3, where the odd-d
    # phase branches (inv2, binom2_mod_oddprime) actually run.
    for T in TYPES
        mk = tableau_maker(T, d, n; storephase = false)
        tag = "storephase=false/$(nameof(T))"
        g["$tag/measure!/noncommuting"] = bench_measure(mk, :product, spread_x(n))
        g["$tag/canonicalize!"] = bench_canonicalize(mk)
        g["$tag/expect!"] = bench_expect(mk, spread_z(n))
    end

    # JustInTimeInvMod is a type parameter, so each of these forces a fresh set
    # of specializations. Two leaves only: the gap between an @inbounds table
    # lookup and Base.invmod is around 1% at n = 256, which makes this a
    # tripwire against that changing, not a study.
    let mk = tableau_maker(DestabilizerTableau, d, n; inversemod = QuditClifford.JustInTimeInvMod())
        g["invmod=jit/canonicalize!"] = bench_canonicalize(mk)
        g["invmod=jit/measure!/noncommuting"] = bench_measure(mk, :product, spread_x(n))
    end

    # Pauli sparsity. commutation_col is O(weight) for the FewQuditPauli types
    # but O(n) for GeneralPauli, and the destabilizer dual re-orthogonalization
    # is now restricted to the operator's support -- so a dense operator
    # regresses that path back to full O(n*m). This is the probe that guards it.
    let mk = tableau_maker(DestabilizerTableau, d, n)
        # Sites spread across the chain at every weight, so these rows differ in
        # weight and not in where the support happens to sit.
        g["pauli/measure!/SinglePauli"] = bench_measure(mk, :product, SinglePauli(s1, 1, 0))
        g["pauli/measure!/DoublePauli"] = bench_measure(mk, :product, spread_x(n))
        g["pauli/measure!/NPauli8"] = bench_measure(mk, :product,
            NPauli(spread_eight(n), ntuple(_ -> 1, 8), ntuple(_ -> 0, 8)))
        g["pauli/measure!/GeneralPauli"] = bench_measure(mk, :product, GeneralPauli(dense_xz, 0))
        g["pauli/expect!/SinglePauli"] = bench_expect(mk, SinglePauli(s1, 0, 1))
        g["pauli/expect!/GeneralPauli"] = bench_expect(mk, GeneralPauli(dense_xz, 0))
    end

    # :ghz construction is ~30x :product on a DestabilizerTableau (51 ms vs
    # 1.7 ms at n = 256): rebuild_destabilizers! is O(n^3) and the GHZ
    # generator matrix is its worst case. Worth watching on its own.
    for T in TYPES
        g["construct/ghz/$(nameof(T))"] = bench_construct(tableau_maker(T, d, n); state = :ghz)
    end

    # The mixed branch of entanglement_entropy always builds the complement, so
    # a single site is its expensive case -- a different path from entropy/half.
    let mk = tableau_maker(DestabilizerTableau, d, n)
        g["entropy/single_site"] = bench_entropy(mk, [1])
        g["expect!/out_of_span"] = bench_expect(mk, spread_x(n))
    end

    # is_pure is peripheral and expensive (~28 ms at n = 256), so it gets one
    # leaf at a small size rather than a place in the spine.
    g["is_pure"] = bench_is_pure(tableau_maker(DestabilizerTableau, d, 64))

    SUITE["probe"] = g
end

# ---------------------------------------------------------------- circuits
# End-to-end trajectories. Full profile only: these are seconds each.

if CONFIG.circuits
    for L in (16, 32, 64)
        SUITE["circuit"]["ising/d=2/L=$L"] =
            @benchmarkable(ising_trajectory!(L = $L, d = 2, seed = 1234), evals = 1)
    end
    for n in (16, 32, 64)
        SUITE["circuit"]["purification/d=3/n=$n"] =
            @benchmarkable(purification_trajectory(d = 3, n = $n, seed = 1234), evals = 1)
    end
end

# ------------------------------------------------------------------ budget
# A per-leaf wall-clock cap. BenchmarkTools runs setup inside the sample loop
# and checks the elapsed-time budget across it, so `seconds` bounds the total
# job REGARDLESS of how slow the runner is -- a slow machine yields fewer
# samples, not a longer job. That makes the CI budget exact rather than a hope.
#
# Tiers are set so even the slowest leaf clears ~13 samples on a runner ~3x
# slower than the development machine. That is deliberately modest: the first
# CI run showed sample count is NOT what limits this suite. Error bars come
# back under 2% almost everywhere, while ratios between the two revisions
# scatter up to 14% -- that gap is build-to-build difference, not sampling, and
# no amount of extra samples touches it. Spending budget to shrink an error bar
# that is already 10x below the noise floor is waste. See benchmark/README.md.

function _budget(profile, keypath)
    profile == "smoke" && return 0.05
    scale = profile == "full" ? 2.0 : 1.0
    # Dispatch on the top-level group, not a substring of the joined path:
    # "midcircuit" contains "circuit", and matching that handed the midcircuit
    # leaves the five-second slot meant for whole trajectories.
    group = first(keypath)
    joined = join(keypath, "/")

    group == "circuit" && return 5.0 * scale

    # canonicalize! from a cold :ghz tableau, and :ghz construction itself, are
    # the expensive leaves: ~25 ms (Stab) and ~48-52 ms (Destab) at n = 256,
    # ~150 ms on a CI runner, against well under 1 ms for everything else in
    # the cell. They still need a bigger slice than the default.
    #
    # 2.0 s rather than the 3.0 s this started at. The first CI run showed
    # these are already the most precise rows in the table -- 0.998 +/- 0.0019
    # at ~19 samples, because they are pure integer work with no allocation
    # variance. Within-revision precision of 0.2% is far below the ~5-10%
    # between-revision scatter it is being compared against, so the extra
    # second bought nothing. 2.0 s still clears ~13 samples on a runner 3x
    # slower than a dev machine.
    #
    # The midcircuit canonicalize! starts from a nearly canonical tableau and
    # is ~20x cheaper, so it stays in the default tier.
    cold = group == "micro" || group == "probe"
    if cold && (occursin("canonicalize!", joined) || occursin("construct/ghz", joined)) &&
       !occursin("n=64", joined)
        return 2.0 * scale
    end

    occursin("is_pure", joined) && return 1.0 * scale
    occursin("n=64", joined) && return 0.2 * scale
    return 0.4 * scale
end

for (keypath, b) in BenchmarkTools.leaves(SUITE)
    b.params.seconds = _budget(PROFILE, keypath)
    b.params.gctrial = true
end

if get(ENV, "QC_BENCH_VERBOSE", "0") != "0"
    @info "QuditClifford benchmark suite" profile = PROFILE
    leaves = BenchmarkTools.leaves(SUITE)
    @info "leaves" count = length(leaves) budget_s = sum(b.params.seconds for (_, b) in leaves)
end
