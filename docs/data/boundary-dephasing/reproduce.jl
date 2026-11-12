# Generates the stored averages behind the "At the paper's sizes" section of
# docs/src/examples/boundary-dephasing.md: the open monitored circuit of
# Weinstein, Bao & Altman, PRL 129, 080501 (2022), arXiv:2202.12905.
#
# An open chain of L qudits starts in |0…0⟩. Every step applies one brick layer
# of random two-qudit Cliffords, measures Z on each site with probability p
# (outcome recorded), and dephases sites 1 and L completely in Z. After 4L steps
# the half-chain logarithmic negativity E and mutual information I are recorded
# in log-d units.
#
# Usage, from the repository root:
#
#   julia --project=docs -t auto docs/data/boundary-dephasing/reproduce.jl [d] [pmax]
#
# d is 2 (default) or 3, and the sweep covers p = 0:0.025:pmax (default 0.3)
# over L = 40:40:280 with 200 trajectories per point. One dimension up to
# pmax = 0.3 takes about 17 minutes on 12 threads of an Apple M3 Pro.
#
# The results go to results_d<d>.jld2 next to this script:
#
#   L, p                    chain lengths and measurement probabilities
#   E, dE, I, dI            length(p) × length(L) matrices of the mean negativity,
#                           the mean mutual information and their standard errors
#   depth_factor, trajectories, seed
#
# The file is rewritten after every point, and points it already holds are
# skipped, so an interrupted run resumes and a larger pmax extends the sweep.
# Every trajectory seeds its own Xoshiro from (seed, d, L, p, index), so the
# averages do not depend on the thread count.

using QuditClifford
using Random
using Statistics
using Printf
using JLD2

const SEED = 2022
const CHAIN_LENGTHS = collect(40:40:280)
const TRAJECTORIES = 200
const DEPTH_FACTOR = 4

function trajectory(rng, d, L; p, depth)
    tab = DestabilizerTableau(d, L; state=:Z, storephase=false)
    gates = [random_clifford(rng, d, [i, i + 1]) for i in 1:(L - 1)]
    for step in 1:depth
        for i in (isodd(step) ? 1 : 2):2:(L - 1)
            apply!(tab, random_clifford!(rng, gates[i]))
        end
        for i in 1:L
            if rand(rng) < p
                measure!(tab, SinglePauli(i, 0, 1); outcome=rand(rng, 0:(d - 1)))
            end
        end
        dephase!(tab, SinglePauli(1, 0, 1))
        dephase!(tab, SinglePauli(L, 0, 1))
    end
    half = 1:(L ÷ 2)
    return entanglement_negativity(tab, half), mutual_information(tab, half)
end

function average(d, L, p; depth, trajectories)
    E = zeros(trajectories)
    I = zeros(trajectories)
    Threads.@threads :dynamic for t in 1:trajectories
        rng = Xoshiro(hash((SEED, d, L, p, t)))
        E[t], I[t] = trajectory(rng, d, L; p=p, depth=depth)
    end
    standard_error(x) = std(x) / sqrt(length(x))
    return (mean(E), standard_error(E), mean(I), standard_error(I))
end

# The grid of probabilities, rounded so that stored and requested values compare
# equal.
probabilities(pmax) = [round(p; digits=4) for p in 0.0:0.025:pmax]

# Matrices on the requested grid, filled from an existing file where it has the
# point and NaN elsewhere.
function load_or_empty(path, ps)
    tables = Dict(k => fill(NaN, length(ps), length(CHAIN_LENGTHS)) for k in ("E", "dE", "I", "dI"))
    isfile(path) || return tables
    stored = JLD2.load(path)
    stored["L"] == CHAIN_LENGTHS || error("$path holds L = $(stored["L"]), expected $CHAIN_LENGTHS")
    for (i, p) in enumerate(ps)
        j = findfirst(==(p), stored["p"])
        j === nothing && continue
        for k in keys(tables)
            tables[k][i, :] = stored[k][j, :]
        end
    end
    return tables
end

function main(args)
    d = parse(Int, get(args, 1, "2"))
    pmax = parse(Float64, get(args, 2, "0.3"))
    path = joinpath(@__DIR__, "results_d$(d).jld2")
    ps = probabilities(pmax)
    tables = load_or_empty(path, ps)
    println("d = $d, p ≤ $pmax, threads = $(Threads.nthreads())")

    for (i, p) in enumerate(ps), (j, L) in enumerate(CHAIN_LENGTHS)
        isnan(tables["E"][i, j]) || continue
        depth = DEPTH_FACTOR * L
        seconds = @elapsed result = average(d, L, p; depth=depth, trajectories=TRAJECTORIES)
        for (k, value) in zip(("E", "dE", "I", "dI"), result)
            tables[k][i, j] = value
        end
        jldsave(path; L=CHAIN_LENGTHS, p=ps,
            E=tables["E"], dE=tables["dE"], I=tables["I"], dI=tables["dI"],
            depth_factor=DEPTH_FACTOR, trajectories=TRAJECTORIES, seed=SEED)
        @printf("L = %3d, p = %.3f: E = %.3f ± %.3f, I = %.3f ± %.3f (%.1f s)\n",
            L, p, result..., seconds)
        flush(stdout)
    end
end

main(ARGS)
