# Workload definitions: the tableau states, operators and circuits the suite
# measures, kept separate from benchmarks.jl so they can also be driven by hand
# or reused for one-off comparisons. Deterministic given a seed.
# Single-threaded by design.
#
# The environment this runs in under CI is exactly QuditClifford +
# BenchmarkTools + stdlibs: benchpkg ignores `benchmark/Project.toml` when the
# workflow passes `--script`. Do not `using` anything else here.

using QuditClifford
using BenchmarkTools
using Random

"""Total bytes held by a tableau and all its workspaces."""
tableau_bytes(tab) = Base.summarysize(tab)

# ----------------------------------------------------------------- snapshots

"""
    TableauSnapshot

Everything a mutating operation can change about a tableau, saved so it can be
put back. Restoring is one O(n^2) `copyto!`, ~0.3 ms at n = 256, where building
the tableau afresh costs ~53 ms: `DestabilizerTableau(3, 256; state = :ghz)`
hits the worst case of `rebuild_destabilizers!`, which is O(n^3). Setup runs
inside BenchmarkTools' sample loop and counts against the time budget, so a
53 ms setup would more than halve the sample count of the ~50 ms
`canonicalize!` it exists to prepare.

Fields mirror the mutable, semantically-meaningful state of both tableau types:
`stab`, `destab`, `m`, `iscanonical` and the `xdotz_cache`. The remaining
fields are scratch workspaces that every operation overwrites before reading,
so they need no restoring.
"""
struct TableauSnapshot{TT}
    tab::TT
    stab::Matrix{Int}
    destab::Union{Matrix{Int},Nothing}
    xdotz::Vector{Int}
    m::Int
    iscanonical::Bool
end

snapshot(tab::StabilizerTableau) = TableauSnapshot(
    tab, copy(tab.stab), nothing, copy(tab.xdotz_cache), tab.m, tab.iscanonical)

snapshot(tab::DestabilizerTableau) = TableauSnapshot(
    tab, copy(tab.stab), copy(tab.destab), copy(tab.xdotz_cache), tab.m, tab.iscanonical)

"""Put the tableau back into the state it had when the snapshot was taken."""
function restore!(s::TableauSnapshot)
    copyto!(s.tab.stab, s.stab)
    s.destab === nothing || copyto!(s.tab.destab, s.destab)
    copyto!(s.tab.xdotz_cache, s.xdotz)
    s.tab.m = s.m
    s.tab.iscanonical = s.iscanonical
    return nothing
end

# ------------------------------------------------------------ leaf builders
#
# Each returns one `Benchmark`. `micro_group` assembles the eight-operation
# spine from them; the probes call them individually with one axis flipped.
#
# `mk` is a thunk `state -> tableau`, closing over d, n, T, storephase and
# inversemod, so a probe changes one keyword without duplicating any of this.

"""Build the `state -> tableau` thunk that every leaf builder takes."""
function tableau_maker(T, d::Int, n::Int;
                       storephase::Bool = true,
                       inversemod = QuditClifford.PrecomputedInvMod(d))
    return (state; basis = :Z) -> T(d, n; state = state, basis = basis,
                                    storephase = storephase, inversemod = inversemod)
end

# --------------------------------------------------------------- operators
#
# Qudit 1 is a soft best case: first column, first pivot, and the sparse
# destabilizer dual update touches the smallest possible support. The spine
# therefore measures a weight-2 operator spread across the chain instead.

"""Two sites a quarter and three quarters along the chain."""
spread_sites(n::Int) = (max(1, n ÷ 4), max(2, 3n ÷ 4))

"""
X on both spread sites. On a Z-basis `:product` state this anticommutes with
the stabilizer on each of them, so `measure!` takes the non-commuting branch.
`x*z = 0`, so it is Hermitian at d = 2.
"""
function spread_x(n::Int)
    i, j = spread_sites(n)
    return DoublePauli(i, 1, 0, j, 1, 0)
end

"""
Z on both spread sites: the product of two Z-basis generators, so on a
`:product` state it is in the stabilizer span -- with two nonzero coefficients
rather than the single one a weight-1 operator gives.
"""
function spread_z(n::Int)
    i, j = spread_sites(n)
    return DoublePauli(i, 0, 1, j, 0, 1)
end

"""Eight sites spaced evenly around the chain; distinct for every n >= 8."""
spread_eight(n::Int) = ntuple(k -> mod1(1 + (k - 1) * cld(n, 8), n), 8)

bench_construct(mk; state = :product) =
    @benchmarkable $mk($state) evals = 1

"""
`measure!` on a fresh-each-sample tableau. `evals = 1` and a snapshot restore
because the call mutates: it rewrites generators, can raise `m`, and clears
`iscanonical`. `phase_policy = 2` silences the d = 2 Hermiticity warning, which
would otherwise flood the log and dominate the timing.
"""
function bench_measure(mk, state, op)
    tab = mk(state)
    snap = snapshot(tab)
    return @benchmarkable(measure!($tab, $op; outcome = 0, phase_policy = 2),
                          setup = (restore!($snap)), evals = 1)
end

"""
`canonicalize!` from a genuinely non-canonical starting point. A `:product`
tableau is already in RCEF, so every `β == 0 && continue` guard fires and the
call collapses to O(n*m) -- it must be `:ghz`. `canonicalize!` has no
`iscanonical` early return, so resetting the flag alone would not help: the
*data* has to be restored.
"""
function bench_canonicalize(mk)
    tab = mk(:ghz)
    snap = snapshot(tab)
    return @benchmarkable(canonicalize!($tab), setup = (restore!($snap)), evals = 1)
end

"""
`expect!` on a shared, pre-canonicalized tableau -- deliberately the warm span
pipeline. A fresh `StabilizerTableau` is dirty, so its first `expect!` pays a
canonicalize and later ones do not; canonicalizing up front makes that choice
explicit instead of an artefact of sample ordering.

Note this is the *cheap* in-span query: on a freshly built `:product` state a
weight-1 Pauli decomposes into a single generator, so almost nothing
accumulates. On a `DestabilizerTableau` at n = 256 this is 3.5 us, against
13 us for the same operator on a mid-circuit state and 37 us for a dense
in-span operator there. `midcircuit_group` covers both.
"""
function bench_expect(mk, op; state = :product)
    tab = mk(state)
    canonicalize!(tab)
    expect!(tab, op)          # warm the workspaces
    return @benchmarkable expect!($tab, $op)
end

# entanglement_entropy, is_pure and reset! share one tableau each: the first two
# only write workspaces, and reset! costs the same on an already-reset tableau.
function bench_entropy(mk, sub)
    tab = mk(:ghz)
    return @benchmarkable entanglement_entropy($tab, $sub)
end

# Peripheral: one probe leaf only, at a small n. See `micro_group`.
function bench_is_pure(mk)
    tab = mk(:ghz)
    return @benchmarkable is_pure($tab)
end

function bench_reset(mk)
    tab = mk(:product)
    return @benchmarkable reset!($tab; state = :product) evals = 1
end

# ---------------------------------------------------------------- the spine

"""
    micro_group(; d, n, T, storephase = true, inversemod = QuditClifford.PrecomputedInvMod(d))

The eight core operations, for one `(type, d, n)` cell. The three `measure!`
branches are separated deliberately: their costs differ substantially and a
blended workload hides which dominates.

`is_pure` is deliberately not here. It is a peripheral function in this package
and by far the most expensive kernel (~28 ms at n = 256, where the whole rest of
the cell costs ~5 ms), so paying for it in all eight cells buys little. It gets
one cheap probe leaf instead.
"""
function micro_group(; d::Int, n::Int, T,
                     storephase::Bool = true,
                     inversemod = QuditClifford.PrecomputedInvMod(d))
    mk = tableau_maker(T, d, n; storephase = storephase, inversemod = inversemod)
    g = BenchmarkGroup()

    g["construct"] = bench_construct(mk)

    # X on both spread sites anticommutes with the Z-basis stabilizer on each,
    # so a generator is replaced.
    g["measure!/noncommuting"] = bench_measure(mk, :product, spread_x(n))
    # Z on both spread sites is a product of two stabilizers: the state is
    # unchanged and the cost is the
    # span decomposition. On a StabilizerTableau that includes a canonicalize
    # (225 us here vs 4.3 us on a Destabilizer), but only the collapsed O(n*m)
    # one -- a :product tableau is already in RCEF. The full canonicalize is
    # measured by the canonicalize! leaf, and the mid-circuit cost by
    # midcircuit_group.
    g["measure!/deterministic"] = bench_measure(mk, :product, spread_z(n))
    # From m = 0 everything commutes and lies outside the span, so m grows.
    g["measure!/append"] = bench_measure(mk, :mixed, spread_z(n))

    g["expect!"] = bench_expect(mk, spread_z(n))
    g["canonicalize!"] = bench_canonicalize(mk)
    # The pure-state path takes the smaller side, so half the chain is worst case.
    g["entropy/half"] = bench_entropy(mk, collect(1:(n ÷ 2)))
    g["reset!"] = bench_reset(mk)

    return g
end

# ---------------------------------------------------------- mid-circuit
#
# Everything above starts from a freshly constructed tableau. That is the right
# setup for construct/reset!, and it is the worst case for canonicalize!, but it
# is not the regime a user is usually in: by the time they call expect! or
# entanglement_entropy the state has been through a circuit. The difference is
# not small. On a DestabilizerTableau at n = 256, against the fresh-state leaf:
# measure!/deterministic 4.3 us -> 72 us, expect! 3.5 us -> 13 us, and
# canonicalize! 49 ms -> 2.1 ms (the mid-circuit tableau is nearly canonical
# already, so this one goes the other way).

"""
    scrambled_state(T, d, n; layers = 4, seed)

A pure state part-way through a measurement-only circuit: `:product` put
through `layers` seeded brickwork layers of random two-site measurements. `m`
stays at `n` throughout, so the state stays pure and no measurement can hit the
`m == n` append error.

Built once at suite-construction time and shared, so its cost never lands in a
timed region. Seeded, so both revisions of an A/B measure the identical state.
"""
function scrambled_state(T, d::Int, n::Int; layers::Int = 4, seed::Int = 20260910)
    rng = Random.MersenneTwister(seed)
    tab = T(d, n; state = :product)
    for layer in 1:layers
        random_brickwork_layer!(rng, tab, d, n, layer)
    end
    return tab
end

"""
Return the first candidate that anticommutes with some generator of `tab`.

A Pauli that commutes with every generator but lies outside their span makes
`measure!` throw once `m == n`, so the non-commuting benchmark cannot simply
assume a hardcoded operator still anticommutes after the state was scrambled.
"""
function first_anticommuting(tab, candidates)
    for op in candidates
        for j in 1:tab.m
            if mod(QuditClifford.commutation_col(tab.stab, j, op), tab.d) != 0
                return op
            end
        end
    end
    error("no anticommuting operator among the candidates")
end

"""One of the tableau's own generators, as a dense operator: in span by construction."""
in_span_operator(tab) = GeneralPauli(copy(tab.stab[1:(2 * tab.n), 1]), 0)

"""
    midcircuit_group(; d, n, T, seed)

The state-sensitive operations, measured on a state produced by an actual
circuit rather than by a constructor. Complements `micro_group`; it does not
replace it, because the cold-start figures are the worst case and worth
tracking on their own.
"""
function midcircuit_group(; d::Int, n::Int, T, seed::Int = 20260910)
    tab = scrambled_state(T, d, n; seed = seed)
    snap = snapshot(tab)
    g = BenchmarkGroup()

    # spread_x first; the SinglePaulis are only a fallback for the case where a
    # scrambled state happens to commute with it.
    anti = first_anticommuting(tab, Any[spread_x(n);
                                        [SinglePauli(i, 1, 0) for i in 1:min(n, 32)]])
    inspan = in_span_operator(tab)

    g["measure!/noncommuting"] = @benchmarkable(
        measure!($tab, $anti; outcome = 0, phase_policy = 2),
        setup = (restore!($snap)), evals = 1)
    g["measure!/deterministic"] = @benchmarkable(
        measure!($tab, $inspan; outcome = 0, phase_policy = 2),
        setup = (restore!($snap)), evals = 1)
    g["canonicalize!"] = @benchmarkable(
        canonicalize!($tab), setup = (restore!($snap)), evals = 1)

    # Read-only, so they can share the scrambled tableau with no per-sample setup.
    let ro = scrambled_state(T, d, n; seed = seed)
        canonicalize!(ro)
        loc = spread_z(n)
        expect!(ro, loc)
        g["expect!/local"] = @benchmarkable expect!($ro, $loc)
        dense = in_span_operator(ro)
        expect!(ro, dense)
        g["expect!/in_span"] = @benchmarkable expect!($ro, $dense)
        g["entropy/half"] = @benchmarkable entanglement_entropy($ro, $(collect(1:(n ÷ 2))))
    end

    return g
end

# ------------------------------------------------- macro workload: Ising

"""
    ising_trajectory!(; L, d = 2, p = 0.5, depth = 8L, seed, T = DestabilizerTableau)

One trajectory of the periodic measurement-only Ising circuit: `ZZ^(d-1)` on
each bond with probability `1 - p`, then `X` on each site with probability `p`.
`depth = 8L` reaches the steady-state regime. The RNG is seeded so every
revision runs the identical circuit.
"""
function ising_trajectory!(; L::Int, d::Int = 2, p::Float64 = 0.5,
                             depth::Int = 8L, seed::Int, T = DestabilizerTableau)
    rng = Random.MersenneTwister(seed)
    state = T(d, L; state = :X)

    for _ in 1:depth
        for i in 1:L
            if rand(rng) < 1 - p
                j = mod1(i + 1, L)          # periodic boundary
                measure!(state, DoublePauli(i, 0, 1, j, 0, d - 1);
                         outcome = rand(rng, 0:(d - 1)), phase_policy = 2)
            end
        end
        for i in 1:L
            if rand(rng) < p
                measure!(state, SinglePauli(i, 1, 0);
                         outcome = rand(rng, 0:(d - 1)), phase_policy = 2)
            end
        end
    end

    return state
end

# ------------------------------------------ macro workload: purification

"""Draw a nonidentity Pauli X^x Z^z on one site."""
function random_local_pauli(rng, d)
    x = rand(rng, 0:(d - 1))
    z = rand(rng, 0:(d - 1))
    while iszero(x) && iszero(z)
        x = rand(rng, 0:(d - 1))
        z = rand(rng, 0:(d - 1))
    end
    return x, z
end

"""Measure one alternating nearest-neighbour matching of the periodic chain."""
function random_brickwork_layer!(rng, tab, d, n, layer)
    first_site = isodd(layer) ? 1 : 2
    for offset in 0:(n ÷ 2 - 1)
        i = mod1(first_site + 2offset, n)
        j = mod1(i + 1, n)
        x_i, z_i = random_local_pauli(rng, d)
        x_j, z_j = random_local_pauli(rng, d)
        # phase_policy = 2: at d = 2 a random (x, z) is non-Hermitian half the
        # time, and the default policy would @warn on every such measurement.
        measure!(tab, DoublePauli(i, x_i, z_i, j, x_j, z_j);
                 outcome = rand(rng, 0:(d - 1)), phase_policy = 2)
    end
    return nothing
end

"""
    purification_trajectory(; d, n, max_layers = 10_000, seed, T = DestabilizerTableau)

Follow one maximally mixed state until its tableau reaches full rank. Returns
the rank history and the layer at which it purified (`max_layers + 1` if it did
not). `storephase = false` because only the rank is tracked, so this workload
deliberately exercises the phase-free code path.
"""
function purification_trajectory(; d::Int, n::Int, max_layers::Int = 10_000,
                                   seed::Int, T = DestabilizerTableau)
    iseven(n) || throw(ArgumentError("n must be even"))
    rng = Random.MersenneTwister(seed)

    tab = T(d, n; state = :mixed, storephase = false)
    rank_history = fill(n, max_layers + 1)
    rank_history[1] = tab.m

    for layer in 1:max_layers
        random_brickwork_layer!(rng, tab, d, n, layer)
        rank_history[layer + 1] = tab.m
        tab.m == n && return rank_history, layer
    end
    return rank_history, max_layers + 1
end
