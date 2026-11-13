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

"""
Evaluations per sample for leaves whose body neither mutates the tableau nor
depends on how often it ran: `expect!`, construction, the `CliffordOperator`
algebra and the samplers. With `evals = 1` the cheapest of them sit below 1 us
and come out quantized to the ~42 ns timer tick; 100 evaluations put every one
of them well above it. A leaf whose body mutates the state it measures keeps
`evals = 1` and a snapshot restore instead.
"""
const READONLY_EVALS = 100

"""Total bytes held by a tableau and all its workspaces."""
tableau_bytes(tab) = Base.summarysize(tab)

# ----------------------------------------------------------------- snapshots

"""
    TableauSnapshot

Everything a mutating operation can change about a tableau, saved so it can be
put back. Restoring is one O(n^2) `copyto!`, ~0.3 ms at n = 256.

Setup runs inside BenchmarkTools' sample loop and counts against the time
budget, so restoring rather than rebuilding keeps the sample count up. More
importantly it makes every sample start from the *same* state: the `measure!`
leaves take a different branch depending on the tableau they are handed, so a
leaf that drifted would stop measuring the branch it is named for.

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
bench_canonicalize(mk) = bench_canonicalize_state(mk(:ghz))

"""
The prime behind the `fallback-prime` probe. `benchmarks.jl` builds the leaves
from it and `test_workloads.jl` asserts the tier against it, so it is defined
once: the two cannot name different primes.

The Barrett guard REJECTS it, which is what routes the probe to the
divide-twice fallback in `src/modular.jl`. Acceptance is not monotonic in `d`,
so this cannot be replaced by "some prime above a threshold" -- 137, 139 and
443 are all accepted and 131 is not.
"""
const FALLBACK_PRIME = 131

"""
The tableau behind the `fallback-prime` probe, and the single definition both
`benchmarks.jl` and `test_workloads.jl` refer to, so the thing benchmarked and
the thing asserted cannot drift apart.

Rejection by the Barrett guard is necessary for the probe but not sufficient: a
constructor-built tableau holds only `0`, `1` and `d-1`, so every elimination
multiplier is `d-1` and it takes the multiply-free tier at any prime.
Scrambling is what puts general residues in.
"""
fallback_probe_state(n::Int; d::Int = FALLBACK_PRIME) =
    scrambled_state(DestabilizerTableau, d, n)

"""
`canonicalize!` on an already-built tableau, restored from a snapshot before
every sample. `bench_canonicalize` is this function applied to a fresh `:ghz`
tableau; take this one directly when the starting state has to be something
else.

A `:ghz` tableau's generators are `Z_{i-1} Z_i^{d-1}`, so the only nonzero
entries are `1` and `d-1`, every pivot is `1` and every elimination multiplier
is `d-1`. That takes the multiply-free tiers of `src/modular.jl` at *any* `d`
-- measured: zero general multipliers at d = 131 -- so a `:ghz` leaf cannot
exercise the Barrett or fallback paths no matter which prime it is given. Pass
a scrambled tableau to reach them.
"""
function bench_canonicalize_state(tab)
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
accumulates. On a `DestabilizerTableau` at n = 256 and d = 3 this is 1.5 us,
against 2.0 us for the same operator on a mid-circuit state and 39 us for a
dense in-span operator there. `midcircuit_group` covers both, and `dense_group`
a product of n/2 generators on a volume-law state.
"""
function bench_expect(mk, op; state = :product)
    tab = mk(state)
    canonicalize!(tab)
    expect!(tab, op)          # warm the workspaces
    return @benchmarkable expect!($tab, $op) evals = READONLY_EVALS
end

# entanglement_entropy, is_pure and reset! share one tableau each: the first two
# only write workspaces, and reset! costs the same on an already-reset tableau.
function bench_entropy(mk, sub)
    tab = mk(:ghz)
    return @benchmarkable entanglement_entropy($tab, $sub)
end

# Dense, seeded states keep this comparison on the general elimination path.
# Fixture generation is outside both timed bodies; neither body changes the
# state, so repeated samples need no restore.
function entropy_profile_fixture(T, d::Int, n::Int; m::Int=n, seed::Int=20261019)
    tab = T(d, n)
    random_state!(Random.MersenneTwister(seed), tab; m)
    return tab
end

entropy_profile_loop(tab) = [entanglement_entropy(tab, 1:l) for l in 0:tab.n]

# Keep the loop leaf on a 0.3 baseline that lacks entanglement_profile. Older
# revisions without random_state! cannot build these fixtures and skip the
# group altogether. Use d=3 to limit the cost of the full-profile loop at n=512.
function register_entropy_profile_group!(suite, types, ns; api::Module=QuditClifford)
    isdefined(api, :random_state!) || return suite
    for T in types, n in ns, m in (n ÷ 2, n)
        tab = entropy_profile_fixture(T, 3, n; m)
        group = BenchmarkGroup()
        group["loop"] = @benchmarkable entropy_profile_loop($tab)
        if isdefined(api, :entanglement_profile)
            group["profile"] = @benchmarkable QuditClifford.entanglement_profile($tab)
        end
        if isdefined(api, :entanglement_profile!)
            dest = Vector{Int}(undef, n + 1)
            group["profile!"] = @benchmarkable QuditClifford.entanglement_profile!($dest, $tab)
        end
        suite["entropy_profile"]["$(nameof(T))/d=3/n=$n/m=$m"] = group
    end
    return suite
end

# `verify=true` deliberately: the default `is_pure` is `m == n`, which there is
# nothing to benchmark in. The verifying form runs the commutation Gram and the
# rank elimination. One probe leaf only, at a small n. See `micro_group`.
#
# The keyword is probed rather than assumed. This file is run against BOTH
# revisions of an A/B comparison, and a revision that predates `verify` would
# raise a MethodError and lose its whole column -- so on those, call the
# keyword-less `is_pure`, which does the same work there. Both forms re-derive
# the full contract, so the two columns stay comparable either way.
#
# The probe is at leaf-construction time, never inside the measured body.
function bench_is_pure(mk)
    tab = mk(:ghz)
    return if hasmethod(is_pure, Tuple{typeof(tab)}, (:verify,))
        @benchmarkable is_pure($tab; verify = true)
    else
        @benchmarkable is_pure($tab)
    end
end

function bench_reset(mk)
    tab = mk(:product)
    return @benchmarkable reset!($tab; state = :product) evals = READONLY_EVALS
end

"""
`apply!` on an already-built tableau, restored from a snapshot before every
sample. Like `bench_measure` the call mutates — it rewrites the target rows of
every active column, patches the dual basis and the `xdotz_cache`, and clears
`iscanonical` — so it needs `evals = 1` and a snapshot restore.

Take this one directly, instead of `bench_apply`, when the tableau is shared
across several leaves rather than built fresh per leaf — e.g. one
[`scrambled_tableau`](@ref), built once outside any timed body, reused by
every `k` in [`stored_clifford_group`](@ref).
"""
function bench_apply_on(tab, g)
    snap = snapshot(tab)
    return @benchmarkable(apply!($tab, $g), setup = (restore!($snap)), evals = 1)
end

"""
`apply!` on a fresh-each-sample tableau: `mk(state)`, then
[`bench_apply_on`](@ref). A `:ghz` state has `m = n`, which is the full
column count.
"""
function bench_apply(mk, state, g)
    tab = mk(state)
    return bench_apply_on(tab, g)
end

"""
`apply!` on a tableau at `m = 0`, timed over `evals` evaluations per sample
rather than restored once per sample. At `m = 0` the column loops in `apply!`
run zero times -- the only effect is clearing `tab.iscanonical`, which does
not accumulate from one call to the next -- so the call is idempotent there,
and repeated evaluations need no snapshot restore between them the way
`bench_apply_on` does.

A single call is far below the timer floor: measured median `0.001 ns` at
`evals = 1` (BenchmarkTools' way of saying "could not resolve this"), against
a stable ~11-15 ns at `evals = 1000` -- checked across both tableau types,
`d = 2` and `d = 5`, and `k = 1` and `k = 8`, all consistent. `evals` defaults
to that measured 1000 rather than being recomputed per cell.
"""
function bench_apply_m0(mk, g; evals::Int = 1000)
    tab = mk(:mixed)
    return @benchmarkable(apply!($tab, $g), evals = evals)
end

"""
The `apply!` spine: one- and two-qudit gates, on the phase-carrying path.

`Fourier` has zero image `x·z`, so its odd-prime phase term is pure cross-term;
`Phase` has a nonzero one and exercises the `D` vector. `SUM` across the chain
is the two-qudit case with non-adjacent targets, matching how the measurement
leaves use `spread_sites`.
"""
function clifford_group(; d::Int, n::Int, T,
                        storephase::Bool = true,
                        inversemod = QuditClifford.PrecomputedInvMod(d))
    mk = tableau_maker(T, d, n; storephase = storephase, inversemod = inversemod)
    s1, s2 = spread_sites(n)
    g = BenchmarkGroup()
    g["apply!/fourier"] = bench_apply(mk, :ghz, Fourier(s1))
    g["apply!/phase"] = bench_apply(mk, :ghz, Phase(s1))
    g["apply!/sum"] = bench_apply(mk, :ghz, SUM(s1, s2, 1))
    return g
end

"""
    stored_clifford_fixture(d, k)

A nontrivial `k`-qudit `CliffordOperator` at dimension `d`, built by
conjugating each basis Pauli through a fixed sequence of named gates
(`Fourier` and `Phase` on every qudit, `SUM` chaining adjacent qudits).
Construction runs outside any measured benchmark, and tests use this SAME
builder, so a benchmark fixture and a correctness fixture never drift apart.

Returns `(; U, gates, targets, F, a)`: `U` is the stored operator, `gates` the
named-gate sequence it was built from, and `targets`, `F`, `a` its raw
constructor arguments.
"""
function stored_clifford_fixture(d::Int, k::Int)
    k > 0 || throw(ArgumentError("benchmark support must be positive"))
    # Construct a nontrivial k-qudit action from named basis-image conjugation.
    # This setup is outside measured apply!, and tests use this SAME builder.
    gates = QuditClifford.AbstractClifford[]
    for q in 1:k
        push!(gates, QuditClifford.Fourier(q), QuditClifford.Phase(q))
    end
    for q in 1:(k - 1)
        push!(gates, QuditClifford.SUM(q, q + 1))
    end
    F = zeros(Int, 2k, 2k)
    a = zeros(Int, 2k)
    for j in 1:(2k)
        xz = zeros(Int, 2k); xz[j] = 1
        op = QuditClifford.GeneralPauli(xz, 0)
        for g in gates
            op = QuditClifford.conjugate(g, op; d)
        end
        F[:, j] = op.xz
        a[j] = op.phase
    end
    targets = collect(1:k)
    U = QuditClifford.CliffordOperator(d, targets, F, a)
    return (; U, gates, targets, F, a)
end

"""
    scrambled_tableau(mk, d, n; seed, depth = 4)

A pure state produced by `depth` seeded rounds of *unitary* Clifford action on
`mk(:product)`, rather than by measurement: each round, for every qudit `q` in
`1:n`, applies `Fourier(q)` or `Phase(q)` (a seeded coin) and then `SUM(q, q2,
c)` to a seeded partner `q2 != q` with coefficient `c in 1:(d-1)`. Only named
gates are used, so `apply!` keeps a `DestabilizerTableau`'s dual basis and
`xdotz_cache` consistent with no extra bookkeeping here, and the result stays
physical: `is_pure(tab; verify = true)`.

Deterministic given `(mk, d, n, seed)` -- the RNG is seeded once and driven in
a fixed order, so two calls with the same arguments return bit-identical
tableaux and a different `seed` gives a different one.

This is a different kind of scrambling from [`scrambled_state`](@ref), which
drives `measure!` for the mid-circuit group: that one always builds
`T(d, n; state = :product)` directly, ignoring `storephase`/`inversemod`, and
its non-commuting outcomes are only as scrambled as a measurement-only circuit
gets. This one exists to give the stored-operator benchmarks DENSE local
coordinates. A `:product` or `:mixed` tableau's active rows are mostly zero or
one-hot; above `S = 24` the dense matvec in `_matvec_prepared!`
(`src/apply_clifford.jl`) skips a zero column outright, so those leaves
measure a sparse fast path rather than the general case. Repeatedly coupling
every qudit to a random partner via `SUM` spreads support across the whole
register, so a small-`k` stored operator's target rows see mostly-nonzero
entries instead.
"""
function scrambled_tableau(mk, d::Int, n::Int; seed::Int, depth::Int = 4)
    n >= 2 || throw(ArgumentError(
        "scrambled_tableau needs at least 2 qudits for SUM, got n = $n."))
    rng = Random.MersenneTwister(seed)
    tab = mk(:product)
    for _ in 1:depth
        for q in 1:n
            if rand(rng, Bool)
                apply!(tab, Fourier(q))
            else
                apply!(tab, Phase(q))
            end
            q2 = rand(rng, 1:(n - 1))
            q2 >= q && (q2 += 1)
            c = rand(rng, 1:(d - 1))
            apply!(tab, SUM(q, q2, c))
        end
    end
    return tab
end

"""
    stored_clifford_group(; d, n, T, ks = (1, 2, 8), product = false,
                           storephase = true,
                           inversemod = QuditClifford.PrecomputedInvMod(d),
                           seed = 20260910)

Stored `CliffordOperator` `apply!`, for one `(type, d, n)` cell and each
support size `k` in `ks`. Every `k` gets:

- `"apply!/stored/k=\$k"`, applying to a [`scrambled_tableau`](@ref) built ONCE
  for the cell and shared across every `k` -- dense, non-one-hot local
  coordinates, so this measures the general dense matvec rather than the
  zero-skip fast path (see `scrambled_tableau`'s docstring).
- `"apply!/stored/m=0/k=\$k"`, applying to the maximally mixed state, making
  the validation-path cost `apply!` pays before any generator exists visible
  on its own. Built by [`bench_apply_m0`](@ref): a single call there is far
  below the timer floor, so it is timed over many evaluations per sample
  instead of restored once per sample.

When `product`, also `"apply!/stored/product/k=\$k"` on a freshly restored
`:product` state -- the sparse, mostly-zero/one-hot input the dense matvec's
zero-skip fast path serves, which the scrambled leaf above deliberately avoids.
Meant for the `full` profile only (see `CONFIG` in
`benchmark/benchmarks.jl`), so this demonstration leaf never costs the
PR-blocking `ci` job any samples.

Construction cost is not in here: see
[`stored_clifford_construct_group`](@ref), registered once per `d` rather
than once per `(T, n)` cell, because construction touches no tableau.
"""
function stored_clifford_group(; d::Int, n::Int, T, ks = (1, 2, 8),
                               product::Bool = false,
                               storephase::Bool = true,
                               inversemod = QuditClifford.PrecomputedInvMod(d),
                               seed::Int = 20260910)
    mk = tableau_maker(T, d, n; storephase, inversemod)
    scrambled = scrambled_tableau(mk, d, n; seed)
    group = BenchmarkGroup()
    for k in unique(collect(ks))
        1 <= k <= n || continue
        fixture = stored_clifford_fixture(d, k)
        U = fixture.U
        group["apply!/stored/k=$k"] = bench_apply_on(scrambled, U)
        # Every k also has an m=0 leaf, making validation scaling visible.
        # evals = 1000 (bench_apply_m0's default): a single call is far below
        # the timer floor, and apply! at m=0 is idempotent, so no restore.
        group["apply!/stored/m=0/k=$k"] = bench_apply_m0(mk, U)
        if product
            group["apply!/stored/product/k=$k"] = bench_apply(mk, :product, U)
        end
    end
    return group
end

"""
    stored_clifford_construct_group(d; ks = (1, 2, 8))

`CliffordOperator` construction, checked and unchecked, for each support size
`k` in `ks` at dimension `d`. Construction takes no tableau or register size,
so unlike [`stored_clifford_group`](@ref) this group depends only on `d` and
is registered once per `d` by `register_clifford_group!`, not once per
`(T, n)` cell -- registering it per cell would duplicate the same `(d, k)`
measurement across every tableau type and register size.
"""
function stored_clifford_construct_group(d::Int; ks = (1, 2, 8))
    group = BenchmarkGroup()
    for k in unique(collect(ks))
        fixture = stored_clifford_fixture(d, k)
        t, F, a = fixture.targets, fixture.F, fixture.a
        group["k=$k/checked"] = @benchmarkable(
            QuditClifford.CliffordOperator($d, $t, $F, $a), evals = READONLY_EVALS)
        group["k=$k/unchecked"] = @benchmarkable(
            QuditClifford.CliffordOperator($d, $t, $F, $a; check = false),
            evals = READONLY_EVALS)
    end
    return group
end

"""
    stored_clifford_algebra_group(d; ks = (1, 2, 8), seed = 20260910)

`conjugate`, `inv` and composition (`∘`) on a stored [`CliffordOperator`](@ref),
for each support size `k` in `ks` at dimension `d`. Like
[`stored_clifford_construct_group`](@ref), these touch no tableau, so this is
registered once per `d` by `register_clifford_group!` rather than once per
`(T, n)` cell.

`U = stored_clifford_fixture(d, k).U`, built outside any timed body:

- `"conjugate/k=\$k"` conjugates a seeded `GeneralPauli` on `U`'s own
  `k`-qudit register, dense in all `2k` exponent coordinates and with a random
  phase -- exercising the general Heisenberg path rather than a single weight-1
  basis generator. `test_workloads.jl`
  checks it against sequential named conjugation through `fixture.gates`.
- `"inv/k=\$k"` and `"compose/k=\$k"` (`U ∘ U`) are allocating calls that leave
  `U` and its scratch untouched, so neither needs a snapshot restore.

All three run `READONLY_EVALS` evaluations per sample.
"""
function stored_clifford_algebra_group(d::Int; ks = (1, 2, 8), seed::Int = 20260910)
    group = BenchmarkGroup()
    rng = Random.MersenneTwister(seed)
    for k in unique(collect(ks))
        fixture = stored_clifford_fixture(d, k)
        U = fixture.U
        S = 2k
        xz = [rand(rng, 0:(d - 1)) for _ in 1:S]
        phase = rand(rng, 0:(QuditClifford.phase_modulus(d) - 1))
        op = QuditClifford.GeneralPauli(xz, phase)
        group["conjugate/k=$k"] =
            @benchmarkable QuditClifford.conjugate($U, $op) evals = READONLY_EVALS
        group["inv/k=$k"] = @benchmarkable inv($U) evals = READONLY_EVALS
        group["compose/k=$k"] = @benchmarkable $U ∘ $U evals = READONLY_EVALS
    end
    return group
end

"""
    stored_clifford_safe_group(types)

The overflow-safe tier's only timed coverage, at one fixed, large prime:
`d = 1_000_000_007`, `n = 8`, `k = 8`.

Every `d` elsewhere in the suite (2, 3, 5, 7) is small enough that
`clifford_fast_dots` (`src/clifford_arithmetic.jl`) takes the unguarded `Int`
fast path for a `2k`-term Clifford dot, so without this group the
`add_mod`/`mul_mod`/`widemul` safe tier those dots fall back to would never
run under a timer. `n*(d-1)^2 ≈ 8.0e18` stays under `typemax(Int) ≈ 9.22e18`,
so the tableau itself is inside the package-wide envelope
(`max_safe_dimension`, `src/helper.jl`) and raises no overflow warning; but a
Clifford dot sums `2k` rather than `n` terms, so `clifford_fast_dots(2k, d) =
clifford_fast_dots(16, d)` is false regardless of that headroom -- exactly the
gap `src/clifford_arithmetic.jl`'s header comment describes.
`test_workloads.jl` asserts both facts directly rather than trusting this
arithmetic.

`JustInTimeInvMod()` throughout: `PrecomputedInvMod` allocates a `d - 1`-entry
table, which would be gigabytes at this `d`.

Leaves: `"apply!/\$(nameof(T))/k=8"` on a [`scrambled_tableau`](@ref) for each
`T` in `types`, built directly here; plus `"conjugate/k=8"`, `"inv/k=8"` and
`"compose/k=8"`, merged in from `stored_clifford_algebra_group(d; ks = (8,))`
rather than rebuilt inline, so the fixture, the random observable and the
leaf names cannot drift from that group's own `k = 8` cell.

Registered as `suite["clifford"]["stored/safe/d=1000000007"]`, in every
profile: it is cheap (one `n = 8` tableau per type, one `k = 8` fixture) and
none of `CONFIG`'s own `ds` ever reaches this `d`.
"""
function stored_clifford_safe_group(types)
    d = 1_000_000_007
    n = 8
    k = 8
    group = BenchmarkGroup()
    fixture = stored_clifford_fixture(d, k)
    U = fixture.U
    for T in types
        mk = tableau_maker(T, d, n; inversemod = QuditClifford.JustInTimeInvMod())
        scrambled = scrambled_tableau(mk, d, n; seed = 20260910)
        group["apply!/$(nameof(T))/k=$k"] = bench_apply_on(scrambled, U)
    end
    # Identical in shape to stored_clifford_algebra_group's own k=8 cell, so
    # build it there and merge rather than duplicating the fixture and the
    # random-observable draw here.
    for (key, leaf) in stored_clifford_algebra_group(d; ks = (k,))
        group[key] = leaf
    end
    return group
end

# The PR-head script also runs against the baseline package, which may lack APIs
# this file uses. Check that every API a group needs exists before calling any
# of its constructors.
#
# `ks` and `product` thread straight through to `stored_clifford_group`.
# `stored_clifford_construct_group` and `stored_clifford_algebra_group` reuse
# the same `ks`, once per `d` rather than once per `(T, n)` cell, since
# neither touches a tableau. `stored_clifford_safe_group` takes no `ks`: it is
# pinned to one fixed, large `d` and registered unconditionally, in every
# profile.
function register_clifford_group!(suite, types, ds, ns; ks = (1, 2, 8),
                                  product::Bool = false, api::Module = QuditClifford)
    required = (:apply!, :Fourier, :Phase, :SUM)
    all(name -> isdefined(api, name), required) || return suite
    for T in types, d in ds, n in ns
        suite["clifford"]["$(nameof(T))/d=$d/n=$n"] =
            clifford_group(; d, n, T)
    end
    isdefined(api, :CliffordOperator) || return suite
    for T in types, d in ds, n in ns
        suite["clifford"]["stored/$(nameof(T))/d=$d/n=$n"] =
            stored_clifford_group(; d, n, T, ks, product)
    end
    # Construction and the algebra ops (conjugate/inv/compose) depend only on
    # d and k, not on T or n, so both are registered once per d here rather
    # than once per (T, n) cell inside the loop above.
    for d in ds
        suite["clifford"]["stored/construct/d=$d"] = stored_clifford_construct_group(d; ks)
        suite["clifford"]["stored/algebra/d=$d"] = stored_clifford_algebra_group(d; ks)
    end
    # The overflow-safe tier, once, regardless of profile or ds: see
    # stored_clifford_safe_group for why none of CONFIG's own ds reaches it.
    suite["clifford"]["stored/safe/d=1000000007"] = stored_clifford_safe_group(types)
    return suite
end

# ---------------------------------------------------------------- the spine

"""
    micro_group(; d, n, T, storephase = true, inversemod = QuditClifford.PrecomputedInvMod(d))

The eight core operations, for one `(type, d, n)` cell. The three `measure!`
branches are separated deliberately: their costs differ substantially and a
blended workload hides which dominates.

`is_pure` is deliberately not here: it tests `m == n` and nothing else, so there
is no kernel behind it to measure. Its verifying form, `is_pure(; verify=true)`,
does `O(n^3)` work and gets one probe leaf.
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
    # (134 us here vs 2.4 us on a Destabilizer at d = 3), but only the
    # collapsed O(n*m) one -- a :product tableau is already in RCEF. The full canonicalize is
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
# measure!/deterministic 2.4 us -> 77 us at d = 3, and canonicalize!
# 8.4 ms -> 0.49 ms (the mid-circuit tableau is nearly canonical already, so
# this one goes the other way).

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

    # Read-only, so they can share one scrambled tableau with no per-sample
    # setup. Canonicalized up front for the reason in `bench_expect`.
    let ro = scrambled_state(T, d, n; seed = seed)
        canonicalize!(ro)
        loc = spread_z(n)
        expect!(ro, loc)
        g["expect!/local"] = @benchmarkable expect!($ro, $loc) evals = READONLY_EVALS
        dense = in_span_operator(ro)
        expect!(ro, dense)
        g["expect!/in_span"] = @benchmarkable expect!($ro, $dense) evals = READONLY_EVALS
    end

    # entanglement_entropy gets its OWN tableau, deliberately NOT canonicalized.
    # Unlike expect!, it never canonicalizes -- it reads tab.stab straight into
    # the workspace -- so a caller reaching it after a run of measure! has
    # iscanonical == false, and how much the elimination in rank_fp_cols! has to
    # do depends entirely on how close the generators already are to echelon
    # form. Sharing the canonicalized tableau above would measure a state no
    # caller is in: at n = 256 the restricted generator matrix of a dirty
    # DestabilizerTableau needs 424 eliminations against 2 canonical, so the leaf
    # would understate the real cost and be nearly blind to changes in the
    # kernel.
    #
    # A StabilizerTableau barely moves either way, and that is a real asymmetry
    # rather than a flaw here: `coeffs_from_generators!` canonicalizes it on
    # every commuting measurement, so it ends a circuit near echelon form
    # already, carrying a stale iscanonical == false.
    let ro = scrambled_state(T, d, n; seed = seed)
        g["entropy/half"] = @benchmarkable entanglement_entropy($ro, $(collect(1:(n ÷ 2))))
    end

    return g
end

# ------------------------------------------------------- dense states
#
# The spine starts from constructors, whose tableaux hold only 0, 1 and d-1,
# and the mid-circuit state is scrambled by local measurements only. A uniformly
# random stabilizer state is volume-law: most entries are general residues, a
# local operator anticommutes with most generators, and an in-span operator
# decomposes into many of them. These leaves time that regime.

"""Seed of every [`dense_state`](@ref) fixture."""
const DENSE_SEED = 20261113

"""
    dense_state(T, d, n; m = n, seed = DENSE_SEED)

A uniformly random stabilizer state with `m` generators, drawn by
`random_state!` from a seeded RNG. Both tableau types hold the identical state:
the `DestabilizerTableau` is drawn, and the `StabilizerTableau` is built from
its generator matrix, so a `StabilizerTableau` and a `DestabilizerTableau` row
of the same `(d, n, m)` time the same physics.

`random_state!` leaves the generators in a non-canonical basis, which is what
the `canonicalize!` leaves need.
"""
function dense_state(T, d::Int, n::Int; m::Int = n, seed::Int = DENSE_SEED)
    D = DestabilizerTableau(d, n)
    random_state!(Random.Xoshiro(seed + 7919 * d + n), D; m)
    T === DestabilizerTableau && return D
    return StabilizerTableau(d, D.stab[:, 1:m]; m)
end

"""
    span_product(tab; k = cld(tab.m, 2), seed = DENSE_SEED)

A product of `k` distinct generators of `tab`, each raised to a seeded power in
`1:d-1`, as a dense `GeneralPauli`. It lies in the stabilizer group, so
`measure!` takes the deterministic branch and `expect!` returns a phase, and
its decomposition has exactly `k` nonzero coefficients -- against the one or
two a weight-2 operator has on a `:product` state.
"""
function span_product(tab; k::Int = cld(tab.m, 2), seed::Int = DENSE_SEED)
    rng = Random.Xoshiro(seed)
    n, d = tab.n, tab.d
    xz = zeros(Int, 2n)
    for j in Random.randperm(rng, tab.m)[1:k]
        c = rand(rng, 1:(d - 1))
        @views xz .= mod.(xz .+ c .* tab.stab[1:(2n), j], d)
    end
    return GeneralPauli(xz, 0)
end

"""
    commuting_out_of_span(D::DestabilizerTableau; seed = DENSE_SEED)

A dense Pauli that commutes with every generator of the mixed state `D` but is
not in its stabilizer group, so `measure!` takes the append branch and raises
`m` by one. A seeded random Pauli `v` is projected onto the commutant with the
dual basis: `w = v - sum_j (<s_j, v> / <s_j, t_j>) t_j` over the generators
`s_j` and their destabilizers `t_j`. With `m < n` the commutant is strictly
larger than the span, and `test_workloads.jl` asserts this draw lands outside
it.
"""
function commuting_out_of_span(D::DestabilizerTableau; seed::Int = DENSE_SEED)
    n, d, m = D.n, D.d, D.m
    rng = Random.Xoshiro(seed)
    w = rand(rng, 0:(d - 1), 2n)
    for j in 1:m
        t = D.destab[1:(2n), j]
        c = mod(QuditClifford.commutation_col(D.stab, j, GeneralPauli(w, 0)), d)
        λ = mod(QuditClifford.commutation_col(D.stab, j, GeneralPauli(t, 0)), d)
        w .= mod.(w .- c * invmod(λ, d) .* t, d)
    end
    return GeneralPauli(w, 0)
end

"""
    bench_measure_on(tab, op)

`measure!(tab, op)` on an already-built tableau, restored from a snapshot
before every sample. [`bench_measure`](@ref) is this applied to a fresh
constructor state.
"""
function bench_measure_on(tab, op)
    snap = snapshot(tab)
    return @benchmarkable(measure!($tab, $op; outcome = 0, phase_policy = 2),
                          setup = (restore!($snap)), evals = 1)
end

"""
    bench_expect_on(tab, op)

`expect!` on a canonicalized copy of `tab`, as in [`bench_expect`](@ref).
"""
function bench_expect_on(tab, op)
    ro = copy(tab)
    canonicalize!(ro)
    expect!(ro, op)
    return @benchmarkable expect!($ro, $op) evals = READONLY_EVALS
end

"""
    dense_group(; d, n, T)

The state-sensitive operations on [`dense_state`](@ref) fixtures:

- `measure!/noncommuting`: `spread_x` on the pure state, which anticommutes with
  most generators, so the update rewrites most columns.
- `measure!/deterministic`: a [`span_product`](@ref) of `n/2` generators on the
  pure state.
- `measure!/append`: a [`commuting_out_of_span`](@ref) Pauli on the state with
  `m = n/2`, so the dual-basis loop of the append branch runs over `n/2`
  generators rather than none.
- `expect!`: the same `span_product`, on a canonicalized copy.
- `canonicalize!` of the pure state.

`test_workloads.jl` asserts the branch and `m` of each measurement leaf.
"""
function dense_group(; d::Int, n::Int, T)
    pure = dense_state(T, d, n)
    mixed = dense_state(T, d, n; m = n ÷ 2)
    inspan = span_product(pure)
    g = BenchmarkGroup()
    g["measure!/noncommuting"] = bench_measure_on(pure, first_anticommuting(pure, [spread_x(n)]))
    g["measure!/deterministic"] = bench_measure_on(pure, inspan)
    g["measure!/append"] = bench_measure_on(mixed,
        commuting_out_of_span(dense_state(DestabilizerTableau, d, n; m = n ÷ 2)))
    g["expect!"] = bench_expect_on(pure, inspan)
    g["canonicalize!"] = bench_canonicalize_state(pure)
    return g
end

# ------------------------------------------------------- modulus breadth

"""
The primes behind the `modulus` group, one per arithmetic tier of
`src/modular.jl` that a general multiplier can take above `d = 5`:

- `131` is rejected by the Barrett guard: the divide-twice fallback.
- `443` is accepted although it is larger: the Barrett tier at a mid-size prime.
- `1031` is above the guard's table: the fallback through the out-of-table path.

`test_workloads.jl` counts the tier every leaf of the group reaches.
"""
const MODULUS_PRIMES = (131, 443, 1031)

"""
    modulus_group(; d, n, T)

`measure!` (both branches), `expect!`, `entropy/half` and `canonicalize!` on a
pure [`dense_state`](@ref) at dimension `d`. The dense state is what makes the
modulus matter: its multipliers are general residues, where a constructor
state's are `1` or `d-1` and take the multiply-free tiers at every prime.
"""
function modulus_group(; d::Int, n::Int, T)
    tab = dense_state(T, d, n)
    inspan = span_product(tab)
    g = BenchmarkGroup()
    g["measure!/noncommuting"] = bench_measure_on(tab, first_anticommuting(tab, [spread_x(n)]))
    g["measure!/deterministic"] = bench_measure_on(tab, inspan)
    g["expect!"] = bench_expect_on(tab, inspan)
    let ro = copy(tab), sub = collect(1:(n ÷ 2))
        g["entropy/half"] = @benchmarkable entanglement_entropy($ro, $sub)
    end
    g["canonicalize!"] = bench_canonicalize_state(tab)
    return g
end

# ------------------------------------------------------------ channels

"""
    channel_group(; d, n, T)

The non-unitary channels on a pure [`dense_state`](@ref), each restored from a
snapshot before every sample: `dephase!` along `spread_x`, `traceout!` of one
site and of a contiguous quarter of the chain, and the single-site `reset!`.
"""
function channel_group(; d::Int, n::Int, T)
    tab = dense_state(T, d, n)
    snap = snapshot(tab)
    s1, _ = spread_sites(n)
    quarter = collect(s1:(s1 + n ÷ 4 - 1))
    op = spread_x(n)
    g = BenchmarkGroup()
    g["dephase!"] = @benchmarkable(dephase!($tab, $op), setup = (restore!($snap)), evals = 1)
    g["traceout!/site"] = @benchmarkable(traceout!($tab, $s1),
                                         setup = (restore!($snap)), evals = 1)
    g["traceout!/quarter"] = @benchmarkable(traceout!($tab, $quarter),
                                            setup = (restore!($snap)), evals = 1)
    g["reset!/site"] = @benchmarkable(reset!($tab, $s1), setup = (restore!($snap)), evals = 1)
    return g
end

"""The first three contiguous quarters of the chain, as the subsystems `A`, `B`, `C`."""
quarters(n::Int) = (collect(1:(n ÷ 4)), collect((n ÷ 4 + 1):(n ÷ 2)),
                    collect((n ÷ 2 + 1):(3n ÷ 4)))

"""
    information_group(; d, n, T)

Mutual, tripartite and conditional mutual information and the negativity on
adjacent quarters of a [`dense_state`](@ref), pure (`m = n`) and mixed
(`m = n/2`). None of them changes the state, so the tableaux are shared across
samples.
"""
function information_group(; d::Int, n::Int, T)
    A, B, C = quarters(n)
    g = BenchmarkGroup()
    for m in (n, n ÷ 2)
        tab = dense_state(T, d, n; m)
        g["m=$m/mutual_information"] = @benchmarkable mutual_information($tab, $A, $B)
        g["m=$m/tripartite_information"] =
            @benchmarkable tripartite_information($tab, $A, $B, $C)
        g["m=$m/conditional_mutual_information"] =
            @benchmarkable conditional_mutual_information($tab, $A, $B, $C)
        g["m=$m/entanglement_negativity"] =
            @benchmarkable entanglement_negativity($tab, $A, $B)
    end
    return g
end

# --------------------------------------------------------------- gates

"""
    gate_group(; d, n, T)

The named gates the `clifford` spine leaves out -- `Multiplier`, `PauliGate`,
`CPhase`, `SWAP` and `SUM` with a coefficient other than `1` -- plus `Fourier`
and `SUM` on a `storephase = false` tableau. Applied to a restored `:ghz` state
like the spine gates. Every coefficient is `2`, so at `d >= 5` none of them is
`1` or `d-1`.
"""
function gate_group(; d::Int, n::Int, T)
    s1, s2 = spread_sites(n)
    mk = tableau_maker(T, d, n)
    g = BenchmarkGroup()
    g["apply!/multiplier"] = bench_apply(mk, :ghz, Multiplier(s1, 2))
    g["apply!/pauli"] = bench_apply(mk, :ghz, PauliGate(s1, 1, 2))
    g["apply!/cphase"] = bench_apply(mk, :ghz, CPhase(s1, s2, 2))
    g["apply!/swap"] = bench_apply(mk, :ghz, SWAP(s1, s2))
    g["apply!/sum/a=2"] = bench_apply(mk, :ghz, SUM(s1, s2, 2))
    nophase = tableau_maker(T, d, n; storephase = false)
    g["storephase=false/apply!/fourier"] = bench_apply(nophase, :ghz, Fourier(s1))
    g["storephase=false/apply!/sum"] = bench_apply(nophase, :ghz, SUM(s1, s2, 1))
    return g
end

# ------------------------------------------------------ tableau utilities

"""
    tableau_group(; d, n, T)

Tableau housekeeping on a dense, non-canonical [`scrambled_tableau`](@ref):

- `copy` and `copy!` into a preallocated tableau.
- `==` against a copy in the same basis (the entrywise fast path) and against
  a canonicalized copy (the comparison canonicalizes a temporary), and `hash`.
- The raw-matrix constructor, with and without `check`, at `n ÷ 4` qudits:
  building a `DestabilizerTableau` from a dense generator matrix costs ~0.1 s at
  `n = 256`, which leaves too few samples in a CI budget.
- `measure!(rng, ...)` with the outcome drawn from the RNG, which is reseeded
  in `setup` together with the snapshot restore. Registered only where that
  method exists.

The read-only leaves share one tableau. `measure!/rng` gets its own copy,
because a restoring leaf leaves its last sample's state behind.
"""
function tableau_group(; d::Int, n::Int, T)
    mk = tableau_maker(T, d, n)
    tab = scrambled_tableau(mk, d, n; seed = DENSE_SEED)
    same = copy(tab)
    other = copy(tab)
    canonicalize!(other)
    dst = copy(tab)
    nraw = max(2, n ÷ 4)
    M = scrambled_tableau(tableau_maker(T, d, nraw), d, nraw; seed = DENSE_SEED).stab
    g = BenchmarkGroup()
    g["copy"] = @benchmarkable copy($tab)
    g["copy!"] = @benchmarkable copy!($dst, $tab)
    g["==/same-basis"] = @benchmarkable $tab == $same evals = READONLY_EVALS
    g["==/other-basis"] = @benchmarkable $tab == $other
    g["hash"] = @benchmarkable hash($tab)
    g["construct/raw/n=$nraw/checked"] = @benchmarkable $T($d, $M; m = $nraw)
    g["construct/raw/n=$nraw/unchecked"] =
        @benchmarkable $T($d, $M; m = $nraw, check = false)
    op = first_anticommuting(tab, [spread_x(n)])
    if hasmethod(measure!, Tuple{Random.AbstractRNG,typeof(tab),typeof(op)})
        rng = Random.Xoshiro(DENSE_SEED)
        mtab = copy(tab)
        snap = snapshot(mtab)
        g["measure!/rng"] = @benchmarkable(
            measure!($rng, $mtab, $op; phase_policy = 2),
            setup = (restore!($snap); Random.seed!($rng, $DENSE_SEED)), evals = 1)
    end
    return g
end

"""
    register_breadth_groups!(suite, types; n, dense_ds, modulus_n)

Register `dense` (every `d` in `dense_ds` at size `n`), `modulus` (every prime
in [`MODULUS_PRIMES`](@ref) at `modulus_n`), and the `channels`,
`information`, `gates` and `tableau` groups at size `n`. Everything here
builds its fixtures with `random_state!`, so a baseline without it keeps its
column and loses only these groups.
"""
function register_breadth_groups!(suite, types; n::Int, dense_ds, modulus_n::Int,
                                  api::Module = QuditClifford)
    isdefined(api, :random_state!) || return suite
    for T in types
        tn = nameof(T)
        for d in dense_ds
            suite["dense"]["$tn/d=$d/n=$n"] = dense_group(; d, n, T)
        end
        for d in MODULUS_PRIMES
            suite["modulus"]["$tn/d=$d/n=$modulus_n"] = modulus_group(; d, n = modulus_n, T)
        end
        suite["channels"]["$tn/d=3/n=$n"] = channel_group(; d = 3, n, T)
        suite["information"]["$tn/d=3/n=$n"] = information_group(; d = 3, n, T)
        suite["gates"]["$tn/d=5/n=$n"] = gate_group(; d = 5, n, T)
        suite["tableau"]["$tn/d=3/n=$n"] = tableau_group(; d = 3, n, T)
    end
    return suite
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
not). Purification slows sharply with `n`: with seed 1234 at `d = 3`, `n = 16`
purifies after 42 layers and `n = 32` after 1876, while `n = 64` is still at
rank 60 after 10 000, so that run is a fixed-length one. `storephase = false` because only the rank is tracked, so this workload
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

# ------------------------------------------ random Cliffords and states

"""
    random_operator_fixture(d, k; seed = 20260925)

A seeded `Xoshiro` and a `k`-qudit `CliffordOperator` sampled from it at
dimension `d`, as `(; rng, U)`. The refill leaves and their tests share this
builder.
"""
function random_operator_fixture(d::Int, k::Int; seed::Int = 20260925)
    rng = Random.Xoshiro(seed + k)
    return (; rng, U = QuditClifford.random_clifford(rng, d, 1:k))
end

"""
    random_refill_group(d; ks = (1, 2, 8), seed = 20260925)

`random_clifford!` on a warmed operator, `"refill/d=\$d/k=\$k"` for each `k`
in `ks`. A refill reads nothing of the operator's previous action, so the
leaf needs no restore between evaluations, and it allocates nothing. The RNG
is reseeded in `setup`, so every sample draws the same operators.
"""
function random_refill_group(d::Int; ks = (1, 2, 8), seed::Int = 20260925)
    group = BenchmarkGroup()
    for k in unique(collect(ks))
        (; rng, U) = random_operator_fixture(d, k; seed)
        s = seed + k
        group["refill/d=$d/k=$k"] = @benchmarkable(
            QuditClifford.random_clifford!($rng, $U),
            setup = (Random.seed!($rng, $s)), evals = READONLY_EVALS)
    end
    return group
end

"""
    random_state_fixture(T, d, n; seed = 20260925)

A seeded `Xoshiro` and a `T(d, n)` tableau, as `(; rng, tab)`, shared by the
state leaves and their tests.
"""
random_state_fixture(T, d::Int, n::Int; seed::Int = 20260925) =
    (; rng = Random.Xoshiro(seed + n), tab = T(d, n))

"""
    random_state_group(; d, n, T, seed = 20260925)

`random_state!` with `m = 1` and `m = n` on one `T(d, n)` tableau:
`"state/\$(nameof(T))/d=\$d/n=\$n/m=\$m"`. A replacement reads nothing of the
previous state, so repeated evaluations need no restore. Both leaves allocate
the same `2n × 2n` scratch, so their difference is the `O(n²m)` sampling. The
RNG is reseeded in `setup`, so every sample draws the same state.
"""
function random_state_group(; d::Int, n::Int, T, seed::Int = 20260925)
    group = BenchmarkGroup()
    (; rng, tab) = random_state_fixture(T, d, n; seed)
    s = seed + n
    for m in unique((1, n))
        group["state/$(nameof(T))/d=$d/n=$n/m=$m"] = @benchmarkable(
            QuditClifford.random_state!($rng, $tab; m = $m),
            setup = (Random.seed!($rng, $s)), evals = 1)
    end
    return group
end

# ------------------------------------ macro workload: monitored circuit

"""
    monitored_setup(; d, n, seed, T = DestabilizerTableau)

The state of a [`monitored_trajectory`](@ref) before its first layer, as
`(tab, bonds, rng)`: a seeded `Xoshiro`, a `:Z` product tableau, and one
two-qudit operator per bond of the periodic chain, `bonds[q]` acting on qudits
`q` and `mod1(q + 1, n)`.
"""
function monitored_setup(; d::Int, n::Int, seed::Int, T = DestabilizerTableau)
    n >= 2 && iseven(n) || throw(ArgumentError("n must be even and at least 2"))
    rng = Random.Xoshiro(seed)
    tab = T(d, n; state = :Z)
    bonds = [QuditClifford.random_clifford(rng, d, [q, mod1(q + 1, n)]) for q in 1:n]
    return tab, bonds, rng
end

"""
    monitored_layers!(tab, bonds, rng, layers; p = 0.25)

The hot loop of [`monitored_trajectory`](@ref): `layers` brickwork layers,
each refilling and applying the operator of every bond in its matching, then
measuring `Z` on each qudit with probability `p`. Every measurement gets an
outcome drawn from `rng` through the `outcome` keyword rather than
`measure!(rng, ...)`, which a baseline revision may lack. Allocates nothing
once warm.
"""
function monitored_layers!(tab, bonds, rng, layers::Int; p::Float64 = 0.25)
    n, d = tab.n, tab.d
    for layer in 1:layers
        for q in (isodd(layer) ? 1 : 2):2:n
            QuditClifford.apply!(tab, QuditClifford.random_clifford!(rng, bonds[q]))
        end
        for q in 1:n
            rand(rng) < p || continue
            QuditClifford.measure!(tab, QuditClifford.SinglePauli(q, 0, 1);
                                   outcome = rand(rng, 0:(d - 1)))
        end
    end
    return tab
end

"""
    monitored_trajectory(; d, n, layers = 4n, p = 0.25, seed, T = DestabilizerTableau)

One trajectory of a monitored brickwork circuit on a periodic chain of `n`
qudits: random two-qudit Cliffords on alternating bonds, each followed by
single-qudit `Z` measurements with probability `p`. One seeded `Xoshiro`
drives the gates, the measurement choices and their outcomes, so every
revision runs the identical circuit. Returns the final tableau, which is pure.
"""
function monitored_trajectory(; d::Int, n::Int, layers::Int = 4n, p::Float64 = 0.25,
                                seed::Int, T = DestabilizerTableau)
    tab, bonds, rng = monitored_setup(; d, n, seed, T)
    return monitored_layers!(tab, bonds, rng, layers; p)
end

# The PR-head script also runs against a baseline that may lack random
# sampling. Check for both APIs before building any fixture, so such a
# baseline keeps its whole column.
function register_random_group!(suite, types, ds, ns; ks = (1, 2, 8),
                                circuits::Bool = false,
                                api::Module = QuditClifford)
    all(name -> isdefined(api, name), (:random_clifford!, :random_state!)) || return suite
    n = first(ns)
    for d in ds
        for (key, leaf) in random_refill_group(d; ks)
            suite["random"][key] = leaf
        end
        let rng = Random.Xoshiro(20260925), targets = collect(1:8)
            suite["random"]["sample/d=$d/k=8"] = @benchmarkable(
                QuditClifford.random_clifford($rng, $d, $targets),
                setup = (Random.seed!($rng, 20260925)), evals = READONLY_EVALS)
        end
        for T in types, (key, leaf) in random_state_group(; d, n, T)
            suite["random"][key] = leaf
        end
    end
    # One short trajectory per tableau type in every profile, so each pull
    # request runs a whole circuit end to end.
    for T in types
        suite["circuit"]["monitored/$(nameof(T))/d=3/n=32/layers=64"] = @benchmarkable(
            monitored_trajectory(d = 3, n = 32, layers = 64, seed = 1234, T = $T), evals = 1)
    end
    if circuits
        for nn in (16, 32, 64)
            suite["circuit"]["monitored/d=2/n=$nn"] =
                @benchmarkable(monitored_trajectory(d = 2, n = $nn, seed = 1234), evals = 1)
        end
    end
    return suite
end
