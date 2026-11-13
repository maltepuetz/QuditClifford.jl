# Benchmarks

A regression suite for the operations whose cost actually matters, wired into
CI so that every pull request touching `src/` gets an A/B table against `main`.

## What runs where

`.github/workflows/Benchmark.yml` runs
[AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl) on
pull requests that touch `src/**`, `benchmark/**.jl`, or either `Project.toml`.
It benchmarks the PR head **and** `main` in the same runner job and writes a
comparison to the run's job summary. Nothing is stored between runs; every job
is a self-contained two-point comparison.

Docs-only pull requests are skipped on purpose.

The same workflow has a second job, `suite self-test`, which runs
`benchmark/test_workloads.jl` and then one sample of every `smoke` leaf.
`@benchmarkable` bodies are quoted, so the A/B job can pass while a leaf
measures a different branch than its name says; the self-test is what catches
that. It needs no baseline revision and runs in parallel with the comparison.

## Reading the table

The ratio column is **base / head, so a ratio above 1 means the branch is
faster.** This is the single most misread thing in the output.

| ratio | reading |
| --- | --- |
| 0.91 – 1.10 | noise. Ignore. |
| 1.10 – 1.25 or 0.80 – 0.91 | suggestive; reproduce locally before believing it |
| > 1.25 or < 0.80 | real |
| any memory / allocation change | real — allocation counts are deterministic |

Reported values are **medians**, not minima.

### What the noise actually looks like

These numbers are measured, not estimated, in a null experiment: a run whose
two revisions have identical `src/`, so all 108 rows have a true ratio of
exactly 1.00. On `ubuntu-latest`:

- median and mean ratio **1.000 / 0.999** — no systematic bias between the two
  revisions
- **105 of 108** rows inside 0.91 – 1.10, 99 inside ±5%
- worst rows **0.738** and **1.140**

So the band above holds for about 97% of rows — but three rows escaped it in a
run where nothing had changed. **Never conclude anything from a single row.**

### The error bars do not bound the noise

Each cell renders as `median ± interquartile range`. It is tempting to dismiss
any ratio whose deviation from 1 is smaller than that `±`. That rule is wrong,
and the null run shows why:
**39 of the 108 rows deviated from 1.00 by more than their own error bar.**

The error bar measures how precisely each revision was measured *within* its
own run, and it is tight — a median of 0.7% for kernels above 1 ms and under
2% for the rest. The scatter between the two revisions is five to ten times
larger. The two revisions are separate builds measured minutes apart, so they
differ in code layout, allocation addresses and CPU state, and none of that
shrinks with more samples. Collecting more samples tightens the `±` and leaves
the ratio where it was.

A small error bar therefore means the measurement is trustworthy, not that the
ratio is. Use it in one direction only: a ratio whose `±` is *larger* than its
deviation is definitely noise (this catches the very cheapest kernels, where
`evals = 1` puts them near the timer floor).

### How to read it instead

- **Judge groups, not rows.** A real regression in a shared routine moves every
  cell that reaches it. If all four `pauli/measure!/*` rows move together, or
  both `d = 2` and `d = 3` at the same `n`, that is signal. One row moving
  alone, with its neighbours flat, is noise — that is exactly the pattern the
  null run produced.
- **Read the memory table first.** Allocation counts are deterministic, so any
  change there is real and points at where to look in the time table.
- **Reproduce locally before believing a time ratio.** A local A/B removes the
  two-runner variance, but it is still two builds measured at different moments
  on a machine doing other things, so treat a single local ratio inside the
  noise band the same way. Run it both ways round: if the ordering changes the
  sign of a small ratio, it was noise.

## Profiles

`QC_BENCH_PROFILE` selects the size sweep. Default `ci`.

| profile | sizes | dimensions | leaves | ~time/revision | used by |
| --- | --- | --- | --- | --- | --- |
| `smoke` | n = 8 | 2, 3 | 284 | ~40 s | local sanity check; one sample of each leaf in the self-test job |
| `ci` | n ∈ {64, 256} | 2, 3, 5 | 477 | ~170 s (measured locally: 145 s of samples, 18 s warmup, the rest suite construction) | the pull-request job |
| `full` | n ∈ {64, 256, 512}; entropy profiles also n = 128 | 2, 3, 5, 7 | 999 | ~25 min (budget sum; not measured directly) | `workflow_dispatch`; adds the Ising, purification and monitored random circuits |

The `dense`, `channels`, `information`, `gates` and `tableau` groups use the
largest `n` of the profile and the `modulus` group `n = 64` (`8` in `smoke`).

`d = 5` is in `ci` rather than only in `full` because it is the smallest prime
that reaches the Barrett tier in `src/modular.jl`; `d = 2` and `d = 3` take the
multiply-free tiers exclusively. Without a `d = 5` row, no pull-request
benchmark exercises that tier, and a Barrett guard that began rejecting every
`d` would show up as no change rather than as a regression.

Only leaves built on a state with general residues reach it. Every `micro` leaf
starts from a constructor, whose tableau entries are just `0`, `1` and `d-1`, so
its multipliers are always `1` or `d-1` and take the multiply-free tiers at every
`d` -- the `:ghz` `canonicalize!` rows make zero Barrett calls even at `d = 131`.
The `midcircuit` and `dense` leaves at `d = 5` and the `modulus` leaves at
`d = 443` hold general residues and do reach it.

The divide-twice fallback is reached by the `probe/fallback-prime` rows and by
the `modulus` leaves at `d = 131` and `d = 1031`. `d = 131` is rejected by the
Barrett guard and `d = 1031` lies above its table; either is necessary but not
sufficient, for the same reason as above. `n = 16` in the fallback probe is
there because the fallback's per-call tier dispatch is a fixed cost that only
shows on short columns. `test_workloads.jl` counts the tier every `dense` and
`modulus` leaf reaches, and replays the fallback probe's elimination.

## Coverage

The phase-free deterministic measurement probes use `Z_i Z_j^-1` on a pure GHZ
state for both tableau types. Each sample restores the noncanonical tableau,
so unnecessary canonicalization cannot be hidden by an earlier sample (#26).

The `entropy_profile` group compares `entanglement_profile` and
`entanglement_profile!` with a loop over `entanglement_entropy(tab, 1:l)` for
every `l = 0:n`. All three leaves use the same seeded, uniformly random qutrit
state, built outside the timed body, for both tableau types and `m = n ÷ 2`
(mixed) or `m = n` (pure). The `full` profile
sweeps `n = 64, 128, 256, 512`. On a baseline without `entanglement_profile`,
the loop still runs; the in-place leaf is also omitted when its API is absent.
A baseline without `random_state!` skips this group. The in-place leaf reuses
an output vector allocated outside the timed body, so it should allocate zero
bytes after compilation.

Within one revision, divide the loop median by the profile median to measure
the speedup from computing all cuts together. This differs from the usual
base/head ratio, which compares the same leaf across revisions.

The spine is eight operations × {`StabilizerTableau`, `DestabilizerTableau`} ×
{d = 2, d = 3, d = 5} × two sizes: `construct`, the three `measure!` branches
(non-commuting, deterministic, append — separated because their costs differ
substantially and a blended workload hides which dominates), `expect!`,
`canonicalize!`, `entropy/half`, and `reset!`.

The `measure!` and `expect!` leaves use a weight-2 operator on qudits `n/4` and
`3n/4` rather than something on qudit 1. Qudit 1 is a soft best case — first
column, first pivot, and the smallest possible support for the destabilizer's
sparse dual update — and moving off it costs about 1.8× more work per call,
which also lifts the cheapest leaves further off the timer floor. `spread_x`
(X⊗X, non-commuting on a Z-basis product state) and `spread_z` (Z⊗Z, a product
of two generators and so in span) keep each leaf on the branch it is named
for; `test_workloads.jl` asserts that for every n and d in the profiles.

Everything in the spine starts from a freshly constructed tableau. That is the
right setup for `construct` and `reset!`, and the worst case for
`canonicalize!`, but it is not the regime most calls happen in. A separate
`midcircuit` group repeats the state-sensitive operations — both `measure!`
branches, `canonicalize!`, `expect!` and `entropy/half` — on a state put
through four seeded brickwork layers of random two-site measurements. The gap
is not cosmetic — at n = 256, d = 3 on a `DestabilizerTableau`,
`measure!/deterministic` goes 2.4 µs → 77 µs, and `expect!` goes 1.5 µs → 2.0 µs
for the local operator and 39 µs for a dense in-span one. `canonicalize!` goes
the other way, 8.4 ms → 0.49 ms, because the fresh `:ghz` state is its *worst*
case. `entropy/half` costs about the same on both (84 µs and 85 µs).

A `dense` group repeats the state-sensitive operations on a uniformly random
(volume-law) state from `random_state!`, at the largest `n` of the profile, for
every `d` and both tableau types. Most entries of such a tableau are general
residues, `spread_x` anticommutes with a large share of the generators, and an
in-span operator decomposes into many of them, so these leaves reach the parts
of each kernel the constructor and mid-circuit states leave idle:

- `measure!/noncommuting`: `spread_x`, rewriting most generator columns.
- `measure!/deterministic` and `expect!`: a product of `n/2` generators with
  seeded powers (`span_product`), so the decomposition has `n/2` terms
  instead of one or two.
- `measure!/append`: the `m = n/2` state and a Pauli that commutes with all of
  its generators but lies outside their span (`commuting_out_of_span`), so `m`
  goes from `n/2` to `n/2 + 1` and the dual-basis loop runs over `n/2`
  generators. The spine's `append` leaf starts from `m = 0`, where that loop
  runs zero times.
- `canonicalize!` of the pure state.

Both tableau types hold the identical state: the `StabilizerTableau` fixture is
built from the generator matrix of the `DestabilizerTableau` one.
`test_workloads.jl` asserts the branch, the number of decomposition terms and
the change in `m` for every leaf.

A `modulus` group runs `measure!` (both branches), `expect!`, `entropy/half`
and `canonicalize!` on the same kind of dense state at `d ∈ {131, 443, 1031}`
and `n = 64`, both tableau types: one prime per tier described under Profiles.

`channels` times `dephase!`, `traceout!` of one site and of a quarter of the
chain, and the single-site `reset!` on a pure dense state (`d = 3`), each from
a restored snapshot. `information` times `mutual_information`,
`tripartite_information`, `conditional_mutual_information` and
`entanglement_negativity` on adjacent quarters of a dense state, pure and at
`m = n/2`; none of them changes the state, so they need no restore.

`gates` covers the named gates the `clifford` spine leaves out --
`Multiplier`, `PauliGate`, `CPhase`, `SWAP` and `SUM` with coefficient `2` --
at `d = 5`, where `2` is neither `1` nor `d-1`, plus `Fourier` and `SUM` on a
`storephase = false` tableau. `tableau` covers `copy`, `copy!`, `==` in the
same and in another generator basis, `hash`, the raw-matrix constructor with
and without `check`, and `measure!(rng, ...)` with the outcome drawn from an
RNG that is reseeded with every snapshot restore. All of these sit at the
largest `n` of the profile only.

A separate `clifford` group benchmarks `apply!` itself, as its own top-level
group alongside the spine and `midcircuit` above: `Fourier` and `Phase` (one
target) and `SUM` (two non-adjacent targets, the same `spread_sites` qudits
the `measure!` leaves use) applied to a freshly restored `:ghz` state, across
both tableau types and every `d` and `n` in the profile — three named-gate
leaves per `(type, d, n)` cell. `storephase = true` throughout, so both the
`d = 2` bitmask phase evaluator and the odd-prime one run: `Fourier` has zero
image `x·z`, so its phase term is a pure cross-term, while `Phase` has a
nonzero one and exercises the `D` vector. Gate application costs `O(k²·m)`
for `k ≤ 2` targets — linear in the generator count rather than
branch-sensitive the way `measure!` is — so one leaf per gate is enough;
there is no separate deterministic/append/non-commuting split here.

Each `(type, d, n)` cell also gets a `stored/...` sibling exercising the same
`apply!` path through a stored `CliffordOperator`, at support sizes `k` set by
`CONFIG.stored_ks` (clipped to `k ≤ n`): `{1, 2, 8}` in `smoke` and `ci`, plus
`32` and `128` in `full`.

The base leaf, `"apply!/stored/k=$k"`, applies to a `scrambled_tableau`: a
pure state reached by seeded rounds of `apply!`-ing named gates (`Fourier` or
`Phase` on every qudit, then a `SUM` to a random partner), rather than by a
constructor. Its target rows are therefore dense and mostly nonzero, unlike a
freshly built `:product` or `:mixed` tableau's mostly-zero, mostly-one-hot
rows. That distinction is not cosmetic: the dense matvec behind a stored
operator's `apply!` (`_matvec_prepared!`, `src/apply_clifford.jl`) skips a zero
column outright above `S = 2k = 24`, so a `:product`-state leaf at `k = 32` or
`128` would measure that sparse fast path rather than the general case a
scrambled circuit actually produces. The `full` profile only
(`CONFIG.stored_product`) therefore keeps a second, `"apply!/stored/product/k=$k"`
leaf on a freshly restored `:product` state — deliberately the sparse input,
so the fast path itself stays under coverage rather than going untested. A
third leaf, `"apply!/stored/m=0/k=$k"`, applies to the maximally mixed state,
making the validation-path cost `apply!` pays before any generator exists
visible on its own. A single such call is far below the timer floor, so this
leaf runs many evaluations per sample instead of the usual snapshot restore —
safe because `apply!` at `m = 0` is idempotent (the column loops run zero
times, and only `iscanonical` changes).

`CliffordOperator` construction, checked and unchecked, is timed separately
again: it takes no tableau, so it depends only on `d` and `k`, never on the
type or `n`, and is registered once per `d` rather than once per
`(type, d, n)` cell. The algebra operations — `conjugate` (against a seeded,
dense `GeneralPauli` observable, not a single sparse generator), `inv`, and
composition (`"compose/k=$k"`, `U ∘ U`) — are timed the same way, once per
`(d, k)` in a `stored/algebra/d=$d` group, since none of them touch a tableau
either.

One more cell, `stored/safe/d=1000000007` (`n = k = 8`), is registered once,
regardless of profile and outside any `d` in `CONFIG.ds`. Every `d` elsewhere
in the suite is small enough that a `2k`-term Clifford dot
(`clifford_fast_dots`, `src/clifford_arithmetic.jl`) takes the unguarded `Int`
fast path; `d = 1000000007` is chosen so that dot does not, while
`n(d-1)^2` still stays inside the tableau's own overflow envelope
(`max_safe_dimension`, `src/helper.jl`) — so this cell is the only place in
the suite that times the `add_mod`/`mul_mod`/`widemul` safe arithmetic tier
`src/clifford_arithmetic.jl` falls back to. It uses `JustInTimeInvMod`
throughout, since `PrecomputedInvMod`'s lookup table would be gigabytes at
this `d`.

A top-level `random` group times the samplers. `refill/d=$d/k=$k` refills a
warmed operator with `random_clifford!` at every `k` in `CONFIG.stored_ks`; a
refill reads nothing of the previous action, so repeated evaluations need no
restore, and it allocates nothing. `sample/d=$d/k=8` times the allocating
`random_clifford`. `state/$(nameof(T))/d=$d/n=$n/m=$m` replaces a tableau's
state with `random_state!` at `m = 1` and `m = n`, for the smallest `n` in the
profile only, because at `n = m = 256` one call takes about 45 ms, which a CI
leaf could time only two or three times. Both leaves allocate the same
`2n × 2n` scratch, so the gap between them is the `O(n²m)` sampling. The
`full` profile also adds `circuit/monitored/d=2/n=$n` for `n ∈ {16, 32, 64}`:
a monitored brickwork trajectory with one operator per bond, refilled every
layer, and `Z` measurements whose outcomes come from the same seeded RNG. Its
warm hot loop allocates nothing, which `test_workloads.jl` checks. Every
profile runs a short one, `circuit/monitored/$T/d=3/n=32/layers=64` for both
tableau types, so each pull request times a whole circuit end to end. The
`random` leaves reseed their RNG in `setup`, so every sample draws the same
operators and states.

The `full` profile's purification leaves run until the state is pure. With
seed 1234 at `d = 3` that takes 42 layers at `n = 16` and 1876 at `n = 32`;
at `n = 64` the state is still mixed after the 10 000-layer cap, so that leaf
is named `purification/d=3/n=64/capped` and times a fixed-length run.
`test_workloads.jl` pins all three.

Probes cover one axis at a time at a single representative configuration:
`storephase = false`, `JustInTimeInvMod`, Pauli sparsity (`SinglePauli` /
`DoublePauli` / `NPauli{8}` / dense `GeneralPauli`), `:ghz` construction,
`entropy/single_site`, `expect!/out_of_span`, and `is_pure`.

`is_pure` is a probe rather than part of the spine, and the leaf measures
`is_pure(tab; verify = true)`. The default `is_pure` tests `m == n` and nothing
else, so there is no kernel under it to track; the verifying form costs
`O(n^3)` (2.5 ms at n = 256) and is the only coverage the commutation Gram and
the rank elimination get.

## Running locally

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate(); Pkg.status()'
```

`Pkg.status()` must print a **path** next to `QuditClifford`, not a bare
version. A bare version means you are benchmarking the registered release
instead of your working tree — see the Julia 1.10 note below.

```bash
QC_BENCH_PROFILE=ci julia --project=benchmark -e '
    include("benchmark/benchmarks.jl"); using BenchmarkTools
    show(stdout, MIME"text/plain"(), run(SUITE))'
```

Sanity-check the suite itself (fast; this is what catches a renamed export,
because `@benchmarkable` bodies are quoted and merely including the file
proves nothing):

```bash
julia --project=benchmark benchmark/test_workloads.jl
```

## Reproducing the CI comparison locally

```bash
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("AirspeedVelocity"); Pkg.build("AirspeedVelocity")'
export PATH="$HOME/.julia/bin:$PATH"

git fetch origin main          # a stale local `main` gives a stale baseline
OUT=$(mktemp -d)
QC_BENCH_PROFILE=ci benchpkg QuditClifford --path=. --rev=main,dirty \
    --script=benchmark/benchmarks.jl --output-dir="$OUT"
benchpkgtable QuditClifford --input-dir="$OUT" --rev=main,dirty --mode=time --ratio
```

`--rev=main,dirty` measures your working tree, uncommitted changes included.
CI instead uses `--rev=main,<head sha>` and installs both revisions with
`Pkg.add`, so it never sees uncommitted work. Pass explicit SHAs to match CI
exactly.

## Things that will bite

- **Julia 1.10 ignores `[sources]` entirely.** `Pkg.instantiate()` will
  silently resolve `QuditClifford` from the General registry and you will
  benchmark the *released* package. On 1.10 you must
  `Pkg.develop(path=pwd())` first, then check `git diff benchmark/Project.toml`
  is clean. On ≥ 1.11 do *not* `Pkg.develop` into `benchmark/` — it rewrites
  the tracked file, possibly with a machine-specific absolute path. Either way,
  confirm with `Pkg.status()`.
- **The CI benchmark environment is only `QuditClifford` + `BenchmarkTools` +
  stdlibs.** Because the workflow passes `--script`, benchpkg never reads
  `benchmark/Project.toml`. Adding a dependency there will work locally and
  fail in CI. The escape hatches are the action's `extra-pkgs:` input, or
  switching the workflow from `script:` to `bench-on: <head sha>` (which costs
  an extra clone and resolve).
- **A branch that has not been rebased since the suite landed will fail the
  job**, because `--script` resolves against the head checkout.
- **`canonicalize!` must start from genuinely non-canonical data.** A
  `:product` tableau is already in RCEF, and `canonicalize!` has no
  `iscanonical` early return, so it would collapse to O(n·m) and report a
  fraction of the real 5.8 ms (`StabilizerTableau`) and 8.4 ms
  (`DestabilizerTableau`) at n = 256. The suite restores a `:ghz` snapshot
  before every sample; `test_workloads.jl` pins this.
- **`measure!` at d = 2 warns on every non-Hermitian Pauli** under the default
  `phase_policy`. Everything here passes `phase_policy = 1`, which measures the
  Hermitian form of the operator.
- **The cheapest mutating leaves sit near the timer floor.** Leaves that leave
  the state alone (`expect!`, construction, the `CliffordOperator` algebra, the
  samplers, `reset!`) run `READONLY_EVALS = 100` evaluations per sample, which
  lifts them well above the ~42 ns tick. The mutating ones cannot use
  `evals > 1` (running `measure!` a hundred times per sample measures a
  different thing — `m` grows and the state moves), so a few n = 64 `measure!`
  and `apply!` leaves stay at a few hundred ns. Their ratios carry a `±` larger
  than their deviation from 1, which is exactly the signal to ignore them; they
  are there for scaling context, not as primary evidence.
- **Fork pull requests may not resolve.** CI installs each revision with
  `Pkg.add(PackageSpec(..., rev=<sha>, url=<base clone url>))`; whether a
  fork's head SHA is fetchable that way is untested. Every PR in this
  repository so far has been same-repo or Dependabot.

## Why not stored history

An absolute-time series on GitHub-hosted runners is mostly jitter, and it needs
somewhere to live plus a baseline to keep valid across refactors. The
same-runner A/B answers the question that actually blocks a merge — "did this
change make anything slower" — with no stored state at all. Meaningful absolute
numbers need fixed hardware; that is a separate decision.
