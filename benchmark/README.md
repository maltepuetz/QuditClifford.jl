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

## Reading the table

The ratio column is **base / head, so a ratio above 1 means the branch is
faster.** This is the single most misread thing in the output.

| ratio | reading |
| --- | --- |
| 0.91 – 1.10 | noise. Ignore. |
| 1.10 – 1.25 or 0.80 – 0.91 | suggestive; reproduce locally before believing it |
| > 1.25 or < 0.80 | real |
| any memory / allocation change | real — allocation counts are deterministic |

10% rather than the 20–30% usually quoted for GitHub runners because both
revisions are measured back-to-back in the *same* job on the *same* host, which
removes between-host variance and leaves only within-run drift. Each cell also
renders as `median ± interquartile range`; a supplementary rule is to ignore any
ratio whose deviation from 1 is smaller than the `±` on either cell.

Reported values are **medians**, not minima.

## Profiles

`QC_BENCH_PROFILE` selects the size sweep. Default `ci`.

| profile | sizes | dimensions | leaves | ~time/revision | used by |
| --- | --- | --- | --- | --- | --- |
| `smoke` | n = 8 | 2, 3 | 75 | seconds | local sanity check |
| `ci` | n ∈ {64, 256} | 2, 3 | 107 | ~60 s | the pull-request job |
| `full` | n ∈ {64, 256, 512} | 2, 3, 5, 7 | 265 | ~6 min | `workflow_dispatch`; adds the Ising and purification circuits |

## Coverage

The spine is eight operations × {`StabilizerTableau`, `DestabilizerTableau`} ×
{d = 2, d = 3} × two sizes: `construct`, the three `measure!` branches
(non-commuting, deterministic, append — separated because their costs differ
substantially and a blended workload hides which dominates), `expect!`,
`canonicalize!`, `entropy/half`, and `reset!`.

Everything in the spine starts from a freshly constructed tableau. That is the
right setup for `construct` and `reset!`, and the worst case for
`canonicalize!`, but it is not the regime most calls happen in. A separate
`midcircuit` group repeats the state-sensitive operations — both `measure!`
branches, `canonicalize!`, `expect!` and `entropy/half` — on a state put
through four seeded brickwork layers of random two-site measurements. The gap
is not cosmetic: a mid-circuit `expect!` costs ~22× the fresh-`:product` one on
a `DestabilizerTableau`, and `entropy/half` on `:ghz` is ~26× a typical
circuit state (`:ghz` has one fully delocalised generator, which makes the
subsystem elimination dense).

Probes cover one axis at a time at a single representative configuration:
`storephase = false`, `JustInTimeInvMod`, Pauli sparsity (`SinglePauli` /
`DoublePauli` / `NPauli{8}` / dense `GeneralPauli`), `:ghz` construction,
`entropy/single_site`, `expect!/out_of_span`, and `is_pure`.

`is_pure` is a probe rather than part of the spine: it is peripheral to this
package and by far the most expensive kernel (~28 ms at n = 256, against ~5 ms
for the whole rest of the cell).

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
  `iscanonical` early return, so it would collapse to O(n·m) and report ~0.2 ms
  instead of the real 25–48 ms at n = 256. The suite restores a `:ghz` snapshot
  before every sample; `test_workloads.jl` pins this.
- **`measure!` at d = 2 warns on every non-Hermitian Pauli** under the default
  `phase_policy`. Everything here passes `phase_policy = 2`.
- **The cheapest n = 64 leaves sit near the timer floor.** A few are ~125 ns,
  and the mutating ones cannot use `evals > 1` (running `measure!` a hundred
  times per sample measures a different thing — `m` grows and the state moves).
  Their ratios carry a `±` larger than their deviation from 1, which is exactly
  the signal to ignore them; they are there for scaling context, not as primary
  evidence.
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
