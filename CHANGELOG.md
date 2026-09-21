# Changelog

All notable changes to QuditClifford.jl are documented in this file.

Every pull request with a user-facing change adds an entry under `## Unreleased`.
A pull request that genuinely needs no entry — a typo fix, a CI-only change, a
dependency bump — carries the `skip-changelog` label instead.

## Unreleased

### Added

- Clifford unitaries: `apply!(tab, gate)` applies a named Clifford to a
  stabilizer or destabilizer tableau in place, and `conjugate(gate, pauli[, n];
  d)` returns `U P U†` (`n` is required for a sparse Pauli and inferred for a
  `GeneralPauli`). The gate set is qudit-native — `Fourier`, `Phase`,
  `Multiplier`, `PauliGate`, `SUM`, `CPhase`, `SWAP` — with no qubit aliases;
  `docs/src/unitaries.md` gives the `d = 2` translations. General Clifford
  operators, composition, inversion and uniform random sampling are not yet
  included.
- Logo and favicon for the documentation site — three dots in the Julia colours
  at the cube roots of unity, the three levels of a qutrit, swept by an arrow for
  the cyclic shift. Each ships as a light/dark pair and follows the reader's
  theme.

## 0.2.0 - 2026-09-15

### Breaking

- The raw-matrix constructors `StabilizerTableau(d, tableau)` and
  `DestabilizerTableau(d, tableau)` now enforce the generator contract: the
  active columns must pairwise commute and be independent. A matrix violating
  either throws, where 0.1.0 silently built a tableau that `measure!` then
  decomposed against as if it were a basis. Pass `check=false` to skip the
  validation, which is O(n³) and takes raw-matrix construction at n = 256 from
  0.135 ms to 2.69 ms. The preset `(d, n)` constructors satisfy the contract by
  construction, take no such keyword, and are unaffected.
- `is_pure(tab)` now decides from `m == n` alone, in O(1), instead of
  re-deriving commutation and independence on every call (4.87 ms at n = 256).
  The full derivation remains available as `is_pure(tab; verify=true)`, which is
  also the only form that can see through `check=false`.

### Added

- `check` keyword on the raw-matrix constructors and `verify` keyword on
  `is_pure`, as described above.
- `benchmark/` — a BenchmarkTools suite with `smoke`/`ci`/`full` profiles
  selected by `QC_BENCH_PROFILE`, tests for the suite's own invariants, and a
  GitHub Actions job running AirspeedVelocity on pull requests that touch
  `src/`.
- `docs/src/conventions.md` — the modular-arithmetic preconditions and the
  generator contract, written down for contributors.

### Performance

- The tableau inner loops go through branch-free modular primitives
  (`submul_mod!`, `addmul_mod!`, `mulcopy_mod!`, `scale_mod!`), which resolve an
  arithmetic tier once per call instead of dividing once per element.
  `canonicalize!` is 98.8% of the benchmark suite's wall-clock and its inner
  loop was exactly that shape; mid-circuit leaves at d = 5 measure 2.5–2.76×.
- The commutation check behind `is_pure(; verify=true)` reads the entire
  pairwise symplectic form off a single Gram matrix instead of scanning pairs —
  about 2× across n = 64…512.
- Preset tableaux and `reset!` are built in closed form rather than by generic
  reduction, and a preset `DestabilizerTableau` no longer runs the generic dual
  rebuild for duals that are known analytically. At n = 256, `StabilizerTableau`
  `:ghz` construction goes 207 → 13.5 µs and `DestabilizerTableau` `:ghz`
  52.6 → 0.051 ms; `reset!` goes 102 → 5.2 µs and 1255 → 10.3 µs respectively,
  and remains allocation-free.
- `rank_fp_cols!`, which backs `entanglement_entropy`, eliminates forwards only
  instead of running a full Gauss-Jordan sweep: 1.8× on a mid-circuit state and
  118× on `:ghz`, where the leftward pass had been filling in an otherwise
  sparse matrix.
- The dual re-orthogonalization in the non-commuting `measure!` branch skips
  zero entries.
- A `DestabilizerTableau` no longer carries the dual-rebuild scratch space as
  fields, since the rebuild runs at most once per tableau and never on the
  preset path. Its footprint drops from 2.7–3.0× a `StabilizerTableau` to
  1.4–1.5× — 6.32 → 3.17 MB at n = 256.

### Documentation

- Expanded `DestabilizerTableau` docstring, with smaller corrections to
  `canonicalize!`, `expect!` and `StabilizerTableau`.

## 0.1.0 - 2026-07-16

- Initial public release.
- Stabilizer and destabilizer tableaux for prime-dimensional qudits.
- Pure and mixed-state initialization, projective Pauli measurements,
  expectation values, canonicalization, purity checks, and entanglement entropy.
- Continuous integration, API documentation, and automated release tagging.
