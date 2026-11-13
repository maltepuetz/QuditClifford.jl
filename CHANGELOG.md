# Changelog

All notable changes to QuditClifford.jl are documented in this file.

Every pull request with a user-facing change adds an entry under `## Unreleased`.
A pull request that genuinely needs no entry — a typo fix, a CI-only change, a
dependency bump — carries the `skip-changelog` label instead.

## Unreleased

### Breaking

- `measure!` on qubits returns the outcome `t ∈ {0, 1}`, meaning eigenvalue
  `(-1)^t`, in every branch. A deterministic measurement with `storephase=true`
  used to return the `i`-exponent instead, so eigenvalue `-1` came back as `1`
  from a random measurement and as `2` from the same measurement repeated.
  Code that compared a deterministic qubit result with `2` must compare with
  `1`. Odd `d` is unchanged, `expect_int!` still returns the `i`-exponent
  (`2t` for a Hermitian operator), and with `storephase=false` a deterministic
  measurement still returns `0`.
- A deterministic qubit `measure!` whose operator has eigenvalue `±i` on the
  state throws an `ArgumentError` instead of returning `1` or `3`. This needs a
  non-Hermitian operator kept by `phase_policy` `0` or `2`, or a non-Hermitian
  stored generator; `phase_policy=1` and `expect_int!` are unaffected.
- `measure!` throws an `ArgumentError` for an `outcome` outside `0:(d-1)`.
  Such a value was used modulo `d` for the state but returned as given.

### Added

- `measure!(rng, tab, op; outcome, phase_policy)` draws the outcome of a
  random measurement from `rng`, for both a Pauli and a raw exponent vector, so
  one seeded RNG reproduces a monitored-circuit trajectory. It follows the
  rng-first convention of `random_clifford` and `random_state!`, allocates
  nothing, and draws before the tableau changes, so an exception from `rng`
  leaves the state as it was. `measure!(tab, op)` forwards with
  `Random.default_rng()`. `dephase!`, `traceout!` and `reset!` draw no random
  numbers and take no RNG.

### Changed

- `measure!` draws its outcome only when the measurement is random: when `op`
  anticommutes with a generator, or commutes with all of them without lying in
  their span. A deterministic measurement no longer advances the RNG, so code
  seeded with `Random.seed!` that mixes deterministic and random measurements
  produces a different outcome sequence than under 0.4.0. Each random draw is
  `rand(rng, 0:(d - 1))` as before. The `outcome` keyword defaults to `nothing`
  in place of an eager `rand(0:tab.d-1)`; an explicit `outcome` still replaces
  the draw.
- A precompile workload (PrecompileTools.jl, a new dependency) caches the
  common calls for both tableau types: a fresh session's first `measure!`,
  `apply!`, `canonicalize!`, entanglement and `random_clifford` calls no longer
  compile, cutting about 7 s of first-use latency to a few milliseconds, at
  about 40 ms more load time. Disable it with the `precompile_workload = false`
  preference for `QuditClifford`, as described in PrecompileTools' docs.
- Preset construction (`StabilizerTableau(d, n; state)`, `DestabilizerTableau`)
  zeroes the tableau once instead of twice: about 25–38 % faster at `n = 256`.

### Documentation

- The README shows a Zenodo DOI badge and gives the concept DOI
  `10.5281/zenodo.24170577`, which resolves to the latest archived release;
  each release also has its own version DOI. `CITATION.cff` carries the same
  concept DOI, so GitHub's "Cite this repository" entry includes it.
- The figures in the documentation are vector SVGs instead of PNGs, so they stay
  sharp at any zoom level. In the dark theme, axis ticks take the color of the
  spines; error-bar caps are drawn without an outline in both themes.
- The boundary-dephasing example writes the logarithmic negativity as
  ℰ (`\mathcal{E}`), as in the reference paper.
- The measurement-only Ising, purification and boundary-dephasing examples and
  the monitored-circuit manual example pass their seeded RNG to `measure!`.

### Fixed

- The raw-matrix constructors `StabilizerTableau(d, tableau)` and
  `DestabilizerTableau(d, tableau)` store a copy of `tableau`. A `Matrix{Int}`
  with `n` columns in the natural orientation used to become the tableau's
  storage itself: construction reduced the caller's matrix in place, later
  operations kept changing it, and two tableaux built from one matrix shared
  their state.
- With `check=true`, the raw-matrix constructors reject a non-Hermitian qubit
  generator (phase exponent `≢ x·z mod 2`, such as `i·Z`), which squares to
  `-I` and stabilizes nothing. `check=false` still accepts it.
- All tableau constructors reject a `PrecomputedInvMod` built for another
  dimension. It used to be read past its end, producing invalid tableau
  entries.
- `StabilizerTableau(d, zeros(Int, 0, 0); m=0, storephase=false)` and the
  `DestabilizerTableau` equivalent return an empty register; they used to hang.
- `∘` with a named gate as an operand, as in `U ∘ Phase(1)` or
  `Fourier(1) ∘ Phase(2)`, throws an `ArgumentError` that points to
  `CliffordOperator(g, d)`. It used to return a Base `ComposedFunction`, which
  `apply!` rejected only later with a `MethodError`.
- `measure!`, `expect!` and `expect_int!` validate their Pauli operand and throw
  an `ArgumentError` before changing the tableau, as `dephase!` and
  `conjugate` do. A sparse Pauli with a qudit index outside `1:n` used to
  throw a `BoundsError`, read out of bounds, or (index `n + 1` with
  `storephase=true`) silently measure a different operator and corrupt the
  tableau. A repeated qudit index, as in `DoublePauli(1, 0, 1, 1, 0, 1)`, gave
  a wrong result, and a `GeneralPauli` whose length is not `2n` was truncated
  or read out of bounds. `dephase!` also rejects repeated qudit indices now.

## 0.4.0 - 2026-11-12

### Breaking

- `==` and `hash` on tableaux compare the stabilizer state: two tableaux are
  equal when they have the same `d`, `n` and `storephase` and their generators
  span the same group with the same phases, whatever the generator basis and
  across the two tableau types. Destabilizers and `iscanonical` are ignored,
  and neither function modifies its arguments. Previously both fell back to
  object identity, so code that keeps tableaux in a `Set` or uses them as
  `Dict` keys now merges equal states, and a tableau mutated while it is a key
  is filed under its old hash.

### Added

- `reset!(tab, sites; basis=:Z)` resets individual qudits without recording an
  outcome, alongside the existing `reset!(tab; state)` for the whole register:
  it traces out `sites` and prepares each one in the eigenstate that
  `state=:product` gives it for the chosen basis, leaving the rest of the
  register mixed if it was entangled with them. `basis` is `:X`, `:Y` or `:Z`
  for every site, or one entry per site. It supports both tableau types, every
  prime `d` and `storephase=false`, draws no random numbers and allocates
  nothing. Invalid or repeated sites and an invalid `basis` throw an
  `ArgumentError` before `tab` is modified.
- `traceout!(tab, sites)` discards qudits, leaving them maximally mixed while
  keeping `n`; supported for both tableau types, every prime `d` and
  `storephase=false`, allocation-free; invalid or repeated sites throw an
  `ArgumentError`.
- `dephase!(tab, op)` measures a Pauli and forgets the outcome, applying the
  complete dephasing channel in `op`'s eigenbasis. It supports both tableau
  types, every prime `d`, `storephase=false` and sparse and dense Paulis, is
  deterministic and allocates nothing for an `AbstractPauli` (a raw exponent
  vector is wrapped in a `GeneralPauli` first, as in `measure!`). Paulis outside
  the register throw an `ArgumentError`.
- `entanglement_negativity(tab, A, B)` returns the logarithmic negativity
  `E(A:B) = log_d ‖ρ_ABᵀᴮ‖₁` between disjoint subsystems `A` and `B`, with
  every other qudit traced out, as an integer in log-`d` units;
  `entanglement_negativity(tab, A)` takes `B` to be the complement of `A`.
  Unlike mutual information it vanishes on classical correlations, so it
  measures entanglement in mixed states. It supports pure and mixed states,
  both tableau types, every prime `d` and `storephase=false`, runs in at most
  cubic time, allocates nothing and leaves the generators unchanged. Invalid,
  repeated or overlapping sites throw an `ArgumentError`.
- `mutual_information(tab, A, B)` returns `I(A:B) = S(A) + S(B) - S(AB)` and
  `tripartite_information(tab, A, B, C)` the tripartite information
  `I₃(A:B:C)`, in log-`d` units with every other qudit traced out;
  `mutual_information(tab, A)` takes `B` to be the complement of `A`. They
  agree with the corresponding sums of `entanglement_entropy` calls, but share
  the eliminations: on mixed states they are about twice as fast, on pure
  states as fast. Both support pure and mixed states, both tableau types,
  every prime `d` and `storephase=false`, allocate nothing and leave the
  generators unchanged. Invalid, repeated or overlapping sites throw an
  `ArgumentError`.
- `conditional_mutual_information(tab, A, B, C)` returns
  `I(A:B|C) = S(AC) + S(BC) - S(C) - S(ABC)` in log-`d` units, with every
  other qudit traced out. It agrees with the sum of four
  `entanglement_entropy` calls but takes one elimination, which makes it about
  three times as fast on mixed states and at least as fast on pure ones. It
  supports pure and mixed states, both tableau types, every prime `d` and
  `storephase=false`, allocates nothing and leaves the generators unchanged.
  Invalid, repeated or overlapping sites throw an `ArgumentError`.
- `copy(tab)` returns an independent `StabilizerTableau` or
  `DestabilizerTableau` with the same state and generator basis, and
  `copy!(dst, src)` overwrites an existing tableau of the same kind, `d`, `n`
  and `storephase` without allocating, for branching trajectories off one
  state. Previously `copy` threw a `MethodError`.

### Documentation

- `CITATION.cff` holds the citation metadata: title, author with ORCID and
  affiliation, license and keywords. GitHub turns it into a "Cite this
  repository" entry, and Zenodo reads it when archiving a release.
- The documentation is organized into Getting Started, a Manual chapter
  (`docs/src/manual/`), an Examples chapter with one page per example
  (`docs/src/examples/`) and the API reference. Links to the old
  `/examples`, `/conventions`, `/measurements`, `/unitaries` and
  `/dense-states` pages move under `/examples/…` and `/manual/…`.
- A manual section on `dephase!`, `traceout!` and `reset!` explains the
  difference between recording a measurement outcome and forgetting it, with a
  qutrit Bell pair example.
- An example reproduces the open monitored circuit of Weinstein, Bao and
  Altman (PRL 129, 080501 (2022)): random Clifford gates and `Z` measurements
  on a chain whose two ends are dephased with `dephase!`. A small run during
  the docs build shows the three regimes for qubits and qutrits, and stored
  averages over the paper's sizes (`L = 40` to `280`, 200 trajectories),
  generated by `docs/data/boundary-dephasing/reproduce.jl`, reproduce its
  `L^{1/3}` growth of the negativity and locate the qutrit transition near
  `p ≈ 0.33`.
- The figures in the documentation follow the light/dark theme toggle: each is
  rendered once as before and once restyled for the dark theme.
- The glow behind the logo on the documentation's home page is a dithered
  image per theme, generated by `docs/hero_glow.jl`, with one soft blob behind
  each dot of the logo. Safari rendered the default blurred gradient
  differently on first load than after toggling the theme.

### Fixed

- `entanglement_entropy` checks that the subsystem holds distinct sites in
  `1:n` and throws an `ArgumentError` otherwise. An out-of-range site used to
  read out of bounds and return a wrong value or crash Julia, and a repeated
  site gave a wrong entropy. `check=false` skips the `O(n)` check.

## 0.3.2 - 2026-10-21

### Added

- `entanglement_profile(tab, order)` and `entanglement_profile!(dest, tab,
  order)` take the prefixes along a site `order`, a permutation of `1:n`:
  entry `l + 1` is the entropy of `order[1:l]`. Cyclic shifts give every arc
  from one starting site of a periodic chain, and other nested subsystems
  work the same way, at the cost of the natural-order profile and still
  without allocating. An invalid `order` throws before the destination is
  written.

### Changed

- On a pure tableau with `storephase=false`, `measure!`, `expect_int!` and
  `expect!` settle a commuting Pauli from the commutation check alone, in
  `O(km)` for a sparse Pauli on `k` qudits (`O(nm)` for a dense Pauli),
  instead of decomposing it over the generators. A
  `StabilizerTableau` no longer canonicalizes on such deterministic outcomes,
  which dominated monitored circuits at large `n` (#26). Outcomes and the
  stabilizer group are unchanged, but the generator basis and `iscanonical`
  are now left as they were.

### Fixed

- `measure!` reduces a Pauli's exponents modulo `d` before taking
  commutators with the generators, and reduces its phase modulo `d`
  (modulo 4 for qubits) before outcome arithmetic. A sparse Pauli or
  `GeneralPauli` with huge exponents, such as `typemin(Int)`, could overflow
  there for odd `d` and take
  the wrong branch: a commuting Pauli was measured as noncommuting and
  replaced a generator, a `DestabilizerTableau` could build a wrong
  destabilizer or throw on an appended generator, and the stored or returned
  phase could be wrong.

## 0.3.1 - 2026-10-20

### Added

- `entanglement_profile(tab)` returns the entropies of every prefix of a
  qudit chain, including the empty and full prefixes, in log-`d` units.
  A single right-endpoint elimination computes the profile in at most cubic
  time instead of the quartic worst case of repeated `entanglement_entropy`
  calls. It supports pure and mixed states, both tableau types, and
  `storephase=false`. Use `entanglement_profile!(dest, tab)` with a reusable
  `Int` vector of length `n + 1` to avoid allocating the result on each call.

## 0.3.0 - 2026-09-30

### Added

- Clifford unitaries: `apply!(tab, gate)` applies a named Clifford to a
  stabilizer or destabilizer tableau in place, and `conjugate(gate, pauli[, n];
  d)` returns `U P U†` (`n` is required for a sparse Pauli and inferred for a
  `GeneralPauli`). The gate set is qudit-native — `Fourier`, `Phase`,
  `Multiplier`, `PauliGate`, `SUM`, `CPhase`, `SWAP` — with no qubit aliases;
  `docs/src/unitaries.md` gives the `d = 2` translations.
- `CliffordOperator(d, targets, F, a; check)` stores an arbitrary Clifford on an
  ordered support at a fixed dimension, and `CliffordOperator(g, d)` materializes
  a named gate. Stored operators work with `apply!` and `conjugate` exactly as
  named gates do, compose with `∘` (right operand first) and invert with `inv`,
  so Heisenberg evolution is `conjugate(inv(U), op)`. Construction validates the
  symplectic condition and, at `d = 2`, generator-image Hermiticity; `check=false`
  skips only those two algebraic checks. Warmed stored `apply!` allocates nothing.
- Uniformly random Clifford operators and stabilizer states.
  `random_clifford([rng,] d, targets)` returns a `CliffordOperator` drawn
  uniformly from the Clifford group modulo global phase, and
  `random_clifford!([rng,] U)` refills an existing operator in place, allocating
  nothing once warm. `random_state!([rng,] tab; m = tab.n)` replaces a tableau's
  state with a uniformly random stabilizer state with `m` generators: pure at
  `m = n`, the normalized projector onto a random stabilizer code for `m < n`.
  The RNG comes first and defaults to `Random.default_rng()`.
- Tools for inspecting and debugging small systems, not for computation.
  `ket(tab)` returns a `StabilizerKet`, the exact computational-basis expansion
  of a pure state: `d^k` terms with integer phase exponents, printed in Dirac
  notation such as `(|00⟩ + |11⟩)/√2`, and stored as digits, so a 100-qutrit
  GHZ state is three terms. `state_vector` and `density_matrix` return dense
  `ComplexF64` arrays, the latter for mixed states too, with qudit 1 as the most
  significant digit so that product states agree with `kron`. All three leave
  the tableau untouched, require `storephase=true`, and refuse a result larger
  than their `maxentries` budget (`2^24` by default) without ever forming
  `d^n`. `docs/src/dense-states.md` ("Inspecting States") walks through them.

### Documentation

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
