# Changelog

All notable changes to QuditClifford.jl are documented in this file.

Every pull request with a user-facing change adds an entry under `## Unreleased`.
A pull request that genuinely needs no entry — a typo fix, a CI-only change, a
dependency bump — carries the `skip-changelog` label instead.

## Unreleased

### Changed

- The precompile workload covers packed storage (`eltype = Bits`) in both
  layouts, the layout switch, and `stabilizer_matrix`/`destabilizer_matrix`.
  The first calls on a packed tableau in a new session no longer compile
  (Julia 1.13, M3 Pro: first `measure!` 1.4 s, first `==` 0.4 s, first
  gate 70–90 ms, each now under 0.1 ms; the first keyword constructor call
  still compiles its keyword wrapper, about 10–15 ms), and neither does the
  first `stabilizer_matrix` (about 0.2 s, now 1 ms). Loading the package
  takes as long as before (0.25 s on Julia 1.13, 0.38–0.39 s on Julia 1.10);
  generating the package image takes about 8–10 s longer.
- Building a `DestabilizerTableau` from a raw matrix is faster, with
  identical destabilizers: at `d = 2` every storage type eliminates over
  GF(2) on packed rows, as packed storage already did (`n = 512`: 255 to
  4.8 ms for `UInt8`; `n = 1024`: 4.5 s to 24 ms), and at odd `d` the
  elimination runs along contiguous columns (`n = 512`, `d = 3`: 341 to
  82 ms for `UInt8`, 1.1 s to 48 ms for `Int`; `n = 1024`, `d = 5`: 6.3 to
  1.1 s). M3 Pro.
- The preset constructors no longer zero storage that is already zero
  (`DestabilizerTableau(2, 4096)`: 1.9 to 0.6 ms for `UInt8`, 0.34 to
  0.13 ms packed), and packed GHZ destabilizers are written a word at a time
  (`DestabilizerTableau(2, 4096; state = :ghz, eltype = Bits)` and its
  `reset!`: 6–16 to 0.2 ms).
- The raw-matrix constructors read their input entry by entry in its own
  integer type, and a transposed input through a lazy `transpose`, instead
  of converting it to a `Matrix{Int}` (and transposing it) first. A `UInt8`
  input of `n = 4096` qudits allocated about 550 MiB and now allocates
  38 MiB (12 MiB packed), and takes 19 instead of 32 ms. Entries outside the
  range of `Int`, such as a `BigInt` or a `UInt64` above `typemax(Int)`, are
  now reduced mod `d` instead of throwing an `InexactError`.
- `measure!` with an outcome fixed by the state, the append of a new
  generator, and `expect!`/`expect_int!` are faster on packed storage
  (`eltype = Bits`), with identical results, random draws and tableaux
  afterwards: the decomposition of the operand over the generators, which
  ran bit by bit, now works on 64 generators or qudits per word. For an
  operand that is a product of many generators (single-qubit `Z` and a
  dense Pauli on scrambled states, medians of three interleaved runs, M3
  Pro), a `DestabilizerTableau` takes 30–78 times less at `n = 256` and
  90–183 times less at `n = 1024`, in both layouts (a fixed `Z` outcome at
  `n = 1024`: 793 to 4.9 µs generator-major, 1050 to 11.6 µs qudit-major),
  and a canonical `StabilizerTableau` 9–43 and 13–188 times less. Appending
  a dense operand is 7–16 and 22–44 times faster, appending a single-qubit
  one 1.0–4.4 times. At `n = 16` the gains are 2.1–6.5 times (the dense
  append 1.6–2.3 times). These calls are now 70–118 times faster than
  `UInt8` storage for a `DestabilizerTableau` at `n = 1024` instead of up to
  2.4 times slower. In a monitored brickwork circuit at `n = 1024` with `Z`
  measurements at rate 0.16, where one measurement in seventeen has a
  fixed outcome and its operand is a product of one or two generators,
  such a measurement takes 2.2 instead of 7.2 µs on a qudit-major
  `DestabilizerTableau` and a layer 0.52 instead of 0.58 ms. A
  `StabilizerTableau` that is not canonical still solves for the
  coefficients on a copy in `measure!`, which dominates there (1.1–1.2
  times faster). The random branch is unchanged. A pure qudit-major packed
  tableau takes an operand that commutes with every generator as in the
  span without computing a residual, as every pure tableau already does
  without phases; results differ only for a tableau built with
  `check = false` whose `n` generators are not independent and commuting.
  Not measured on x86 or on 32-bit hosts.
- One-, two- and three-qudit gates on a qudit-major tableau are faster at
  the dimensions whose reduction multiplies in a wider type than the
  kernel's lane, with identical results: 1.5–1.8 times at
  `d = 127, 251, 257, 4099` and 1.03–1.19 times at `d = 5, 31, 61`
  (`n = 2048`, both tableau types, M3 Pro; `d = 5` on one qudit and `d = 61`
  on three keep their reduction and their speed, as do `d = 2` and `d = 3`).
  The reduction there uses an unsigned constant of the lane's width, with a
  product that NEON and AVX2 have a vector instruction for. Such a constant
  exists for every odd dimension a lane holds, so the dimensions of the
  32-bit lane that had no constant and took the per-generator path (the
  smallest is 11777, on three qudits) run in the kernel too. At `n ≤ 16` a
  gate at `d = 31` takes about 4 ns (up to 8%) longer. Not measured on x86
  or on 32-bit hosts.
- `==` and `hash` on a tableau that is not canonical are faster on packed
  and on qudit-major storage, with identical results: the canonical copy
  they compare is now made by the storage's own `canonicalize!` kernel.
  Packed (`eltype = Bits`) tableaux: `hash` 11–18 times and `==` 17–30
  times faster, with a seventh to an eighth of the allocation; unpacked
  qudit-major tableaux: `hash` 4.1–5.5 times and `==` 4.5–6.1 times faster,
  allocating 17–37% more for the padded copies (random states, d = 2 and 3,
  n = 512 and 1024, both tableau types, M3 Pro). Generator-major unpacked
  tableaux and canonical tableaux are unchanged. Not measured on x86 or on
  32-bit hosts.
- `entanglement_negativity` is 3.8–7.9 times faster on `UInt8` storage
  (the default for `3 ≤ d ≤ 251`), with identical results: the symplectic
  products between the basis elements now vectorize (d = 3, 31 and 251,
  n = 512 and 1024, half cut, both layouts, M3 Pro; 7.3–7.9 times at d = 3,
  3.8 times at d = 251). `UInt16` storage keeps its speed (d = 257). The
  products of both narrow types are now summed in unsigned integers wide
  enough to stay exact on 32-bit hosts, where an `Int` sum could wrap at
  large `n` and `d`.
  Not measured on x86 or on 32-bit hosts.
- `is_pure(tab; verify = true)` and the commutation check of the raw-matrix
  constructors (`check = true`) are 4.9–7.9 times faster on generator-major
  `UInt8` storage, with identical results, including which pair of
  generators an error names (d = 3 and 31, n = 512 and 1024, M3 Pro). The
  products of the commutation check now vectorize. Qudit-major and `UInt16`
  storage are unchanged. Not measured on x86 or on 32-bit hosts.

## 0.5.0 - 2026-11-24

### Breaking

- `StabilizerTableau` and `DestabilizerTableau` have four type parameters,
  `{IM, E, W, V}`: the inverse strategy, the element type of the unpacked
  workspaces, the block word type and the type of `phase` and `xdotz_cache`.
  They also have a new field, `word_workspace`, and on packed tableaux
  `phase` and `xdotz_cache` are bit planes (internal fields; read the state
  through `stabilizer_matrix`). A tableau with `eltype = Bits` also has
  `E = UInt8` (it is a `StabilizerTableau{IM, UInt8, UInt64}`), so a method on
  `StabilizerTableau{IM, UInt8}` receives packed tableaux too; tell them apart
  with `tableau_eltype(tab)`. Dispatch on `StabilizerTableau` or
  `DestabilizerTableau`, or use `tableau_eltype(tab)`, instead of naming the
  parameters; code that names them positionally needs the three new ones. A
  struct field or container declared with `StabilizerTableau{PrecomputedInvMod}`,
  such as `Vector{StabilizerTableau{PrecomputedInvMod}}`, still accepts every
  tableau. `StabilizerTableau{PrecomputedInvMod}` is no longer a concrete type.
  Use `typeof(tab)` (for example
  `StabilizerTableau{PrecomputedInvMod, UInt8, UInt8, Vector{Int}}`) where a
  concrete type matters.
- `StabilizerTableau` and `DestabilizerTableau` have two new fields: `layout`,
  which records the layout, and `coeff_workspace`, a coefficient workspace.
  Code that calls the inner constructors positionally, or that builds the
  structs with all fields, must pass both; the public constructors are
  unchanged.
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
- `measure!` draws its outcome only when the measurement is random: when `op`
  anticommutes with a generator, or commutes with all of them without lying in
  their span. A deterministic measurement no longer advances the RNG, so code
  seeded with `Random.seed!` that mixes deterministic and random measurements
  produces a different outcome sequence than under 0.4.0. Each random draw is
  `rand(rng, 0:(d - 1))` as before. The `outcome` keyword defaults to `nothing`
  in place of an eager `rand(0:tab.d-1)`; an explicit `outcome` still replaces
  the draw.
- `measure!` throws an `ArgumentError` for an `outcome` outside `0:(d-1)`.
  Such a value was used modulo `d` for the state but returned as given.
- `StabilizerTableau` and `DestabilizerTableau` no longer have the field `stab`,
  and `DestabilizerTableau` no longer has the field `destab`; accessing them
  throws an `ArgumentError`. Use `stabilizer_matrix(tab)` and
  `destabilizer_matrix(tab)`, which return the same generator-major matrices as
  copies. Code that wrote into `tab.stab` should build the tableau from a matrix
  with `StabilizerTableau(d, S)` or `DestabilizerTableau(d, S)` instead.
- Tableaux store their generators in `UInt8` for `d ≤ 251` and `UInt16` for
  `d ≤ 65521` by default, and in `Int` above. Every result is the same `Int`
  value as before; `eltype = Int` restores the previous storage, but not its
  speed everywhere: rank-based functions and the deterministic `measure!` on
  a non-canonical `StabilizerTableau` eliminate in the storage element type,
  so with `eltype = Int` they are slower than before at `d ≤ 251` (measured
  at `d = 3` and `31`: entropies 4.6–8.9×, the deterministic `measure!`
  2.3–6.1×). `copy!` between tableaux of different element types throws an
  `ArgumentError`.
- `show` of a tableau prints a `Storage:` line with the layout
  (`generator-major` or `qudit-major`) and the element type after the
  generator count.
- The internal field `workspace` is gone; a tableau holds one rank workspace
  of its storage element type instead.

### Added

- `eltype = Bits` stores the X and Z exponents of a `StabilizerTableau` or
  `DestabilizerTableau` as packed bits at `d = 2`, in either layout (any other
  `d` throws an `ArgumentError`). Results are bit-identical to `UInt8`
  storage, and the switch between layouts stays in place and allocation-free.
  `Bits` is exported, `tableau_eltype(tab)` returns the storage type chosen at
  construction, and `UInt8` remains the default. A packed tableau is about
  4–6 times smaller than a `UInt8` one from `n = 1024` on. Gates,
  measurements, `canonicalize!` and the entanglement measures are mostly
  faster than with `UInt8`, qudit-major gates by an order of magnitude. The
  main paths that are slower than with `UInt8` in the same layout are
  `stabilizer_matrix` in both layouts, single-qubit gates on a generator-major
  tableau, four-qubit `CliffordOperator` gates on a qudit-major tableau, and
  `expect!` and `expect_int!` on a generator-major `DestabilizerTableau`. See
  "Packed qubit storage" in the manual and #62 for the measurements.
- Both tableau types can store their generators qudit-major, with the
  exponents of one qudit across all generators contiguous, instead of the
  default generator-major layout. The `layout = :generator | :qudit` keyword of
  every constructor selects it, `qudit_major!(tab)` and `generator_major!(tab)`
  switch a tableau in place without allocating, and `tableau_layout(tab)`
  returns `:generator` or `:qudit`. Results are bit-identical in both layouts,
  `copy!(dst, src)` gives `dst` the layout of `src`, and no operation leaves
  the layout changed.
- `stabilizer_matrix(tab; layout)` and `destabilizer_matrix(tab; layout)` return a
  tableau's generators as a fresh `Int` matrix, generator-major (generators in
  columns) or qudit-major.
- `measure!(rng, tab, op; outcome, phase_policy)` draws the outcome of a
  random measurement from `rng`, for both a Pauli and a raw exponent vector, so
  one seeded RNG reproduces a monitored-circuit trajectory. It follows the
  rng-first convention of `random_clifford` and `random_state!`, allocates
  nothing, and draws before the tableau changes, so an exception from `rng`
  leaves the state as it was. `measure!(tab, op)` forwards with
  `Random.default_rng()`. `dephase!`, `traceout!` and `reset!` draw no random
  numbers and take no RNG.
- The `eltype` keyword of every `StabilizerTableau` and `DestabilizerTableau`
  constructor selects the element type of the stored X and Z exponents:
  `UInt8` (`d ≤ 251`), `UInt16` (`d ≤ 65521`) or `Int` (any `d`). A type too
  narrow for `d`, or any other type, throws an `ArgumentError`. The precompile
  workload covers the default types, so the first calls on a tableau built
  with `eltype = Int` or `eltype = Bits` compile.

### Changed

- Gates and random-outcome measurements are faster on a qudit-major tableau:
  gates 4–16 times and random-outcome measurements 1.05–2.7 times at
  n = 1023–2049 (M3 Pro, `UInt8`). `canonicalize!`, `dephase!`, `traceout!`
  and `reset!` have qudit-major kernels and are about as fast or faster
  qudit-major. `expect!` and `expect_int!` (up to about 4 times on a
  `DestabilizerTableau`) and `stabilizer_matrix` and `destabilizer_matrix`
  (1.5–3 times at `d = 3`, `n = 1024`–`2048`) are slower qudit-major;
  generator-major stays the default. Timings are in #61.
- A random-outcome `measure!` is faster, with identical tableaux, outcomes,
  RNG draws and no allocations. Each generator that anticommutes with the
  measured operator absorbs a power of the replaced generator. That update
  runs only over the replaced generator's own sites when they are at most half
  of the register, which is the usual case in monitored circuits.
  Everything that depends only on the replaced generator is computed once per
  measurement. A `DestabilizerTableau` updates its cached `x·z` values and
  re-orthogonalizes its duals on the same sites, and the commutation scan
  skips the entries a pure `Z` or `X` operator does not need. This holds for
  every Pauli type. On states after a monitored circuit (`p = 0.05` and
  `0.15`), single-site `Z` and `X` and two-site `ZZ` and `XX` measurements are
  1.7–7× faster at `n = 250`–`2000`; at `n = 2000` a `Z` measurement takes
  23 µs instead of 63 µs at `d = 2` and 0.46 ms instead of 2.1 ms at `d = 31`
  (`StabilizerTableau`, `p = 0.15`). One whole monitored layer of a
  `DestabilizerTableau` runs 1.2–4.4× faster. A `StabilizerTableau` layer
  gains 2–4× at `d = 31`, but at `d = 2` and `3` it is dominated by its
  deterministic measurements, which canonicalize and are unchanged. On random
  states, where the replaced generator covers most sites, the gain is up to
  1.5×, and otherwise within noise.
- A `measure!` whose operator commutes with every generator is faster, with
  identical outcomes, errors and RNG draws and no allocations. On a
  `StabilizerTableau` that is not canonical it no longer canonicalizes: it
  solves for the operator's coefficients over the generators by elimination on
  a copy, one bit per entry at `d = 2` and one entry of the storage element
  type otherwise, and leaves the generator basis and `iscanonical` as they
  were. A fixed outcome therefore leaves the tableau unchanged, and an
  appended generator goes after the existing ones instead of after their
  canonical form. A canonical tableau
  still reads the coefficients off its pivot rows, and `expect!` and
  `expect_int!` still canonicalize. On states after 64 monitored layers
  (`p = 0.05` and `0.15`) a fixed-outcome measurement takes 1.5 ms instead of
  226 ms at `d = 2` and 9.4 ms instead of 1.27 s at `d = 31` (`n = 1000`,
  `p = 0.05`), 14–180× faster at `n = 1000`, 18–520× at `n = 2000` (2.7 s
  to 5.2 ms at `d = 2`, 11 s to 30 ms at `d = 31`) and 16–37× at `n = 250`;
  on young mixed states, fixed and appending measurements are 1.5–14× faster.
  Calling `canonicalize!` first is still faster for many commuting
  measurements in a row on a state whose generators do not change in
  between. A `DestabilizerTableau` builds the dual of an appended generator
  from single tableau entries instead of `2n`-term pairings and
  re-orthogonalizes the other duals on the new generator's sites, so an
  appending measurement is 6–28× faster at `n = 250`–`1000` and 40–58× at
  `n = 2000`, and `reset!` gains the same way. One monitored layer from a
  young mixed state runs up to 9× faster on a `DestabilizerTableau` and up to
  4.8× on a `StabilizerTableau`, and a pure-state `StabilizerTableau` layer
  with fixed outcomes up to 45× (`n = 2000`, `d = 3`). Only at `d = 2`, after a non-Hermitian generator was
  stored under `phase_policy` `0` or `2`, can a fixed outcome on a
  `StabilizerTableau` differ: the stabilizer group then holds `-I`, and the
  outcome depends on the basis it is computed in.
- A precompile workload (PrecompileTools.jl, a new dependency) caches the
  common calls for both tableau types: a fresh session's first `measure!`,
  `apply!`, `canonicalize!`, entanglement and `random_clifford` calls no longer
  compile, cutting about 7 s of first-use latency to a few milliseconds, at
  about 40 ms more load time. Disable it with the `precompile_workload = false`
  preference for `QuditClifford`, as described in PrecompileTools' docs.
- Preset construction (`StabilizerTableau(d, n; state)`, `DestabilizerTableau`)
  zeroes the tableau once instead of twice: about 25–38 % faster at `n = 256`.
- `apply!` of a Clifford gate is faster, with identical results and no
  allocations. A stored `CliffordOperator` on one to three qudits runs on the
  same unrolled kernel as the named gates, the kernel's inner closures are
  inlined, matrix–vector products and phases reduce with a validated Barrett
  constant (bit operations at `d = 2`), and the qubit phase is evaluated in
  closed form without data-dependent branches. At `n = 256` a random one-qudit
  operator applies 2.4–4.7× faster, a random two-qudit operator 2.3–3.6×
  faster and the named gates 1.6–3×; at `n = 64` the gains are 2–7×. A
  monitored random circuit (`d = 3`, `n = 32`) runs 1.6–1.9× faster.
  Operators on four or more qudits are unchanged.
- The rank eliminations behind `entanglement_entropy`, `entanglement_profile`,
  `entanglement_profile!`, `mutual_information`, `tripartite_information`,
  `conditional_mutual_information`, `entanglement_negativity` and
  `is_pure(tab; verify = true)` copy the generator rows into a workspace and
  eliminate there, with identical results and no new allocations: bit-packed
  `UInt64` words at `d = 2`, and entries of the tableau's storage element type
  otherwise, one byte each for `d ≤ 251` by default. At `n = 2048`, the
  half-chain entropy of a random pure state is 40× faster at `d = 2`, 11× at
  `d = 3` and 16× at `d = 31`, and its profile 48×, 8× and 11× faster; on a
  monitored-circuit state with little entanglement the gains are 2.5–12×.
  At `d = 2` the mutual, tripartite and conditional mutual information of
  random pure and mixed states are 7–18× faster at `n = 512` and 26–40× at
  `n = 2048`, and the negativity 14–19× and 34–63×. At `n = 512` these
  measures are 4–13× faster for other `d ≤ 251`, and
  `is_pure(tab; verify = true)`, which also checks commutation, is
  1.2–3.8× faster. Each tableau holds one rank workspace: `2n × n` entries of
  its storage element type for `d ≥ 3`, or `2n × n` bits plus two words per
  column at `d = 2`.
- Tableau operations run on the narrow storage that is the default:
  generator updates in `apply!`, `measure!`, `dephase!` and `canonicalize!`
  operate on `UInt8` or `UInt16` entries. At `n ≥ 1024` a tableau with its
  workspaces takes about twice the bytes of its generators or less: at
  `n = 1024`–`4096` a `StabilizerTableau` takes 2.0–2.2× at `d = 3` and
  1.2–1.35× at `d = 2`, a `DestabilizerTableau` 1.5–1.7× and 1.1–1.25×. At
  small `n` the padding of the generator blocks adds more (a
  `StabilizerTableau` at `d = 3` takes 3.7× at `n = 32`). A
  `StabilizerTableau` at `d = 3` and `n = 1024` takes 4.4 bytes per `n²`
  instead of 32.8 with `Int` storage. Against `main` before the split and
  narrow storage (the state after the commuting-measurement change above), at
  `n = 511`–`1025` and `d = 2, 3, 31`, one-qudit random gates run 1.3–7×
  faster, two-qudit random gates 1.1–4.4×, random-outcome measurements
  1.2–2.3× and `canonicalize!` 1.4–4.3×; at `d = 257` (`UInt16`) the
  half-chain entropy is 10× faster.
  Full timings are in #60.

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

- `entanglement_entropy` throws an `ArgumentError` for a subsystem whose axes
  are not 1-based, such as a view indexed by `Base.IdentityUnitRange(3:3)`,
  also with `check=false`. Such a subsystem used to give a wrong entropy.
- `StabilizerTableau(d, n)` and `DestabilizerTableau(d, n)` throw an
  `ArgumentError` naming `n` for a negative `n`, in place of an internal
  "invalid Array dimensions" error.
- `entanglement_entropy`, `mutual_information`, `tripartite_information`,
  `conditional_mutual_information` and `entanglement_negativity` accept a
  subsystem whose `length` is not an `Int`, such as `big(1):big(3)`. It used
  to throw a `MethodError`.
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
