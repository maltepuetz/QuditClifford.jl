```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Tableaux

A tableau represents a stabilizer state, pure or mixed, on `n` qudits of prime
dimension `d`. This page covers the choice between the two tableau types, how
to build a tableau from preset states or from a raw matrix, the contract its
generators satisfy, copying, comparing and resetting, and the storage element
type and the storage layout. The matrix layout of the generators is defined
in [Tableau layout](@ref).

## Choosing a tableau type

QuditClifford exposes two representations with the same high-level API:

- [`DestabilizerTableau`](@ref) stores the active stabilizers and a dual
  destabilizer basis. The dual basis makes repeated membership, measurement,
  and expectation calculations efficient without repeatedly canonicalizing the
  stabilizers. It is the recommended default for general use and especially
  for deep or measurement-heavy circuits.
- [`StabilizerTableau`](@ref) stores only the stabilizer generators plus its
  workspaces. It uses less memory and can be suitable when tableaux are mainly
  stored, imported, canonicalized once, or queried only occasionally. Repeated
  span-dependent operations can require additional canonicalization work, or,
  in [`measure!`](@ref), an elimination on a copy of the generators. A
  pure state with `storephase=false` is the exception: there a commuting Pauli
  is always in the span, so deterministic measurements and expectation values
  need only a commutation check: ``O(km)`` for a sparse Pauli on `k` qudits, or
  ``O(nm)`` for a dense Pauli, where `m` is the number of generators.

The extra speed of a destabilizer tableau comes from retaining an additional
``2n \times n`` dual matrix and keeping it synchronized after updates. Choose a
plain stabilizer tableau when that memory tradeoff matters; otherwise begin
with a destabilizer tableau.

## Preset states

Both types are constructed from `(d, n)` with a preset `state`: `:mixed` (the
default, maximally mixed, `m = 0`), `:product` with a `basis` (`:X`, `:Y`,
`:Z`, or one basis per qudit), or `:ghz`. `state = :X`, `:Y` or `:Z` is
shorthand for a product state in that basis. For example:

```jldoctest presets
julia> StabilizerTableau(3, 3; state = :ghz)
Stabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 3
    Generators:       m = 3
    Storage:          generator-major, UInt8
    Tableau:
     - X -   - Z -
     1 1 1 | 0 0 0 | 0
     0 0 0 | 1 2 0 | 0
     0 0 0 | 0 1 2 | 0

julia> StabilizerTableau(3, 3; state = :product, basis = [:X, :Z, :X])
Stabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 3
    Generators:       m = 3
    Storage:          generator-major, UInt8
    Tableau:
     - X -   - Z -
     1 0 0 | 0 0 0 | 0
     0 0 0 | 0 1 0 | 0
     0 0 1 | 0 0 0 | 0
```

The GHZ state is fixed by ``X_1 X_2 X_3`` and by ``Z_q Z_{q+1}^{-1}`` on
neighbouring qudits. [A first state](@ref) explains how to read the display.

## Raw tableaux

A tableau can also be built from an integer matrix of generators, either in the
column layout of [Tableau layout](@ref) or its transpose. The constructor
infers the orientation, `n`, and `storephase` from the shape when that is
unambiguous; pass `m` or `storephase` to settle the cases where it is not. The
tableau stores a reduced copy: the caller's matrix is never modified, and later
changes to it do not affect the tableau.

```jldoctest raw-tableau
julia> zz_xx = [0 1; 0 1; 1 0; 1 0; 0 0];  # columns Z₁Z₂ and X₁X₂, then phases

julia> tab = StabilizerTableau(2, zz_xx; m = 2, storephase = true);

julia> is_pure(tab), entanglement_entropy(tab, [1])
(true, 1)
```

## The generator contract

The active columns must generate an abelian group: they must commute pairwise
under the [symplectic form](@ref "The symplectic form"), and be linearly
independent over ``\mathrm{GF}(d)``. At `d = 2` with stored phases, each
generator must also be Hermitian, with phase exponent ``\equiv x \cdot z \pmod 2``.
These are assumed everywhere — [`measure!`](@ref) decomposes an operator against
the generators as a basis, which means nothing if they do not span what `m`
claims.

Construction is the only place this can be violated, so it is the only place it
is checked. The preset `(d, n)` constructors satisfy the contract by definition,
and measurements, gates, channels and [`canonicalize!`](@ref) preserve
commutation and independence; the raw-matrix constructors
validate what they are handed and throw `ArgumentError`:

```jldoctest
julia> bad = zeros(Int, 5, 2); bad[1, 1] = 1; bad[3, 2] = 1;  # X₁ and Z₁

julia> StabilizerTableau(2, bad; m = 2, storephase = true)
ERROR: ArgumentError: Stabilizer generators must commute, but generators 1 and 2 do not. Pass check=false to skip this check.
```

That validation is ``O(n^3)``. Pass `check=false` to skip it when the input is
already known to be well formed — the tableau is then built regardless, and
every later operation rests on a contract that may not hold.

Because the contract holds by construction, [`is_pure`](@ref) tests only
`m == n`, in constant time. `is_pure(tab; verify=true)` re-derives the full
condition instead, which is the only way to detect a tableau built behind
`check=false`.

## Copying, comparing and resetting

`copy(tab)` returns an independent tableau in the same generator basis.
`copy!(dst, src)` overwrites an existing tableau of the same kind, `d`, `n` and
`storephase` without allocating, which is the way to branch many trajectories
off one state.

`==` compares the represented states, not the stored matrices: the generator
basis, the destabilizers and the tableau type do not matter, and `hash` agrees
with `==`.

```jldoctest copy-compare
julia> a = StabilizerTableau(3, 4; state = :ghz);

julia> b = copy(a); canonicalize!(b);

julia> a == b, a == DestabilizerTableau(3, 4; state = :ghz)
(true, true)

julia> measure!(b, SinglePauli(1, 1, 0); outcome = 0); a == b
false
```

[`reset!`](@ref)`(tab; state, basis)` puts an existing tableau into a preset
state, keeping its `d`, `n`, `storephase` and inversion strategy. The per-site
form, `reset!(tab, sites)`, is a channel and is described in
[Forgetting outcomes, discarding and resetting](@ref).

## Storage

A tableau stores its X and Z exponents in an integer type chosen from `d`,
set by the `eltype` keyword of both constructors. The default is the narrowest
type that holds every entry:

| `d`                  | default `eltype` |
|:---------------------|:-----------------|
| `d ≤ 251`            | `UInt8`          |
| `251 < d ≤ 65521`    | `UInt16`         |
| `d > 65521`          | `Int`            |

`eltype = Int` is accepted for every `d` and is the reference the narrow types
are tested against; `UInt16` may also be chosen for `d ≤ 251`. Every result is
identical in each element type: outcomes, expectation values, entropies and
the matrices returned by [`stabilizer_matrix`](@ref) and
[`destabilizer_matrix`](@ref) are the same `Int` values, phases are always
stored as `Int`, and `==` and `hash` compare tableaux across element types.
`copy!` needs both tableaux to have the same element type. The element type is
the second type parameter, as in `StabilizerTableau{IM, UInt8}`, and `show`
prints it on its `Storage` line:

```jldoctest storage
julia> tab = StabilizerTableau(257, 2)
Stabilizer Tableau:
    Qudit dimension:  d = 257
    Number of Qudits: n = 2
    Generators:       m = 0
    Storage:          generator-major, UInt16
    Mixed tableau (m < n generators):
     (no generators; maximally mixed on the full space)

julia> tab == StabilizerTableau(257, 2; eltype = Int)
true
```

The `2n × n` rank workspace behind the entanglement measures (bits at
`d = 2`) uses the same element type, so a narrow element type shrinks the
whole tableau, not just its generators. Measured with `Base.summarysize` on a
fresh tableau, in bytes per `n²`:

| tableau                       | `n = 256` | `n = 1024` | `n = 4096` |
|:------------------------------|----------:|-----------:|-----------:|
| `StabilizerTableau`, `d = 2`   |      4.01 |       2.66 |       2.35 |
| `StabilizerTableau`, `d = 3`   |      5.70 |       4.40 |       4.10 |
| `DestabilizerTableau`, `d = 2` |      7.41 |       4.99 |       4.43 |
| `DestabilizerTableau`, `d = 3` |      9.09 |       6.73 |       6.18 |
| `StabilizerTableau`, `d = 3`, `eltype = Int`   | 35.45 | 32.84 | 32.21 |
| `DestabilizerTableau`, `d = 3`, `eltype = Int` | 54.54 | 49.59 | 48.39 |

The generators alone take 2 bytes per `n²` (4 with destabilizers) in `UInt8`.
At `d ≥ 3` the rank workspace adds as much again for a `StabilizerTableau`,
and the padding of the generator blocks adds the rest, most of it at small
`n`.

## Packed qubit storage

At `d = 2`, `eltype = Bits` stores the X and Z blocks of a tableau as packed
bits, 64 to a `UInt64` word, in either layout and for both tableau types. It is
legal only at `d = 2`; any other `d` throws an `ArgumentError`. The default
stays `UInt8`.

Every result is identical to the `UInt8` tableau's: generators, phases, `m`,
measurement outcomes drawn from the same `rng`, entropies and the matrices
returned by [`stabilizer_matrix`](@ref) and [`destabilizer_matrix`](@ref).
`==` and `hash` compare across storage types, and `copy!` needs both tableaux
to have the same one. [`tableau_eltype`](@ref) returns the storage type chosen
at construction, `UInt8`, `UInt16`, `Int` or [`Bits`](@ref), and `show` prints
it on the `Storage` line.

```jldoctest
julia> tab = StabilizerTableau(2, 4; state = :ghz, layout = :qudit, eltype = Bits);

julia> measure!(tab, SinglePauli(2, 1, 0); outcome = 0)
0

julia> tableau_eltype(tab), tableau_layout(tab)
(Bits, :qudit)

julia> ref = StabilizerTableau(2, 4; state = :ghz);

julia> measure!(ref, SinglePauli(2, 1, 0); outcome = 0);

julia> stabilizer_matrix(tab) == stabilizer_matrix(ref)
true
```

[`qudit_major!`](@ref) and [`generator_major!`](@ref) switch a packed tableau
in place, like any other, by transposing 64 × 64 bit strips. The switch
allocates nothing and costs about a ninth of the `UInt8` switch.

**Memory.** Measured with `Base.summarysize` on a fresh tableau, in bytes per
`n²`, on an Apple M3 Pro:

| tableau                        | `n = 1024` | `n = 2048` | `n = 4096` | `n = 16384` |
|:-------------------------------|-----------:|-----------:|-----------:|------------:|
| `StabilizerTableau`, `UInt8`   |       2.67 |       2.46 |       2.35 |        2.28 |
| `StabilizerTableau`, `Bits`    |       0.59 |       0.67 |       0.59 |        0.52 |
| `DestabilizerTableau`, `UInt8` |       4.99 |       4.62 |       4.43 |        4.30 |
| `DestabilizerTableau`, `Bits`  |       0.84 |       1.06 |       0.90 |        0.79 |

Packed tableaux are 4 to 6 times smaller from `n = 1024` on. The bump at
`n = 2048` is the padding of the word columns.

**Speed.** Compared with `UInt8` storage in the same layout at `n = 1024` and
`2048`:

- Gates and measurements are much faster. Qudit-major gates take 0.05–0.16 of
  the `UInt8` time and random-outcome measurements 0.17–0.32; generator-major
  random-outcome measurements take 0.18–0.38.
- [`canonicalize!`](@ref) takes 0.08–0.21 of the time in both layouts, and a
  whole monitored circuit 0.16–0.70. Packed qudit-major `canonicalize!` switches
  to generator-major internally and switches back, so the layout of the tableau
  is unchanged afterwards.
- The entanglement measures take 0.43–0.86.
- Generator-major `gate2` takes 0.60–0.74 and generator-major `gate1` is
  slower, 1.1–1.6 times. On a [`DestabilizerTableau`](@ref), generator-major
  [`expect!`](@ref) and [`expect_int!`](@ref) are 2.1–2.7 times slower. Both
  are still faster than switching to the other packed layout and back.
- [`stabilizer_matrix`](@ref) is slower, 1.4–2.1 times, and 1.46–1.65 times
  in the qudit-major layout. It is a debugging and I/O path.

`UInt8` remains the default until a benchmark round on the target machines
decides otherwise. `benchmark/storage/README.md` describes the benchmark.

## Storage layout

The generators of a tableau are stored in one of two layouts, which differ in
which exponents are contiguous in memory:

- **Generator-major** (`:generator`, the default): the exponents of one
  generator across all qudits are contiguous. Operations that touch a few
  generators, such as the row operations of a measurement, read short runs.
- **Qudit-major** (`:qudit`): the exponents of one qudit across all generators
  are contiguous. A gate on qudits `i` and `j` combines two such columns, so it
  runs over two contiguous vectors with SIMD lanes along the generators.

The `layout` keyword of every constructor selects the layout, `:generator` by
default. [`qudit_major!`](@ref) and [`generator_major!`](@ref) switch an
existing tableau in place, and [`tableau_layout`](@ref) returns `:generator` or
`:qudit`. `show` prints the layout on its `Storage` line. No operation leaves the layout
changed, and `copy!(dst, src)` gives `dst` the layout of `src`.

The layout affects speed, never a result: generators, phases, `m`, measurement
outcomes drawn from the same `rng`, entropies and the matrices returned by
[`stabilizer_matrix`](@ref) and [`destabilizer_matrix`](@ref) are identical in
both layouts and in every element type, and `==` and `hash` compare tableaux
across layouts.

A switch transposes each `n × n` block inside its existing array and allocates
nothing. It costs about one transpose of each block: two for a
[`StabilizerTableau`](@ref), four for a [`DestabilizerTableau`](@ref). With the
default `UInt8` entries at `n = 2048` that is about 1.6 ms and 3.3 ms on an
Apple M3 Pro, cheap next to a phase of many gates.

Which layout is faster depends on the operation. On the same machine, for
`n` from 1023 to 2049 and `d = 2, 3, 31`:

- Gates are 4–16 times faster qudit-major, and monitored circuits of gates and
  measurements about 2–4 times.
- Random-outcome measurements take 0.4–0.95 times the generator-major time.
- [`canonicalize!`](@ref), [`dephase!`](@ref), [`traceout!`](@ref) and
  [`reset!`](@ref) are about as fast or faster qudit-major, and so are
  [`expect!`](@ref) and [`expect_int!`](@ref) on a non-canonical
  [`StabilizerTableau`](@ref).
- The entropies and other rank-based quantities are about as fast in both
  layouts, up to 1.6 times slower qudit-major.
- [`expect!`](@ref) and [`expect_int!`](@ref) are up to about 4 times slower
  qudit-major on a [`DestabilizerTableau`](@ref), and [`expect!`](@ref) is
  1.2–1.8 times slower on a canonical [`StabilizerTableau`](@ref). A
  measurement that commutes with every generator of a mixed
  [`DestabilizerTableau`](@ref) and adds a generator is about 1.1–1.2 times
  slower.
- [`stabilizer_matrix`](@ref) and [`destabilizer_matrix`](@ref) are 1.5–2
  times slower qudit-major (measured at `d = 3`, `n = 1024` and `2048`), and
  `==` of two qudit-major tableaux whose generators agree entry for entry
  takes the generator-major time.

Keep the default generator-major layout unless a phase of the computation is
dominated by gates and measurements, and switch at the phase boundaries. For
example, run the measurement-only evolution qudit-major, switch back, and
compute [`entanglement_profile`](@ref) generator-major:

```jldoctest layout
julia> tab = DestabilizerTableau(3, 6; state = :ghz, layout = :qudit);

julia> measure!(tab, SinglePauli(3, 1, 0); outcome = 0);

julia> tableau_layout(tab)
:qudit

julia> generator_major!(tab); entanglement_profile(tab)
7-element Vector{Int64}:
 0
 1
 1
 1
 1
 1
 0

julia> tableau_layout(tab)
:generator
```

## Mutating operations

Functions ending in `!` reuse tableau workspaces and may change the tableau or
its canonical representation. In particular, [`measure!`](@ref) updates the
represented state, [`canonicalize!`](@ref) changes the generator basis, and
[`expect!`](@ref) may canonicalize a plain stabilizer tableau as part of its
span calculation, unless the tableau is pure and stores no phases. The
represented quantum state is unchanged by canonicalization. A deterministic
[`measure!`](@ref) never changes the generator basis: a plain stabilizer
tableau that is not canonical solves for the span on a copy of its generators.
