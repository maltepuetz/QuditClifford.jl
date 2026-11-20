```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Tableaux

A tableau represents a stabilizer state, pure or mixed, on `n` qudits of prime
dimension `d`. This page covers the choice between the two tableau types, how
to build a tableau from preset states or from a raw matrix, the contract its
generators satisfy, copying, comparing and resetting, and the storage element
type. The storage layout is defined in [Tableau layout](@ref).

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
whole tableau, not just its generators: workspaces included, a tableau takes
at most about twice the memory of its generator storage.

## Mutating operations

Functions ending in `!` reuse tableau workspaces and may change the tableau or
its canonical representation. In particular, [`measure!`](@ref) updates the
represented state, [`canonicalize!`](@ref) changes the generator basis, and
[`expect!`](@ref) may canonicalize a plain stabilizer tableau as part of its
span calculation, unless the tableau is pure and stores no phases. The
represented quantum state is unchanged by canonicalization. A deterministic
[`measure!`](@ref) never changes the generator basis: a plain stabilizer
tableau that is not canonical solves for the span on a copy of its generators.
