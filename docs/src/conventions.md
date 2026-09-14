```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Representation and Phase Conventions

## Supported dimensions

The local qudit dimension `d` must be prime. Qubits use `d = 2`; qutrits use
`d = 3`; higher odd prime dimensions are supported by the same finite-field
tableau algorithms. Constructors reject composite dimensions.

## Choosing a tableau representation

QuditClifford exposes two representations with the same high-level API:

- [`DestabilizerTableau`](@ref) stores the active stabilizers and a dual
  destabilizer basis. The dual basis makes repeated membership, measurement,
  and expectation calculations efficient without repeatedly canonicalizing the
  stabilizers. It is the recommended default for general use and especially
  for deep or measurement-heavy circuits.
- [`StabilizerTableau`](@ref) stores only the stabilizer generators plus its
  workspaces. It uses less memory and can be suitable when tableaux are mainly
  stored, imported, canonicalized once, or queried only occasionally. Repeated
  span-dependent operations can require additional canonicalization work.

The extra speed of a destabilizer tableau comes from retaining an additional
``2n \times n`` dual matrix and keeping it synchronized after updates. Choose a
plain stabilizer tableau when that memory tradeoff matters; otherwise begin
with a destabilizer tableau.

## Tableau layout

For `n` qudits, Pauli vectors use `2n` entries:

```math
(x_1, \ldots, x_n, z_1, \ldots, z_n).
```

A tableau stores generators in columns. The first `m` columns are active, with

```math
0 \leq m \leq n.
```

The remaining columns are zeroed capacity. A pure stabilizer state has `m = n`
independent commuting generators. Mixed states may have `m < n`.

When `storephase=true`, a final row stores each generator's phase exponent.
Raw tableau constructors accept either column-major generator layout or its
transpose and infer the layout when it is unambiguous.

## Phase exponents

- For odd prime `d`, phase `k` represents ``\omega^k`` with
  ``\omega = \exp(2\pi i/d)`` and `k` reduced modulo `d`.
- For `d = 2`, phase `k` represents ``i^k`` and is reduced modulo `4`.
- With `storephase=false`, phase-sensitive deterministic queries return `0` by
  convention.

For qubits, a Pauli used as a physical observable should be Hermitian.
[`measure!`](@ref) exposes `phase_policy` to warn, repair, or explicitly accept
a non-Hermitian phase convention.

## Mutating operations

Functions ending in `!` reuse tableau workspaces and may change the tableau or
its canonical representation. In particular, [`measure!`](@ref) updates the
represented state, [`canonicalize!`](@ref) changes the generator basis, and
[`expect!`](@ref) may canonicalize a plain stabilizer tableau as part of its
span calculation. The represented quantum state is unchanged by
canonicalization.

## Supported arithmetic envelope

Tableau entries, phases and dot products are all `Int`. Every phase and
symplectic dot product sums `n` terms of size up to ``(d-1)^2``, and the
odd-`d` phase update adds two such terms, so results are exact only while

```math
\max(n, 2)\,(d-1)^2 \le \texttt{typemax(Int)}
```

Past that the accumulators wrap silently, and there is no error to notice. The
tableau constructors therefore emit a warning when `d` is too large for the
given `n`, naming the largest safe dimension. The bound is the worst case --
every entry equal to `d-1` -- so it is a warning rather than an error: sparse
tableaux past it often still compute correctly.

On a 64-bit host the limit is `d ≤ 2147483648` at `n = 1`, `d ≤ 189812532` at
`n = 256`, and `d ≤ 94906266` at `n = 1024`. In practice only
`JustInTimeInvMod` can reach it, since `PrecomputedInvMod` allocates a table of
`d-1` integers and exhausts memory first.

## Modular inversion strategies

Every tableau carries a strategy for inverting residues mod `d`, selected with
the `inversemod` keyword at construction. Neither strategy is exported, so name
them through the module:

```jldoctest
julia> jit = QuditClifford.JustInTimeInvMod();

julia> tab = StabilizerTableau(5, 4; state = :ghz, inversemod = jit);

julia> is_pure(tab)
true
```

The default is `QuditClifford.PrecomputedInvMod(d)`, which builds a `d-1` entry
lookup table once. `QuditClifford.JustInTimeInvMod()` calls `Base.invmod` per
query and stores nothing, which is what makes the large dimensions above
reachable at all.

`PrecomputedInvMod` stores its table as a `Vector{Int}`, and both strategies
return an `Int`, so the package is `Int` arithmetic throughout. A table given
in a narrower or unsigned integer type is converted on construction; one whose
element type is not an `Integer` is rejected there.
