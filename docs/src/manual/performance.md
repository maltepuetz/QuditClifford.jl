```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Limitations and Performance

The first part of this page lists the limits that affect correctness: which
dimensions work, how large `d` may be, and what may not be shared between
tasks. The second part shows how to keep a simulation fast.

## Limitations

### Prime dimensions only

The tableau algorithms divide by residues mod `d`, which requires a field, so
`d` must be prime. Constructors throw `ArgumentError` for any other `d`.
Composite dimensions such as `d = 4` or `d = 6` are not supported.

### Supported arithmetic envelope

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

### Modular inversion strategies

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
lookup table once; a table built for another dimension is rejected.
`QuditClifford.JustInTimeInvMod()` calls `Base.invmod` per query and stores
nothing, which is what makes the large dimensions above reachable at all.

`PrecomputedInvMod` stores its table as a `Vector{Int}`, and both strategies
return an `Int`, so the package is `Int` arithmetic throughout. A table given
in a narrower or unsigned integer type is converted on construction; one whose
element type is not an `Integer` is rejected there.

### Tasks and threads

Every tableau and every stored operator carries preallocated workspaces, and
they are written even by functions that do not change the represented state: [`expect!`](@ref), [`expect_int!`](@ref), the entanglement measures and
[`is_pure`](@ref)`(tab; verify=true)` all use them, and a
[`StabilizerTableau`](@ref) may be canonicalized in place. A stored
[`CliffordOperator`](@ref) likewise uses its own scratch during
[`apply!`](@ref) and [`random_clifford!`](@ref). So a tableau or an operator
must be used by one task at a time. To parallelize over trajectories, give
every task its own tableau and its own operators, made with `copy`; named gates
and Pauli operators are immutable values and can be shared freely.

## Performance

### Allocation-free loops

Once compiled, the operations of a simulation loop allocate nothing:
[`apply!`](@ref) with a named gate or a stored [`CliffordOperator`](@ref),
[`measure!`](@ref), [`expect!`](@ref), the entanglement measures, and the
in-place forms below, because each tableau and each operator works in its own
preallocated workspaces. Three patterns keep allocation out of the loop entirely:

- branch trajectories with `copy!(branch, tab)` into one preallocated tableau
  instead of `copy(tab)`;
- refill a random gate with [`random_clifford!`](@ref) instead of drawing a new
  one with [`random_clifford`](@ref);
- collect entropy profiles with [`entanglement_profile!`](@ref) into one output
  vector.

Construction, [`random_state!`](@ref), `inv`, `∘` and the dense conversions of
[Inspecting States](@ref) allocate.

### Skipping checks

Validation that costs more than the operation it guards can be skipped:

- `check=false` on the raw-matrix tableau constructors skips the ``O(n^3)``
  commutation, independence and Hermiticity check ([The generator contract](@ref));
- `check=false` on [`CliffordOperator`](@ref) skips the ``O(k^3)``
  symplecticity check;
- `check=false` on [`entanglement_entropy`](@ref) skips the ``O(n)`` subsystem
  check.

Each makes its precondition the caller's responsibility, and nothing
downstream re-checks it: invalid input gives wrong results, and in the last
case may read or write out of bounds.

### Dropping phases

`storephase=false` removes the phase row. Phase-sensitive results then report
phase exponent `0` by convention: a deterministic measurement returns `0`,
[`expect_int!`](@ref) returns `0` and [`expect!`](@ref) returns `1` for an
operator in the stabilizer group. Dense conversions are refused. Everything that
depends only on the stabilizer group still works: random outcomes, the
generator structure, and every entanglement measure. For a pure
[`StabilizerTableau`](@ref), deterministic measurements and expectation values
then need only a commutation check (see [Choosing a tableau type](@ref)).

### First-call latency

Julia compiles each method on its first call. The package ships a precompile
workload that runs construction, gates, random sampling, measurements,
channels, expectation values and the common entanglement measures for both
tableau types at `d = 2` and `d = 3`, so these first calls are fast in a new
session. The workload uses PrecompileTools and can be switched off through its
preferences. Other argument types, such as a
different integer type or an unusual Pauli representation, compile on first
use as usual.
