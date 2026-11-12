```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Clifford Unitaries

A Clifford unitary maps Paulis to Paulis under conjugation, so it acts on a
stabilizer tableau by transforming each generator's exponent vector through a
symplectic matrix and shifting its phase. [`apply!`](@ref) does this in place.

This page covers named gates, stored [`CliffordOperator`](@ref) values,
tableau application, Pauli conjugation, inversion and composition, and
uniformly random Clifford operators and stabilizer states.

## The gate set

Names are qudit-native. No qubit aliases are exported; the `d = 2` equivalents
are given below for orientation.

| Gate | Action | At `d = 2` |
|---|---|---|
| `Fourier(q)` | `X ↦ Z`, `Z ↦ X⁻¹` | Hadamard |
| `Phase(q)` | `X ↦ XZ`, `Z ↦ Z` | `S` gate, `diag(1, i)` |
| `Multiplier(q, a)` | `\|j⟩ ↦ \|aj⟩` | identity |
| `PauliGate(q, x, z)` | conjugation by `XˣZᶻ` | Pauli conjugation |
| `SUM(c, t, a)` | `\|u,v⟩ ↦ \|u, v+au⟩` | `CNOT` when `a = 1` |
| `CPhase(q₁, q₂, a)` | `\|u,v⟩ ↦ ω^{auv}\|u,v⟩` | `CZ` when `a = 1` |
| `SWAP(q₁, q₂)` | exchange the two qudits | `SWAP` |

`Multiplier` needs a coefficient invertible mod `d`; `SUM` and `CPhase` accept a
coefficient congruent to zero, which is the identity. In `SUM` it is the second
`Z` image that acquires the backward coupling, `Z₂ ↦ Z₁⁻ᵃZ₂`.

## Applying gates

Named gates are dimension-agnostic values, like the Pauli types: the dimension
comes from the tableau.

```jldoctest clifford-bell
julia> tab = StabilizerTableau(2, 2; state = :product, basis = :Z);

julia> apply!(tab, Fourier(1));

julia> apply!(tab, SUM(1, 2));

julia> tab
Stabilizer Tableau:
    Qudit dimension:  d = 2
    Number of Qudits: n = 2
    Generators:       m = 2
    Tableau:
      X     Z 
     1 1 | 0 0 | 0
     0 0 | 1 1 | 0
```

That circuit prepares a Bell state, so the two qubits are maximally entangled:

```jldoctest clifford-bell
julia> entanglement_entropy(tab, [1])
1
```

`apply!` preserves the generator contract, the generator count `m`, and — for a
[`DestabilizerTableau`](@ref) — the dual basis, which the same symplectic map
transforms without any re-orthogonalization.

## Conjugating operators

[`conjugate`](@ref) gives `U P U†` for the same gates. Since a named gate does
not carry a dimension, pass one:

```jldoctest
julia> conjugate(Fourier(1), SinglePauli(1, 1, 0), 1; d = 3)
Z₁
```

At `d = 2` the phase gate sends `X` to `iXZ`, which the phase exponent records:

```jldoctest
julia> conjugate(Phase(1), SinglePauli(1, 1, 0), 1; d = 2)
[phase=1] X₁ Z₁
```

Applying a gate to a state and conjugating an observable by the same gate agree
on expectation values, which is the relation the test suite checks.

## Phase conventions

Phase exponents follow the package convention in
[Representation and Phase Conventions](@ref): `ω^k` with `k mod d` at odd prime
`d`, and `i^k` with `k mod 4` at `d = 2`. The two regimes genuinely differ — the
qudit phase gate `diag(ω^{j(j-1)/2})` degenerates to the identity if you
substitute `d = 2` — so each has its own evaluator internally.

## Stored Clifford operators

The named gates are dimension-agnostic values. A [`CliffordOperator`](@ref) is
the general alternative: it stores an arbitrary Clifford on an ordered support
at a fixed dimension, as the conjugation images of the ordered generators
`X_t1 … X_tk, Z_t1 … Z_tk`. Column `i` of `F` is the image of generator `i`;
`a[i]` is that image's raw phase exponent.

```jldoctest
julia> U = CliffordOperator(Fourier(1), 3);

julia> U.d, U.targets
(3, [1])

julia> tab = StabilizerTableau(3, 2; state=:product, basis=:Z);

julia> apply!(tab, U) === tab
true
```

Operators compose and invert. `U ∘ V` applies `V` first and requires equal
dimensions and identically ordered targets. Here the composite `U = Phase(1) ∘
Fourier(1)` has raw phases `(0, 1)`, and `inv(U)` recovers raw phases `(1, 0)`:

```jldoctest
julia> U = CliffordOperator(Phase(1), 2) ∘ CliffordOperator(Fourier(1), 2);

julia> U.a
2-element Vector{Int64}:
 0
 1

julia> inv(U).a
2-element Vector{Int64}:
 1
 0
```

Heisenberg evolution of an observable under state evolution by `U` is
`conjugate(inv(U), op)`:

```jldoctest
julia> U = CliffordOperator(Fourier(1), 2);

julia> conjugate(inv(U), GeneralPauli([1, 0], 0))
Z₁
```

Raw construction validates the symplectic condition, and at `d = 2` also the
Hermiticity parity `a[i] ≡ x(F[:,i])·z(F[:,i]) (mod 2)`. `check=false` skips
those two algebraic checks only — shapes, normalization and independent storage
still happen — and makes their preconditions the caller's responsibility.
Nothing downstream re-derives them. Inputs must use one-based indexing;
one-based views, transposes and ranges are accepted and copied into owned dense
storage. Targets remain in the supplied order. For example:

```jldoctest
julia> U = CliffordOperator(2, [3], [0 1; 1 0], [0, 0]);

julia> conjugate(U, SinglePauli(3, 1, 0), 3).xz == [0, 0, 0, 0, 0, 1]
true
```

Stored conjugation infers `d`; an explicit keyword must match. Empty support
represents identity, preserves `iscanonical` under application, and still
returns a fresh normalized Pauli under conjugation:

```jldoctest
julia> E = CliffordOperator(3, Int[], zeros(Int, 0, 0), Int[]);

julia> p = GeneralPauli([-1, 4], -1);

julia> q = conjugate(E, p);

julia> q.xz == [2, 1] && q.phase == 2 && q.xz !== p.xz
true
```

Treat operator fields as read-only. Direct mutation of `targets`, `F`, `a`, or
derived data is unsupported because application trusts the constructor's
invariants and cached context. Construct a new `CliffordOperator`, or refill one
with [`random_clifford!`](@ref), to change its action.

An operator borrows its own scratch during `apply!`, so one operator must not be
applied concurrently from several tasks. Copies and materialized operators own
independent scratch. `inv`, `∘` and `conjugate` allocate and never touch their
operands' data or scratch.

## Random Cliffords and states

[`random_clifford`](@ref) draws a Clifford uniformly from the Clifford group,
modulo global phase, on an ordered support: every symplectic matrix `F` is
equally likely, and so is each of its valid phase vectors. The result is a
stored [`CliffordOperator`](@ref), so it applies, conjugates, inverts and
composes like any other.

```jldoctest random
julia> using Random

julia> rng = Xoshiro(2026);

julia> U = random_clifford(rng, 3, 1:2);

julia> U.d, U.targets, size(U.F)
(3, [1, 2], (4, 4))

julia> E = CliffordOperator(3, 1:2, [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1], zeros(Int, 4));

julia> inv(U) ∘ U == E
true
```

Every coordinate comes from `rand(rng, 0:(d - 1))`, so the distribution is
exact for an ideal RNG. A seed reproduces a sample for the same RNG type, Julia
version and package version, but the draw stream is not a cross-version
guarantee, which is why these examples print only properties that every sample
shares.

[`random_clifford!`](@ref) refills an existing operator in place, on the same
dimension and support, and allocates nothing once warm. The new action never
depends on the old one, and this is the one package operation that changes an
operator in place. A refill changes `==` and `hash`, so an operator used as a
dictionary key should not be refilled. It borrows the same scratch as `apply!`,
so the no-concurrent-use rule above covers refilling too; and if the RNG
throws partway through, `U` is left unusable until a later refill succeeds,
with nothing rolled back.

[`random_state!`](@ref) replaces a tableau's state with a uniformly random
stabilizer state that has `m` independent generators. The default, `m = n`,
gives a pure state:

```jldoctest random
julia> tab = DestabilizerTableau(2, 4);

julia> random_state!(rng, tab) === tab
true

julia> tab.m, is_pure(tab; verify = true)
(4, true)
```

A smaller `m` gives the normalized projector onto a uniformly random stabilizer
code: the stabilizer group has `d^m` elements and the density matrix has rank
`d^(n - m)`. This is not a random mixture of stabilizer states. `m = 0` is the
maximally mixed state.

```jldoctest random
julia> random_state!(rng, tab; m = 2);

julia> tab.m, is_pure(tab)
(2, false)
```

Each call with `m > 0` allocates a `2n × 2n` scratch matrix and costs
`O(n²m)`. Every draw happens before the tableau changes, so an exception from
the RNG leaves the tableau as it was. Without stored phases
(`storephase = false`) the state is uniform over isotropic subspaces, and no
phases are drawn. A [`DestabilizerTableau`](@ref) receives its dual basis
directly.

### A monitored random circuit

Random two-qudit Cliffords interleaved with measurements are the standard model
of a monitored circuit. Keep one operator per bond, refill it for every layer,
and give each measurement an outcome from the same RNG: `measure!` otherwise
draws its outcome from the global RNG, and one seed would no longer reproduce
the whole trajectory.

```jldoctest random
julia> function monitored!(rng, tab; layers = 8, p = 0.25)
           n, d = tab.n, tab.d
           bonds = [random_clifford(rng, d, [q, q + 1]) for q in 1:(n - 1)]
           for layer in 1:layers
               for q in (isodd(layer) ? 1 : 2):2:(n - 1)
                   apply!(tab, random_clifford!(rng, bonds[q]))
               end
               for q in 1:n
                   rand(rng) < p || continue
                   measure!(tab, SinglePauli(q, 0, 1); outcome = rand(rng, 0:(d - 1)))
               end
           end
           return tab
       end;

julia> tab = monitored!(Xoshiro(7), StabilizerTableau(2, 8; state = :product));

julia> is_pure(tab; verify = true)
true
```

Projective measurement keeps a pure state pure, so the final state is pure
whatever the sample.
