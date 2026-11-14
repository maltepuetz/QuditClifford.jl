```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Clifford Unitaries

A Clifford unitary maps Paulis to Paulis under conjugation, so it acts on a
stabilizer tableau by transforming each generator's exponent vector through a
symplectic matrix and shifting its phase. [`apply!`](@ref) does this in place.

This page covers named gates, stored [`CliffordOperator`](@ref) values,
tableau application, Pauli conjugation, inversion and composition. The
direction of the action (``\rho \mapsto U \rho\, U^\dagger``, and `U ∘ V` applies `V` first) is fixed
in [Action direction](@ref); sampling random Cliffords is covered in
[Random Sampling](@ref).

## The gate set

Names are qudit-native. No qubit aliases are exported; the `d = 2` equivalents
are given below for orientation. Here ``\omega = e^{2\pi i/d}`` and all
arithmetic on basis labels is mod `d`, as defined in [Conventions](@ref); at
`d = 2`, ``\omega = -1``. Generators not listed are left unchanged.

| Gate | Generator images | On basis states | At `d = 2` |
|:---|:---|:---|:---|
| `Fourier(q)` | ``X \mapsto Z``, ``Z \mapsto X^{-1}`` | ``\lvert k\rangle \mapsto \frac{1}{\sqrt d} \sum_j \omega^{jk} \lvert j\rangle`` | Hadamard |
| `Phase(q)` | ``X \mapsto XZ``, ``Z \mapsto Z`` | ``\lvert j\rangle \mapsto \omega^{j(j-1)/2} \lvert j\rangle``; at `d = 2`, ``\lvert j\rangle \mapsto i^j \lvert j\rangle`` | `S` gate |
| `Multiplier(q, a)` | ``X \mapsto X^a``, ``Z \mapsto Z^{a^{-1}}`` | ``\lvert j\rangle \mapsto \lvert aj\rangle`` | identity |
| `PauliGate(q, x, z)` | ``P(v) \mapsto \omega^{-\langle (x,z), v\rangle} P(v)`` | ``\lvert j\rangle \mapsto \omega^{zj} \lvert j + x\rangle`` | Pauli conjugation |
| `SUM(c, t, a)` | ``X_c \mapsto X_c X_t^{a}``, ``Z_t \mapsto Z_c^{-a} Z_t`` | ``\lvert u, v\rangle \mapsto \lvert u, v + au\rangle`` | `CNOT` when `a = 1` |
| `CPhase(q₁, q₂, a)` | ``X_1 \mapsto X_1 Z_2^{a}``, ``X_2 \mapsto Z_1^{a} X_2`` | ``\lvert u, v\rangle \mapsto \omega^{auv} \lvert u, v\rangle`` | `CZ` when `a = 1` |
| `SWAP(q₁, q₂)` | ``X_1 \leftrightarrow X_2``, ``Z_1 \leftrightarrow Z_2`` | ``\lvert u, v\rangle \mapsto \lvert v, u\rangle`` | `SWAP` |

At `d = 2` the `Phase` image of `X` is ``iXZ = Y``, which its phase exponent
`1` records. A Clifford is defined only up to a global phase, so the basis-state
column fixes one representative; `PauliGate` in particular is the conjugation
by ``X^x Z^z``, which agrees with that operator up to a phase.

`Multiplier` needs a coefficient invertible mod `d`; `SUM` and `CPhase` accept a
coefficient congruent to zero, which is the identity.

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

[`conjugate`](@ref) gives ``U P\, U^\dagger`` for the same gates. Since a named gate does
not carry a dimension, pass one:

```jldoctest
julia> conjugate(Fourier(1), SinglePauli(1, 1, 0), 1; d = 3)
Z₁
```

At `d = 2` the phase gate sends ``X`` to ``iXZ``, which the phase exponent records:

```jldoctest
julia> conjugate(Phase(1), SinglePauli(1, 1, 0), 1; d = 2)
[phase=1] X₁ Z₁
```

Applying a gate to a state and conjugating an observable by the same gate agree
on expectation values, which is the relation the test suite checks.

## Phase conventions

Phase exponents follow [Dimensions and the phase unit](@ref): ``\omega^k`` with
`k mod d` at odd prime `d`, and ``i^k`` with `k mod 4` at `d = 2`. The two
regimes genuinely differ — the qudit phase gate
``\operatorname{diag}(\omega^{j(j-1)/2})`` degenerates to the identity if you
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

julia> tab = StabilizerTableau(3, 2; state = :product, basis = :Z);

julia> apply!(tab, U) === tab
true
```

Operators compose and invert. `U ∘ V` applies `V` first and requires equal
dimensions and identically ordered targets. Both operands must be stored
operators: a named gate has no dimension, so materialize it with
`CliffordOperator(g, d)` before composing. Here the composite `U = Phase(1) ∘
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
Hermiticity parity ``a_i \equiv x(F_{:,i}) \cdot z(F_{:,i}) \pmod 2``. `check=false` skips
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

