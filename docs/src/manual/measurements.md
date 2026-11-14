```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Measurements, Expectations and Channels

This page covers projective Pauli measurements, expectation values, and the
channels that discard information: dephasing, trace-out and reset. Outcomes
and expectation exponents use the encodings of
[Outcome and expectation encodings](@ref).

## Pauli representations

Use [`SinglePauli`](@ref), [`DoublePauli`](@ref), [`TriplePauli`](@ref), or
[`NPauli`](@ref) when an operator has small support. [`GeneralPauli`](@ref) and
plain integer vectors cover dense operators. [Building Pauli operators](@ref)
has examples of each. The qutrit operator ``Z_1 Z_2``, for instance, is two
`(qudit, x, z)` triplets:

```jldoctest measurements
julia> DoublePauli(1, 0, 1, 2, 0, 1)
Z₁ Z₂
```

## Projective measurement

[`measure!`](@ref) projects the state onto an eigenspace of a Pauli `P` and
returns the outcome `t`. For eigenvalues ``\omega^t``, outcome `t` maps

```math
\rho \mapsto \frac{\Pi_t\, \rho\, \Pi_t}{\operatorname{tr}(\Pi_t \rho)},
\qquad
\Pi_t = \frac{1}{d} \sum_{k=0}^{d-1} \omega^{-tk} P^k,
```

and a random outcome is drawn with probability ``\operatorname{tr}(\Pi_t \rho)``.
For a Hermitian qubit Pauli, ``\omega = -1`` and ``\Pi_t = (I + (-1)^t P)/2``.

On a stabilizer state there are three cases:

1. `P` fails to commute with some generator. The outcome is uniformly random;
   one noncommuting generator is replaced by `P`, and the others are multiplied
   by powers of it so that they commute again.
2. `P` commutes with every generator and lies in the stabilizer group, up to a
   phase. The outcome is deterministic and the state does not change.
3. `P` commutes with every generator but lies outside the group, which requires
   a mixed state (`m < n`). The outcome is uniformly random and `P` is appended
   as a new generator.

An outcome `t` lies in `0:(d - 1)` and denotes the eigenvalue ``\omega^t`` for
odd prime `d` and ``(-1)^t`` for qubits. The `outcome` keyword fixes the
outcome of a random measurement and uses the same encoding as the returned
value, so a repeated measurement returns the outcome of the first:

```jldoctest measurements
julia> state = DestabilizerTableau(2, 1; state = :product, basis = :Z);

julia> x1 = SinglePauli(1, 1, 0);

julia> measure!(state, x1; outcome = 1)  # random case: select eigenvalue -1
1

julia> measure!(state, x1)               # now deterministic
1

julia> expect!(state, x1) ≈ -1
true
```

Without `outcome`, a random measurement draws its outcome uniformly from
`0:(d - 1)`. Pass an RNG first, as in `measure!(rng, state, x1)`, to draw from
it instead of the global RNG; [Reproducible trajectories](@ref) shows how one
seeded RNG reproduces a whole circuit.

With `storephase=false` a deterministic measurement returns `0` by convention,
because the eigenvalue was not stored.

### Non-Hermitian qubit Paulis

For odd `d` every Pauli has eigenvalues ``\omega^t`` and can be measured. For
qubits, a Pauli whose phase exponent has the wrong parity, such as
``XZ = -iY``, has eigenvalues ``\pm i`` and no outcome in `0:1`. The
`phase_policy` keyword of [`measure!`](@ref) decides what happens to such an
operator on a tableau with stored phases:

| `phase_policy` | Effect |
|:---|:---|
| `0` (default) | measure the operator as given and emit a warning |
| `1` | measure the Hermitian operator with the same sign bit instead: ``XZ`` and ``iXZ`` both become ``Y``, ``-XZ`` and ``-iXZ`` become ``-Y`` |
| `2` | measure the operator as given, without a warning |

With `0` or `2`, a deterministic measurement throws an `ArgumentError` because
the eigenvalue is ``\pm i``, and a random one stores a non-Hermitian generator.
Prefer to build the Hermitian operator directly (the qubit `Y` is
`SinglePauli(q, 1, 1, 1)`, see [Pauli operators and their ordering](@ref)), or
pass `phase_policy = 1`:

```jldoctest
julia> state = DestabilizerTableau(2, 1; state = :product, basis = :Y);

julia> measure!(state, SinglePauli(1, 1, 1); phase_policy = 1)  # XZ, measured as Y
0
```

## Expectation values

[`expect!`](@ref) returns ``\langle P \rangle`` as a `ComplexF64`. For a
stabilizer state it is either a power of the phase unit, when `P` lies in the
stabilizer group up to a phase, or zero:

```jldoctest expectations
julia> state = DestabilizerTableau(3, 1; state = :product, basis = :Z);

julia> expect!(state, SinglePauli(1, 0, 1))  # ⟨Z₁⟩
1.0 + 0.0im

julia> expect!(state, SinglePauli(1, 1, 0))  # ⟨X₁⟩
0.0 + 0.0im
```

[`expect_int!`](@ref) returns the same result as an integer instead of
converting it to a complex number:

- `-1` means the expectation value is zero.
- For odd prime `d`, `k` means ``\langle P\rangle=\omega^k``, with `k` in `0:(d - 1)`.
- For `d = 2`, `k` means ``\langle P\rangle=i^k``, with `k` in `0:3`.
- With `storephase=false`, an operator in the stabilizer group returns `0` by
  convention because its phase was not stored.

```jldoctest expectations
julia> expect_int!(state, SinglePauli(1, 0, 1))
0

julia> expect_int!(state, SinglePauli(1, 1, 0))
-1
```

Most code should call `expect!`. Use the integer form to work with phase
exponents directly, or to avoid the complex conversion in an inner loop.

## Forgetting outcomes, discarding and resetting

[`measure!`](@ref) records an outcome: the tableau follows one pure trajectory.
When the outcome is not kept, the state is the average over all outcomes, which
is mixed. [`dephase!`](@ref) applies this averaged channel for a Pauli
operator `P`,

```math
\rho \mapsto \sum_t \Pi_t\, \rho\, \Pi_t = \frac{1}{d} \sum_{k=0}^{d-1} P^k \rho\, P^{-k}.
```

The stabilizer group shrinks to its subgroup of elements that commute with
`P`. At most one generator is lost, because the noncommuting generators can be
multiplied by powers of the first one until only that one fails to commute. The
Pauli's phase does not matter, nothing is sampled, and a state that already
commutes with `P` is left unchanged.

[`traceout!`](@ref) discards qudits. It is dephasing by ``X_q`` and then by
``Z_q``, which leaves qudit `q` maximally mixed and uncorrelated with the rest,
while `n` stays the same:

```math
\rho \mapsto \operatorname{tr}_q(\rho) \otimes \frac{I_q}{d}.
```

[`reset!`](@ref)`(tab, sites; basis)` is a trace-out followed by fixing each
site in the chosen basis eigenstate, again without recording an outcome, so
qudits that were entangled with the site stay mixed.

The example below dephases one half of a qutrit Bell pair. Entropy, mutual
information and negativity are in units of ``\log d``, here trits. The mutual
information drops from `2` to `1` and the negativity from `1` to `0`: the
remaining correlations are classical. A trace-out removes all correlations, and
a reset leaves the partner maximally mixed.

```jldoctest forgetting
julia> tab = DestabilizerTableau(3, 2; state = :ghz);

julia> (tab.m, entanglement_entropy(tab, [1]), mutual_information(tab, [1], [2]), entanglement_negativity(tab, [1], [2]))
(2, 1, 2, 1)

julia> dephase!(tab, SinglePauli(1, 0, 1));  # forget the outcome of Z on qutrit 1

julia> (tab.m, entanglement_entropy(tab, [1]), mutual_information(tab, [1], [2]), entanglement_negativity(tab, [1], [2]))
(1, 1, 1, 0)

julia> tab = DestabilizerTableau(3, 2; state = :ghz);

julia> traceout!(tab, 1);

julia> (tab.m, mutual_information(tab, [1], [2]))
(0, 0)

julia> tab = DestabilizerTableau(3, 2; state = :ghz);

julia> reset!(tab, 1);

julia> (tab.m, entanglement_entropy(tab, [1]), entanglement_entropy(tab, [2]))
(1, 0, 1)
```
