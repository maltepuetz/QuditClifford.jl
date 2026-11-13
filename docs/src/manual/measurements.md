```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Measurements and Expectations

## Pauli representations

Use [`SinglePauli`](@ref), [`DoublePauli`](@ref), [`TriplePauli`](@ref), or
[`NPauli`](@ref) when an operator has small support. [`GeneralPauli`](@ref) and
plain integer vectors cover dense operators. [Constructing Pauli operators](@ref)
contains examples of every representation.

For example, the qutrit operator ``Z_1 Z_2`` is:

```jldoctest measurements
julia> op = DoublePauli(1, 0, 1, 2, 0, 1) # Two (qudit, X, Z) triplets specifying Z₁Z₂.
Z₁ Z₂
```

## Projective measurement

[`measure!`](@ref) has three tableau branches:

1. A noncommuting operator produces a random outcome and replaces one
   generator while updating every other noncommuting generator.
2. A commuting operator already in the stabilizer span has a deterministic
   outcome and leaves the represented state unchanged.
3. A commuting operator outside the span produces a random outcome and appends
   a generator when the state is mixed (`m < n`).

Pass `outcome` to make a nondeterministic measurement reproducible:

```jldoctest measurements
# d = 2, n = 1; create the Z-basis state |0⟩.
julia> state = DestabilizerTableau(2, 1; state=:product, basis=:Z);

julia> x1 = SinglePauli(1, 1, 0); # X on qubit 1: (qudit=1, X exponent=1, Z exponent=0).

julia> measure!(state, x1; outcome=0) # Select the +1 eigenvalue, represented by outcome 0.
0

julia> expect!(state, x1) # Evaluate ⟨X₁⟩ in the post-measurement state.
1.0 + 0.0im
```

An outcome `t` lies in `0:(d - 1)` and denotes the eigenvalue ``\omega^t`` for
odd prime `d` and ``(-1)^t`` for qubits. The `outcome` keyword and the returned
value use this encoding in every branch, so a repeated measurement returns the
outcome of the first:

```jldoctest
julia> state = DestabilizerTableau(2, 1; state=:product, basis=:Z);

julia> measure!(state, SinglePauli(1, 1, 0); outcome=1) # Random branch, eigenvalue -1.
1

julia> measure!(state, SinglePauli(1, 1, 0)) # Now deterministic.
1
```

With `storephase=false` a deterministic measurement returns `0` by convention,
because the eigenvalue was not stored. See [`measure!`](@ref) for
`phase_policy`.

Without `outcome`, a random branch draws its outcome uniformly from
`0:(d - 1)`, by default from the global RNG. Pass an RNG first, as in
`measure!(rng, state, x1)`, to draw from it instead: one seeded RNG shared by
every measurement and every [`random_clifford!`](@ref) call reproduces a whole
monitored-circuit trajectory. Only the random branches draw, so a deterministic
measurement leaves the RNG untouched.

## Forgetting outcomes, discarding and resetting

[`measure!`](@ref) records an outcome: the tableau follows one pure trajectory.
When the outcome is not kept, the state is the average over all outcomes, which
is mixed. [`dephase!`](@ref) applies this averaged channel for a Pauli
operator `P`: the stabilizer group shrinks to its subgroup of elements that
commute with `P`. At most one generator is lost, because the noncommuting generators can
be multiplied by powers of the first one until only that one anticommutes. The
Pauli's phase does not matter, nothing is sampled, and a state that already
commutes with `P` is left unchanged.

[`traceout!`](@ref) discards qudits. It is dephasing by `X_q` and then by `Z_q`,
which leaves qudit `q` maximally mixed and uncorrelated with the rest, while
`n` stays the same. [`reset!`](@ref)`(tab, sites; basis)` is a trace-out
followed by fixing each site in the chosen basis eigenstate, again without
recording an outcome, so qudits that were entangled with the site stay mixed.

The example below dephases one half of a qutrit Bell pair. Entropy, mutual
information and negativity are in units of ``\log d``, here trits. The mutual
information drops from `2` to `1` and the negativity from `1` to `0`: the
remaining correlations are classical. A trace-out removes all correlations, and
a reset leaves the partner maximally mixed.

```jldoctest forgetting
julia> tab = DestabilizerTableau(3, 2; state=:ghz);

julia> (tab.m, entanglement_entropy(tab, [1]), mutual_information(tab, [1], [2]), entanglement_negativity(tab, [1], [2]))
(2, 1, 2, 1)

julia> dephase!(tab, SinglePauli(1, 0, 1)); # Forget the outcome of Z on qutrit 1.

julia> (tab.m, entanglement_entropy(tab, [1]), mutual_information(tab, [1], [2]), entanglement_negativity(tab, [1], [2]))
(1, 1, 1, 0)

julia> tab = DestabilizerTableau(3, 2; state=:ghz);

julia> traceout!(tab, 1);

julia> (tab.m, mutual_information(tab, [1], [2]))
(0, 0)

julia> tab = DestabilizerTableau(3, 2; state=:ghz);

julia> reset!(tab, 1);

julia> (tab.m, entanglement_entropy(tab, [1]), entanglement_entropy(tab, [2]))
(1, 0, 1)
```

## Expectation values

Use [`expect!`](@ref) for expectation values. It returns the physical value as
a `ComplexF64`, including complex roots of unity for qudits, and returns zero
when the Pauli is not in the stabilizer span:

```jldoctest expectations
# d = 3, n = 1; create the qutrit Z-basis state |0⟩.
julia> state = DestabilizerTableau(3, 1; state=:product, basis=:Z);

julia> expect!(state, SinglePauli(1, 0, 1)) # Expectation of Z on qutrit 1.
1.0 + 0.0im

julia> expect!(state, SinglePauli(1, 1, 0)) # Expectation of X on qutrit 1.
0.0 + 0.0im
```

There is also a lower-level [`expect_int!`](@ref), which returns the same result
in exponent form rather than converting it to a complex number. Its convention
is:

- `-1` means the physical expectation value is zero.
- For odd prime `d`, `k` means ``\langle P\rangle=\omega^k`` modulo `d`.
- For `d = 2`, `k` means ``\langle P\rangle=i^k`` modulo `4`.
- With `storephase=false`, an operator in the stabilizer span returns `0` by
  convention because its phase was not stored.

For example:

```jldoctest expectation-exponents
# d = 3, n = 1; create the qutrit Z-basis state |0⟩.
julia> state = DestabilizerTableau(3, 1; state=:product, basis=:Z);
```

```jldoctest expectation-exponents
# Return the phase exponent for the expectation of Z₁.
julia> expect_int!(state, SinglePauli(1, 0, 1))
0
```

```jldoctest expectation-exponents
# Here -1 denotes a zero physical expectation value.
julia> expect_int!(state, SinglePauli(1, 1, 0))
-1
```

Most user code should call `expect!`; use the integer form when manipulating
finite-field phase exponents directly or when avoiding the complex conversion
inside a specialized inner loop.
