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

For odd prime `d`, an outcome `t` denotes the eigenvalue ``\omega^t``. For a
nondeterministic qubit measurement, outcome `0` selects eigenvalue `+1` and
outcome `1` selects eigenvalue `-1`. See [`measure!`](@ref) for deterministic
outcomes and `phase_policy`.

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
