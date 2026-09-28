```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Dense and Exact States

A tableau stores a state in ``O(n^2)`` integers. Three conversions expand it in
the computational basis, for reading, for small-system verification, and for
handing to other libraries:

- [`ket`](@ref) returns the exact expansion of a pure state as a
  [`StabilizerKet`](@ref);
- [`state_vector`](@ref) returns a dense `Vector{ComplexF64}` of length `d^n`;
- [`density_matrix`](@ref) returns a dense `d^n × d^n` matrix, for pure and
  mixed states alike.

None of them modifies the tableau.

## Exact kets

A pure stabilizer state with `k` X-type generators has exactly `d^k` nonzero
amplitudes. All have modulus ``d^{-k/2}`` and a phase that is an integer power
of ``\zeta`` — ``i`` for qubits, ``\omega = e^{2\pi i/d}`` otherwise — so the
expansion needs no floating point at all:

```jldoctest dense
julia> ket(StabilizerTableau(2, 2; state = :ghz))
StabilizerKet (d = 2, n = 2, 2 terms):
  (|00⟩ + |11⟩)/√2
```

Qubit phases print as signs and `i`, odd-prime phases as powers of
``\omega_d``, the subscript naming the dimension. The qudit phase gate puts
``\omega^{j(j-1)/2}`` on ``|j\rangle``:

```jldoctest dense
julia> tab = StabilizerTableau(3, 1; state = :product, basis = :Z);

julia> apply!(tab, Fourier(1)); apply!(tab, Phase(1));

julia> ket(tab)
StabilizerKet (d = 3, n = 1, 3 terms):
  (|0⟩ + |1⟩ + ω₃|2⟩)/√3
```

Labels are stored as digits rather than as indices into the `d^n`-dimensional
space, so a ket stays small whenever its support does, however large `n` is. A
100-qutrit GHZ state is three terms:

```jldoctest dense
julia> k = ket(StabilizerTableau(3, 100; state = :ghz));

julia> size(k.labels)
(100, 3)

julia> show(IOContext(stdout, :max_ket_label => 8), k)
(|0000…0000⟩ + |1111…1111⟩ + |2222…2222⟩)/√3
```

This is an exact support expansion, not a compressed representation:
``|+\rangle^{\otimes n}`` still has `d^n` terms. The display prints at most 16
terms and 32 coordinates per label; the `IOContext` properties `:max_ket_terms`
and `:max_ket_label` change those caps for one `show` call, and a context with
`:limit` set also stops at the display width.

## Canonical form

Labels ascend lexicographically and the first amplitude is real and positive.
Two tableaux holding the same state therefore give equal kets, whichever
generators they store:

```jldoctest dense
julia> a = StabilizerTableau(3, 3; state = :ghz);

julia> b = StabilizerTableau(3, 3; state = :product, basis = :Z);

julia> apply!(b, Fourier(1)); apply!(b, SUM(1, 2)); apply!(b, SUM(1, 3));

julia> ket(a) == ket(b)
true
```

## Dense vectors and matrices

Qudit 1 is the most significant digit of the basis index, so product states
agree with `kron`. Flipping the first qubit of ``|0\rangle \otimes |+\rangle``
moves the amplitude into the second half of the vector:

```jldoctest dense
julia> tab = StabilizerTableau(2, 2; state = :product, basis = [:Z, :X]);

julia> apply!(tab, PauliGate(1, 1, 0));

julia> state_vector(tab)
4-element Vector{ComplexF64}:
                0.0 + 0.0im
                0.0 + 0.0im
 0.7071067811865475 + 0.0im
 0.7071067811865475 + 0.0im
```

A density matrix is the normalized sum over the stabilizer group,
``\rho = d^{-n} \sum_{g \in S} g``, so its expectation values are the
tableau's:

```jldoctest dense
julia> bell = StabilizerTableau(2, 2; state = :ghz);

julia> ρ = density_matrix(bell)
4×4 Matrix{ComplexF64}:
 0.5+0.0im  0.0+0.0im  0.0+0.0im  0.5+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.5+0.0im  0.0+0.0im  0.0+0.0im  0.5+0.0im

julia> ZZ = kron([1 0; 0 -1], [1 0; 0 -1]);

julia> sum((ρ * ZZ)[i, i] for i in 1:4) ≈ expect!(bell, DoublePauli(1, 0, 1, 2, 0, 1))
true
```

Mixed states convert too, where `ket` and `state_vector` do not apply.
Measuring ``Z_1`` on two maximally mixed qubits leaves
``|0\rangle\langle 0| \otimes I/2``:

```jldoctest dense
julia> tab = StabilizerTableau(2, 2; state = :mixed);

julia> measure!(tab, SinglePauli(1, 0, 1); outcome = 0)
0

julia> density_matrix(tab)
4×4 Matrix{ComplexF64}:
 0.5+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.5+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im

julia> ket(tab)
ERROR: ArgumentError: ket needs a pure state (m == n); got m = 1, n = 2. Use density_matrix for mixed states.
```

## Phases must be tracked from the start

A measurement outcome is recorded in a generator's phase. Without stored phases
a tableau does not determine its state, and adding zero phases afterwards picks
a different one:

```jldoctest dense
julia> tracked = StabilizerTableau(2, 1; state = :product, basis = :X);

julia> measure!(tracked, SinglePauli(1, 0, 1); outcome = 1)
1

julia> ket(tracked)
StabilizerKet (d = 2, n = 1, 1 term):
  |1⟩

julia> untracked = StabilizerTableau(2, 1; state = :product, basis = :X, storephase = false);

julia> measure!(untracked, SinglePauli(1, 0, 1); outcome = 1)
1

julia> ket(untracked)
ERROR: ArgumentError: ket needs a tableau with storephase=true: without stored phases the tableau does not determine a state. Track phases from state preparation onward; adding zero phases afterwards does not recover the lost state.

julia> ket(StabilizerTableau(2, [untracked.stab; zeros(Int, 1, 1)]))
StabilizerKet (d = 2, n = 1, 1 term):
  |0⟩
```

So all three conversions refuse `storephase=false`, even for a maximally mixed
state whose density matrix would not depend on phases.

## Limits

Each conversion takes a `maxentries` keyword bounding the scalar storage it
returns — `(n+1)·d^k` integers for a ket, `d^n` entries for a state vector,
`d^(2n)` for a density matrix — with a default of `2^24`, which is 256 MiB of
`ComplexF64`. No bound ever forms `d^n`, so asking for the dense vector of a
100-qutrit state fails cleanly instead of overflowing:

```jldoctest dense
julia> state_vector(StabilizerTableau(3, 100; state = :ghz))
ERROR: ArgumentError: state_vector would store 3^100 entries, which exceeds maxentries = 16777216. Pass a larger maxentries to allow it.
```

Exact phases need exact `Int` arithmetic, so the conversions accept `d` only
up to the safe dimension for the given `n` (see
[Supported arithmetic envelope](@ref)), where the constructors merely warn.
