```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Conventions

This page defines the operators, orderings and signs that every other page and
docstring relies on. Each convention is pinned by an example that runs with the
documentation build.

## Dimensions and the phase unit

The local dimension `d` is prime: `d = 2` for qubits, and any odd prime for
qudits. Constructors reject composite dimensions.

Write ``\omega = e^{2\pi i/d}``. At `d = 2` this is ``\omega = -1``. Phases are
stored as integer exponents of a unit ``\zeta``:

| Dimension | Phase unit ``\zeta`` | Exponent reduced mod |
|:---|:---|:---|
| odd prime `d` | ``\omega = e^{2\pi i/d}`` | `d` |
| `d = 2` | ``i`` | `4` |

Qubits need the finer unit `i` because the Clifford group maps Hermitian Paulis
such as `Y` to products like `iXZ`. A stored phase exponent `k` therefore means
``\omega^k`` for odd `d` and ``i^k`` for qubits. With `storephase=false` no
phases are stored, and phase-sensitive queries return `0` by convention.

## Weyl operators

On one qudit with computational basis ``|0\rangle, \ldots, |d-1\rangle``,

```math
X|j\rangle = |j + 1 \bmod d\rangle, \qquad Z|j\rangle = \omega^{j}|j\rangle,
\qquad ZX = \omega\, XZ.
```

At `d = 2` these are the Pauli matrices `X` and `Z`.

## Pauli operators and their ordering

A local factor with exponents `(x, z)` is ``X^x Z^z``, with `X` to the left. An
operator on `n` qudits with exponent vector `v` and phase exponent `k` is

```math
P(v, k) = \zeta^{k}\, \bigotimes_{q=1}^{n} X_q^{x_q} Z_q^{z_q}.
```

The order matters as soon as both exponents are nonzero. For qubits,
``XZ = -iY``, so `SinglePauli(q, 1, 1)` is not `Y` and not Hermitian. `Y` is
``iXZ``, which is `SinglePauli(q, 1, 1, 1)`. On the `+1` eigenstate of `Y`,
[`expect_int!`](@ref) returns `0` (that is, ``i^0``) for the first and `3` (``i^3 = -i``) for
the second:

```jldoctest conventions-y
julia> tab = StabilizerTableau(2, 1; state = :product, basis = :Y);

julia> expect_int!(tab, SinglePauli(1, 1, 1, 1))  # Y = iXZ
0

julia> expect_int!(tab, SinglePauli(1, 1, 1))     # XZ = -iY
3
```

For odd `d`, every ``X^x Z^z`` with phase exponent `0` already has eigenvalues
in ``\{\omega^t\}``, so no such correction is needed.

## Exponent vectors

Dense Paulis ([`GeneralPauli`](@ref), and the plain integer vectors accepted by
[`measure!`](@ref) and [`expect!`](@ref)) order all X exponents before all Z
exponents:

```math
v = (x_1, \ldots, x_n, z_1, \ldots, z_n),
```

and a plain vector may carry the phase exponent as an extra entry `2n + 1`.

## Tableau layout

A tableau stores generators in **columns**. For `n` qudits, `tab.stab` has
rows `1:n` for the X exponents, rows `n+1:2n` for the Z exponents, and, when
`storephase=true`, row `2n + 1` for the phase exponents. The first `m` columns
are the active generators, with ``0 \leq m \leq n``. The remaining columns are
zeroed capacity for measurements that raise `m`. A pure state has `m = n`; a
mixed state has `m < n`. [Tableaux](tableaux.md) covers construction from
a raw matrix.

## The symplectic form

Two exponent vectors `a` and `b` pair as

```math
\langle a, b \rangle = \sum_{q=1}^{n} \left( a_{x,q}\, b_{z,q} - a_{z,q}\, b_{x,q} \right)
= a^{\mathsf T} \Omega\, b \pmod d,
\qquad
\Omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}.
```

With this sign, ``P(a)\,P(b) = \omega^{-\langle a, b\rangle}\, P(b)\,P(a)``:
for `a = X` and `b = Z`, ``\langle a, b\rangle = 1`` and ``XZ = \omega^{-1} ZX``.
Two Paulis commute exactly when ``\langle a, b\rangle \equiv 0 \pmod d``, and a
tableau's generators must commute pairwise. A matrix `F` that maps exponent
vectors is symplectic when ``F^{\mathsf T} \Omega F = \Omega \pmod d``; the
Clifford unitaries are exactly the maps of this form, together with a phase
for each image.

The sign shows up in conjugation by a Pauli. [`PauliGate`](@ref)`(q, x, z)`
maps ``P(v) \mapsto \omega^{-\langle (x, z), v\rangle} P(v)``. Conjugating `Z`
by `X` at `d = 3` gives ``\omega^{-1} Z = \omega^2 Z``:

```jldoctest
julia> conjugate(PauliGate(1, 1, 0), SinglePauli(1, 0, 1), 1; d = 3)
[phase=2] Z₁
```

## Action direction

A Clifford unitary `U` acts on states and operators by conjugation:

- [`apply!`](@ref)`(tab, U)` replaces the state ``\rho`` by
  ``U \rho\, U^\dagger``, which maps every stabilizer generator `g` to
  ``U g\, U^\dagger``;
- [`conjugate`](@ref)`(U, P)` returns ``U P\, U^\dagger``;
- the Heisenberg picture of state evolution by `U` is therefore
  `conjugate(inv(U), P)`;
- `U ∘ V` is the product ``UV``: it applies `V` first, then `U`.

A [`CliffordOperator`](@ref) stores `U` through the images of the generators:
column `i` of `F` is the exponent vector of ``U g_i U^\dagger`` for
``g_i \in (X_{t_1}, \ldots, X_{t_k}, Z_{t_1}, \ldots, Z_{t_k})``.

Applying `Fourier` and then `Phase` is the same as applying
`Phase ∘ Fourier`, and differs from the other order:

```jldoctest conventions-compose
julia> F, P = CliffordOperator(Fourier(1), 3), CliffordOperator(Phase(1), 3);

julia> a = StabilizerTableau(3, 1; state = :product, basis = :Z);

julia> apply!(apply!(a, F), P);

julia> a == apply!(StabilizerTableau(3, 1; state = :product, basis = :Z), P ∘ F)
true

julia> a == apply!(StabilizerTableau(3, 1; state = :product, basis = :Z), F ∘ P)
false
```

## Computational basis

[`state_vector`](@ref) and [`density_matrix`](@ref) index the computational
basis with qudit 1 as the most significant digit:

```math
\mathrm{index}(c_1, \ldots, c_n) = 1 + \sum_{q=1}^{n} c_q\, d^{\,n-q},
```

so the vector of a product state is `kron` of its factors in qudit order, and a
label prints in reading order. This is independent of the tableau layout, which
groups every X row before every Z row. [`ket`](@ref) fixes the global phase by
making the amplitude of the lexicographically first label real and positive,
and [`state_vector`](@ref) keeps that choice.

On basis states, the Fourier gate is

```math
\mathrm{Fourier}\,|k\rangle = \frac{1}{\sqrt d} \sum_{j=0}^{d-1} \omega^{jk} |j\rangle,
```

which at `d = 2` is the Hadamard gate. Starting from ``|1\rangle`` (prepared
by conjugating ``|0\rangle`` with `X`) at `d = 3`:

```jldoctest
julia> tab = StabilizerTableau(3, 1; state = :product, basis = :Z);

julia> apply!(tab, PauliGate(1, 1, 0));  # |0⟩ ↦ |1⟩

julia> ket(apply!(tab, Fourier(1)))
StabilizerKet (d = 3, n = 1, 3 terms):
  (|0⟩ + ω₃|1⟩ + ω₃²|2⟩)/√3
```

Here `ω₃` is ``\omega`` at `d = 3`.

The action of every named gate on basis states is listed in
[The gate set](@ref).

## Outcome and expectation encodings

Measurement outcomes and integer expectation values are phase exponents, but
in two different units:

| Function | Odd prime `d` | `d = 2` |
|:---|:---|:---|
| [`measure!`](@ref) | `t ∈ 0:(d-1)`, eigenvalue ``\omega^t`` | `t ∈ 0:1`, eigenvalue ``(-1)^t`` |
| [`expect_int!`](@ref) | `k`, ``\langle P\rangle = \omega^k``; `-1` if zero | `k ∈ 0:3`, ``\langle P\rangle = i^k``; `-1` if zero |
| [`expect!`](@ref) | the complex value ``\langle P\rangle`` | the complex value ``\langle P\rangle`` |

A [`measure!`](@ref) outcome uses the same encoding in every branch and for the
`outcome` keyword, so a repeated measurement returns the outcome of the first.
For qubits it counts in units of ``-1``, while [`expect_int!`](@ref) counts in
units of ``i``, so eigenvalue ``-1`` is `1` for the first and `2` for the
second:

```jldoctest
julia> tab = DestabilizerTableau(2, 1; state = :product, basis = :Z);

julia> measure!(tab, SinglePauli(1, 1, 0); outcome = 1), measure!(tab, SinglePauli(1, 1, 0))
(1, 1)

julia> expect_int!(tab, SinglePauli(1, 1, 0))
2
```

With `storephase=false`, a deterministic measurement returns `0` and
[`expect_int!`](@ref) returns `0` for an operator in the stabilizer group,
because the phase was not stored. [Measurements, Expectations and Channels](@ref)
shows all three functions in use.
