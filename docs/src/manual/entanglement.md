```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Entanglement Measures

Every entropy-like quantity of a stabilizer state is a whole number of
``\log d`` units, and the package returns that number as an `Int`: `1` means
``\log d``, so one ebit for qubits and one trit for qutrits. All functions on
this page read only the X and Z rows, so they work with `storephase=false`, and
they leave the represented state unchanged.

| Function | Quantity |
|:---|:---|
| [`entanglement_entropy`](@ref)`(tab, A)` | von Neumann entropy ``S(A)`` |
| [`entanglement_profile`](@ref)`(tab)` | ``S(1{:}l)`` for every prefix `l = 0, …, n` |
| [`entanglement_negativity`](@ref)`(tab, A, B)` | logarithmic negativity ``\mathcal{E}(A{:}B)`` |
| [`mutual_information`](@ref)`(tab, A, B)` | ``I(A{:}B) = S(A) + S(B) - S(AB)`` |
| [`tripartite_information`](@ref)`(tab, A, B, C)` | ``I_3(A{:}B{:}C)`` |
| [`conditional_mutual_information`](@ref)`(tab, A, B, C)` | ``I(A{:}B \mid C)`` |

Subsystems are vectors of 1-based qudit indices. Every qudit outside the named
subsystems is traced out.

## Entropy

[`entanglement_entropy`](@ref) returns ``S(A)`` for pure and mixed states. For a
pure state ``S(A) = S(B)`` with `B` the complement, and the function computes
the rank on the smaller side. For a mixed state it is the entropy of the
reduced state, which also counts the mixedness of the whole: the maximally
mixed state has ``S(A) = |A|``.

For a stabilizer state the entropy is fixed by counting: with ``G`` the
stabilizer group and ``G_A`` its elements supported on `A`,

```math
S(A) = -\operatorname{tr}(\rho_A \log_d \rho_A) = |A| - \log_d |G_A|,
```

which is why it is always a whole number.

```jldoctest entropy
julia> ghz = StabilizerTableau(3, 4; state = :ghz);

julia> entanglement_entropy(ghz, [1]), entanglement_entropy(ghz, [1, 2])
(1, 1)

julia> entanglement_entropy(StabilizerTableau(3, 4), [1, 2])  # maximally mixed
2
```

[`entanglement_profile`](@ref) returns the entropies of all prefixes
`1:l` in one elimination, at the cost of a single entropy. An optional site
order replaces the prefixes `1:l` with `order[1:l]`, for example to obtain the
arcs of a periodic chain. [`entanglement_profile!`](@ref) writes into an
existing vector without allocating.

```jldoctest entropy
julia> entanglement_profile(ghz)
5-element Vector{Int64}:
 0
 1
 1
 1
 0
```

[Entropy across every cut](@ref) builds a profile in a larger example.

## Negativity

The entropy of a mixed state does not separate entanglement from classical
correlation. [`entanglement_negativity`](@ref) does: it is the logarithmic
negativity of the reduced state of ``A \cup B``,

```math
\mathcal{E}(A{:}B) = \log_d \bigl\lVert \rho_{AB}^{T_B} \bigr\rVert_1,
```

where ``T_B`` is the partial transpose on `B` and ``\lVert \cdot \rVert_1`` the
trace norm, and it vanishes on classical correlations. For a pure state and `B` the complement of `A` it equals
``S(A)``. In a three-qutrit GHZ state, any two qutrits share only a classical
correlation once the third is traced out:

```jldoctest negativity
julia> ghz = StabilizerTableau(3, 3; state = :ghz);

julia> entanglement_negativity(ghz, [1], [2]), mutual_information(ghz, [1], [2])
(0, 1)

julia> entanglement_negativity(ghz, [1]), entanglement_entropy(ghz, [1])
(1, 1)
```

## Information measures

The three information measures are combinations of entropies:

```math
\begin{aligned}
I(A{:}B) &= S(A) + S(B) - S(AB), \\
I(A{:}B \mid C) &= S(AC) + S(BC) - S(C) - S(ABC), \\
I_3(A{:}B{:}C) &= S(A) + S(B) + S(C) - S(AB) - S(AC) - S(BC) + S(ABC).
\end{aligned}
```

[`mutual_information`](@ref) counts all correlations between `A` and `B`, quantum
and classical. [`conditional_mutual_information`](@ref) is the correlation
between `A` and `B` that `C` does not account for; it is nonnegative and zero
exactly for a quantum Markov chain ``A - C - B``.
[`tripartite_information`](@ref) can take either sign: `+1` for a GHZ state on
three single qudits, negative for scrambled states. The three are related by
``I_3(A{:}B{:}C) = I(A{:}B) - I(A{:}B \mid C)``.

```jldoctest information
julia> ghz = StabilizerTableau(3, 4; state = :ghz);

julia> mutual_information(ghz, [1], [2])
1

julia> conditional_mutual_information(ghz, [1], [2], [3])  # qudit 3 explains it
0

julia> tripartite_information(ghz, [1], [2], [3])
1
```

[Correlations in mixed states](@ref) uses these to tell quantum from classical
correlations along a circuit.

## Cost and scratch space

Each function runs in at most ``O(n m^2)`` time, cubic in `n`. The
multi-entropy quantities share one elimination instead of calling
[`entanglement_entropy`](@ref) repeatedly, which makes them cheaper on mixed
states. All of them overwrite the tableau's scratch workspaces but never its
generators, phases, destabilizers or canonicalization caches.

Subsystems are checked to hold distinct sites in `1:n`, and the subsystems of
one call to be disjoint, throwing `ArgumentError` otherwise.
[`entanglement_entropy`](@ref) alone lets you skip that ``O(n)`` check with
`check=false`; an invalid subsystem then gives a wrong result or an
out-of-bounds access.
