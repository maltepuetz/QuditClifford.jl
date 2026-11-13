```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Correlations in mixed states

Entanglement entropy alone does not separate quantum from classical
correlations once a state is mixed. This page compares the negativity with
the mutual, tripartite and conditional mutual information.

## Entanglement in mixed states: negativity

Mutual information ``I(A{:}B) = S(A) + S(B) - S(AB)``, computed by
[`mutual_information`](@ref), counts classical correlations as well as
entanglement. [`entanglement_negativity`](@ref) returns the logarithmic
negativity ``\mathcal{E}(A{:}B) = \log_d \lVert \rho_{AB}^{T_B} \rVert_1`` of the
state of ``A \cup B`` with the other qudits traced out, which vanishes when the correlations are classical.
A qutrit GHZ state shows the difference: two of its three qutrits share one
unit of mutual information but no entanglement.

```jldoctest
julia> ghz = StabilizerTableau(3, 3; state=:ghz);

julia> mutual_information(ghz, [1], [2])
1

julia> entanglement_negativity(ghz, [1], [2])
0

julia> entanglement_negativity(ghz, [1], [2, 3])
1
```

With only one subsystem, the second is its complement. For a pure state the
negativity then equals the entanglement entropy:

```jldoctest
julia> ghz = StabilizerTableau(3, 3; state=:ghz);

julia> entanglement_negativity(ghz, [1]) == entanglement_entropy(ghz, [1])
true
```

The negativity uses the same log-`d` units as the entropy. It takes at most
cubic time in the number of sites, allocates nothing, works for both tableau
types and with `storephase=false`, and leaves the generators unchanged.

## Mutual and tripartite information

[`mutual_information`](@ref)`(tab, A, B)` returns `S(A) + S(B) - S(AB)`, and
[`tripartite_information`](@ref)`(tab, A, B, C)` returns

```math
I_3(A:B:C) = S_A + S_B + S_C - S_{AB} - S_{AC} - S_{BC} + S_{ABC},
```

both in log-`d` units with every other qudit traced out. They equal the sums
of [`entanglement_entropy`](@ref) calls but share the eliminations, which
makes them about twice as fast on mixed states and as fast on pure ones.

The sign of ``I_3`` separates two kinds of correlation. In a GHZ state, the
information shared by any two qudits is the same classical value, so it is
counted twice and ``I_3 > 0``. In a scrambled state, information about one
part is spread over the others jointly, and ``I_3 < 0``. The four-qutrit
state ``\sum_{i,j} |i, j, i+j, i+2j\rangle``, in which every pair of qutrits
is maximally entangled with the other pair, is the extreme case:

```jldoctest
julia> ghz = StabilizerTableau(3, 4; state=:ghz);

julia> tripartite_information(ghz, [1], [2], [3])
1

julia> ame = StabilizerTableau(3, 4; state=:product);

julia> for g in (Fourier(1), Fourier(2), SUM(1, 3), SUM(2, 3), SUM(1, 4), SUM(2, 4, 2))
           apply!(ame, g)
       end

julia> mutual_information(ame, [1], [2]), mutual_information(ame, [1], [2, 3])
(0, 2)

julia> tripartite_information(ame, [1], [2], [3])
-2
```

With one subsystem, [`mutual_information`](@ref) takes the second to be its
complement; for a pure state that gives twice the entanglement entropy. Like
the negativity, both functions allocate nothing, leave the generators
unchanged and work for both tableau types and with `storephase=false`.

## Conditional mutual information

[`conditional_mutual_information`](@ref)`(tab, A, B, C)` returns

```math
I(A:B|C) = S_{AC} + S_{BC} - S_C - S_{ABC},
```

the correlation between `A` and `B` that `C` does not account for, again in
log-`d` units with every other qudit traced out. It is never negative, and it
vanishes exactly when ``A - C - B`` is a quantum Markov chain. Conditioning
can remove correlation or create it. In the GHZ state, qudit 3 holds the same
classical value that qudits 1 and 2 share, so given qudit 3 nothing is left:

```jldoctest
julia> ghz = StabilizerTableau(3, 4; state=:ghz);

julia> mutual_information(ghz, [1], [2]), conditional_mutual_information(ghz, [1], [2], [3])
(1, 0)
```

In the absolutely maximally entangled state above, qudits 1 and 2 are
uncorrelated on their own, but together with qudit 3 they hold two units of
correlation. The difference is the tripartite information,
``I_3(A:B:C) = I(A:B) - I(A:B|C)``:

```jldoctest
julia> ame = StabilizerTableau(3, 4; state=:product);

julia> for g in (Fourier(1), Fourier(2), SUM(1, 3), SUM(2, 3), SUM(1, 4), SUM(2, 4, 2))
           apply!(ame, g)
       end

julia> mutual_information(ame, [1], [2]), conditional_mutual_information(ame, [1], [2], [3])
(0, 2)

julia> tripartite_information(ame, [1], [2], [3])
-2
```

The result equals four [`entanglement_entropy`](@ref) calls but needs only
one elimination, which makes it several times faster on mixed states. With
`C` empty it is the mutual information of `A` and `B`.
