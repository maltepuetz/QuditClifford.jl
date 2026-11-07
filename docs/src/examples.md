```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Examples

This chapter combines the basic state, operator, measurement, expectation, and
entropy interfaces into complete workflows. The comments first describe the
physical question and then explain how the function arguments encode it.

## A two-qutrit measurement

We begin with two independent qutrits in the X basis and measure the correlated
operator ``Z_1Z_2``. Each initial X stabilizer fails to commute with the
measurement, so this example exercises the multi-generator update path.

```jldoctest qutrit-measurement
# Construct two qutrits in an X-basis product state.
# d = 3 selects qutrits; n = 2 creates two sites; basis=:X selects X eigenstates.
julia> state = DestabilizerTableau(3, 2; state=:product, basis=:X)
Destabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 2
    Generators:       m = 2
    Tableau:
     Stabilizers:
      X     Z
     1 0 | 0 0 | 0
     0 1 | 0 0 | 0
     ---------
     Destabilizers:
      X     Z
     0 0 | 2 0
     0 0 | 0 2
```

```jldoctest qutrit-measurement
# Build Z₁Z₂ from two (qudit, X exponent, Z exponent) triplets.
# The triplets (1, 0, 1) and (2, 0, 1) put one Z on each qutrit.
julia> z1z2 = DoublePauli(1, 0, 1, 2, 0, 1)
Z₁ Z₂
```

```jldoctest qutrit-measurement
# Measure Z₁Z₂ and choose outcome exponent 2 to keep the example reproducible.
# For d = 3, outcome 2 represents the eigenvalue ω².
julia> measure!(state, z1z2; outcome=2)
2
```

```jldoctest qutrit-measurement
# measure! mutates the tableau, so display state again to inspect the update.
# Z₁Z₂ is now a stabilizer whose phase column encodes the sampled eigenvalue.
julia> state
Destabilizer Tableau:
    Qudit dimension:  d = 3
    Number of Qudits: n = 2
    Generators:       m = 2
    Tableau:
     Stabilizers:
      X     Z
     0 0 | 1 1 | 1
     2 1 | 0 0 | 0
     ---------
     Destabilizers:
      X     Z
     1 0 | 0 0
     0 0 | 0 2
```

```jldoctest qutrit-measurement
# Projection fixes ⟨Z₁Z₂⟩ to ω² instead of zero.
julia> expect!(state, z1z2)
-0.5000000000000004 - 0.8660254037844385im
```

The phase entry `1` on the displayed ``Z_1Z_2`` generator means that
``\omega Z_1Z_2`` stabilizes the post-measurement state. Equivalently,
``Z_1Z_2`` has eigenvalue ``\omega^{-1}=\omega^2``, matching the returned
outcome and expectation value.

## Following entanglement through a measurement

A four-qubit GHZ state has one bit of entropy across every nontrivial cut. We
measure X on the first qubit and compare the entropy before and after. The local
measurement separates qubit 1 while leaving the remaining GHZ cluster
entangled.

```jldoctest ghz-measurement
# Construct a four-qubit GHZ state.
# d = 2 selects qubits; n = 4 creates four sites; :ghz selects the GHZ preset.
julia> state = DestabilizerTableau(2, 4; state=:ghz)
Destabilizer Tableau:
    Qudit dimension:  d = 2
    Number of Qudits: n = 4
    Generators:       m = 4
    Tableau:
     Stabilizers:
     -- X --   -- Z --
     1 1 1 1 | 0 0 0 0 | 0
     0 0 0 0 | 1 1 0 0 | 0
     0 0 0 0 | 0 1 1 0 | 0
     0 0 0 0 | 0 0 1 1 | 0
     -----------------
     Destabilizers:
     -- X --   -- Z --
     0 0 0 0 | 1 0 0 0
     1 0 0 0 | 0 0 0 0
     1 1 0 0 | 0 0 0 0
     1 1 1 0 | 0 0 0 0
```

```jldoctest ghz-measurement
# Compute entropy across the cut {1,2}|{3,4} before measurement.
# The vector [1, 2] identifies the qubits in the first subsystem.
julia> entanglement_entropy(state, [1, 2])
1
```

```jldoctest ghz-measurement
# Construct X₁: qubit 1 with X exponent 1 and Z exponent 0.
julia> x1 = SinglePauli(1, 1, 0)
X₁
```

```jldoctest ghz-measurement
# X₁ does not commute with every GHZ stabilizer, so measurement changes state.
# outcome=0 selects the +1 eigenvalue and makes the walkthrough reproducible.
julia> measure!(state, x1; outcome=0)
0
```

```jldoctest ghz-measurement
# The projection fixes X₁ to +1, so its post-measurement expectation is +1.
julia> expect!(state, x1)
1.0 + 0.0im
```

```jldoctest ghz-measurement
# Qubit 1 is now separated, so its single-site entropy falls to zero.
julia> entanglement_entropy(state, [1])
0
```

```jldoctest ghz-measurement
# Qubits 2 and 3 remain inside the entangled three-qubit GHZ cluster.
julia> entanglement_entropy(state, [2, 3])
1
```

Entropy is an integer in log-`d` units. For qubits it is measured in bits; for
qutrits, one unit means ``\log(3)``.

## Entropy across every cut

Use [`entanglement_profile`](@ref) to compute the entropy of every prefix of a
chain in one elimination. Entry `l + 1` is the entropy of sites `1:l`, so the
vector includes both the empty prefix and the whole chain.

```jldoctest
julia> state = DestabilizerTableau(3, 4; state=:ghz);

julia> entanglement_profile(state)
5-element Vector{Int64}:
 0
 1
 1
 1
 0

julia> entanglement_profile(state) == [entanglement_entropy(state, 1:l) for l in 0:4]
true
```

The profile uses the same log-`d` units as [`entanglement_entropy`](@ref).
It takes at most cubic time in the number of sites, compared with up to
quartic time for the loop above. It also works with `storephase=false`.

For repeated measurements, reuse an output vector with
[`entanglement_profile!`](@ref) to avoid allocating a result each time:

```jldoctest
julia> state = DestabilizerTableau(3, 4; state=:ghz);

julia> profile = Vector{Int}(undef, state.n + 1);

julia> entanglement_profile!(profile, state) === profile
true

julia> profile == [0, 1, 1, 1, 0]
true

julia> measure!(state, SinglePauli(1, 0, 1); outcome=0);

julia> entanglement_profile!(profile, state);

julia> profile == [0, 0, 0, 0, 0]
true
```

The destination must be a writable, one-based `Int` vector of length `n + 1`
that does not share storage with the tableau. Views into independent arrays
also work. The function overwrites every entry and returns the destination.

For pure states, a prefix and its complementary suffix have equal entropy.
This does **not** imply that the prefix profile is symmetric: a Bell pair on
sites 1 and 2 with an unentangled site 3 has profile `[0, 1, 0, 0]`.

Mixed states are supported too. Their last entry is `n - m`, the entropy of
the whole state; a maximally mixed chain has profile `[0, 1, ..., n]`.
These mixed-state subsystem entropies include classical uncertainty and are
not an entanglement measure.

Pass a site order to take the prefixes along a permutation of the sites
instead; entry `l + 1` is then the entropy of `order[1:l]`. On a periodic
chain, a cyclic shift gives every arc that starts at a chosen site. Here two
Bell pairs sit on the opposite bonds `(1, 4)` and `(2, 3)` of a ring of four:

```jldoctest
julia> ring = StabilizerTableau(3, 4; state=:product);

julia> for (a, b) in ((1, 4), (2, 3))
           apply!(ring, Fourier(a))
           apply!(ring, SUM(a, b))
       end

julia> entanglement_profile(ring) == [0, 1, 2, 1, 0]
true

julia> entanglement_profile(ring, [2, 3, 4, 1]) == [0, 1, 0, 1, 0]
true

julia> entanglement_profile(ring, [2, 3, 4, 1]) ==
           [entanglement_entropy(ring, [2, 3, 4, 1][1:l]) for l in 0:4]
true
```

The cost and the allocation behaviour are those of the natural order, and
[`entanglement_profile!`](@ref) accepts the same `order` argument.

## Entanglement in mixed states: negativity

Mutual information `S(A) + S(B) - S(AB)`, computed by
[`mutual_information`](@ref), counts classical correlations as well as
entanglement. [`entanglement_negativity`](@ref) returns the
logarithmic negativity `log_d ‖ρ_ABᵀᴮ‖₁` of the state of `A ∪ B` with the
other qudits traced out, which vanishes when the correlations are classical.
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

## Purification by random two-qudit measurements

A maximally mixed stabilizer state has no stabilizer generators. Projective
measurements gradually add information: a measurement that commutes with the
current generators and lies outside their span increases the tableau rank,
whereas a noncommuting measurement replaces a generator without changing the
rank. Repeating random measurements eventually produces a pure state.

We demonstrate this process on an even periodic chain. Odd and even layers use
the alternating brickwork matchings

```math
(1,2),(3,4),\ldots
\qquad\text{and}\qquad
(2,3),(4,5),\ldots,(n,1).
```

Every bond is measured with a random two-qudit Pauli. Each local factor is
sampled independently from the ``d^2-1`` nonidentity operators ``X^xZ^z``.
The following helpers generate the measurements and run one trajectory.

```@example purification
using QuditClifford
using Random
using Statistics

# Draw a nonidentity Pauli XˣZᶻ on one site.
function random_local_pauli(d)
    x = rand(0:(d - 1))
    z = rand(0:(d - 1))

    while iszero(x) && iszero(z)
        x = rand(0:(d - 1))
        z = rand(0:(d - 1))
    end

    return x, z
end

# Measure one alternating nearest-neighbour matching of the periodic chain.
function random_brickwork_layer!(tab, d, n, layer)
    first_site = isodd(layer) ? 1 : 2

    for offset in 0:(n ÷ 2 - 1)
        # mod1 closes the final even-layer bond from site n back to site 1.
        i = mod1(first_site + 2offset, n)
        j = mod1(i + 1, n)

        x_i, z_i = random_local_pauli(d)
        x_j, z_j = random_local_pauli(d)
        operator = DoublePauli(i, x_i, z_i, j, x_j, z_j)

        measure!(tab, operator)
    end

    return nothing
end


# Follow one maximally mixed state until its tableau reaches full rank.
function purification_trajectory(
    d,
    n;
    max_layers=10000,
)
    iseven(n) || throw(ArgumentError("n must be even"))

    # Phases are unnecessary because only the rank of the tableau is tracked.
    tab = DestabilizerTableau(d, n; state=:mixed, storephase=false)
    @assert tab.m == 0

    # Pre-filling with n extends a completed trajectory along its pure plateau.
    rank_history = fill(n, max_layers + 1)
    rank_history[1] = tab.m

    previous_rank = tab.m
    for layer in 1:max_layers
        random_brickwork_layer!(tab, d, n, layer)

        # A projective measurement can add or replace a generator, but not remove one.
        @assert previous_rank <= tab.m <= n
        previous_rank = tab.m
        rank_history[layer + 1] = tab.m

        if tab.m == n
            return rank_history, layer
        end
    end

    # max_layers + 1 marks a trajectory that remains mixed at the cutoff.
    return rank_history, max_layers + 1
end
nothing # hide
```

Here `tab.m` is the number of independent stabilizer generators. The initial
state has `m = 0`, and `m = n` signals that the state is pure. Since phases do
not affect this rank, `storephase=false` avoids unnecessary phase bookkeeping.

The next helper repeats the circuit and records the complete rank history of
every trajectory. Averaging these histories reveals the typical purification
process, while counting ranks below `n` gives the fraction still mixed.

```@example purification
# Collect every rank history and the purification-time distribution.
function purification_ensemble(
    d,
    n;
    max_layers=10000,
    trajectories=1000,
)
    rank_histories = Matrix{Int}(undef, max_layers + 1, trajectories)
    purification_layers = Vector{Int}(undef, trajectories)

    for trajectory in 1:trajectories
        rank_history, purification_layers[trajectory] = purification_trajectory(
            d,
            n;
            max_layers=max_layers,
        )
        rank_histories[:, trajectory] = rank_history
    end

    mean_rank = mean(rank_histories, dims=2)[:]
    fraction_mixed = mean(rank_histories .< n, dims=2)[:]

    return rank_histories, purification_layers, mean_rank, fraction_mixed
end
nothing # hide
```

We now compare four prime local dimensions. The left panel shows the mean rank
across the trajectory ensemble; the right panel shows the fraction that has
not reached `m = n`. The logarithmic horizontal axes begin at layer one, while
the unmeasured states at layer zero have `m = 0`.

```@example purification
using CairoMakie

include(joinpath(pkgdir(QuditClifford), "docs", "makie_theme.jl")) # hide

# Keep the chain fixed so differences come only from the local dimension.
n = 8
max_layers = 10000
trajectories = 1000
dimensions = (2, 3, 5, 7)

Random.seed!(1234) # hide
results = [
    purification_ensemble(
        d,
        n;
        max_layers=max_layers,
        trajectories=trajectories,
    ) for d in dimensions
]

layers = collect(1:max_layers)

figure = Figure(size=(720, 240), figure_padding=9)
rank_axis = Axis(
    figure[1, 1];
    xlabel=L"measurement layer $t$",
    ylabel=L"mean tableau rank $\langle m\rangle$",
    xscale=log10,
    limits=((1, max_layers), (0, n)),
)
mixed_axis = Axis(
    figure[1, 2];
    xlabel=L"measurement layer $t$",
    ylabel=L"fraction with $m<n$",
    xscale=log10,
    limits=((1, max_layers), (0, 1)),
)

for (index, d) in enumerate(dimensions)
    _, _, mean_rank, fraction_mixed = results[index]

    lines!(
        rank_axis,
        layers,
        mean_rank[2:end];
        label=L"d = %$d",
        color=qudit_colors[index],
        linestyle=:solid,
    )
    lines!(
        mixed_axis,
        layers,
        fraction_mixed[2:end];
        color=qudit_colors[index],
        linestyle=:solid,
    )
end

axislegend(rank_axis; position=:rb)
figure
```

The qudit dimension changes the commutation and linear-dependence statistics
of the randomly sampled measurements, so the purification-time distributions
separate clearly. These curves describe this finite chain and this particular
measurement ensemble; they should not be interpreted as universal constants
of qudit dimension.

## Measurements against boundary dephasing

Weinstein, Bao and Altman study a monitored circuit coupled to an environment
at its two ends: Z. Weinstein, Y. Bao, E. Altman, *Measurement-Induced
Power-Law Negativity in an Open Monitored Quantum Circuit*,
[Phys. Rev. Lett. 129, 080501 (2022)](https://doi.org/10.1103/PhysRevLett.129.080501),
[arXiv:2202.12905](https://arxiv.org/abs/2202.12905). An open chain of `L`
qudits starts in ``|0\ldots0\rangle``. Every time step applies one layer of a
brick wall of random two-qudit Clifford gates, measures each qudit in the `Z`
basis with probability `p` and records the outcome, and then dephases the two
end qudits completely in the `Z` basis,

```math
\mathcal{D}_i[\rho] = \sum_{a=0}^{d-1} P_i^a\,\rho\,P_i^a,
\qquad P_i^a = |a\rangle\langle a|_i,
```

which is [`dephase!`](@ref). The dephasing acts as a bath that the chain keeps
leaking information into, so the state stays mixed, and the measurements
compete with it.

The circuit, the initial state, the `Z` measurements and the dephased end
sites follow the paper, as does the order within a step that its supplement
gives: gates, then measurements, then dephasing. The paper's numerics are for
qubits. Its supplement dephases on even time steps only, while its first
figure dephases between all layers; for the even `L` used here the two agree
exactly, because the odd-step gates are the only ones that touch the end sites
and the `Z` measurements commute with the dephasing. The paper evolves for ``O(L)`` time
steps without a prefactor; the depth of `4L` steps is our choice.

The first function runs one trajectory and returns the logarithmic negativity
and the mutual information between the left and right halves of the chain.
Phases do not affect either quantity, so the tableau stores none. The gates are
preallocated, one per bond, and refilled with fresh random Cliffords as the
circuit runs.

```@example boundary_dephasing
using QuditClifford
using Random
using Statistics

# Run one trajectory of the boundary-dephased circuit on L qudits of dimension d.
# p is the measurement probability per site and step, depth the number of
# brick layers, and gates[i] the preallocated Clifford on the bond (i, i + 1).
function boundary_dephasing_trajectory(rng, d, L, gates; p, depth=4L)
    tab = DestabilizerTableau(d, L; state=:Z, storephase=false)

    for step in 1:depth
        # Odd steps act on the bonds (1,2), (3,4), …; even steps on (2,3), (4,5), ….
        for i in (isodd(step) ? 1 : 2):2:(L - 1)
            apply!(tab, random_clifford!(rng, gates[i]))
        end

        # Measure Z on each site with probability p, drawing the outcome from rng.
        for i in 1:L
            if rand(rng) < p
                measure!(tab, SinglePauli(i, 0, 1); outcome=rand(rng, 0:(d - 1)))
            end
        end

        # Dephase both end sites in the Z basis; nothing is recorded.
        dephase!(tab, SinglePauli(1, 0, 1))
        dephase!(tab, SinglePauli(L, 0, 1))
    end

    left_half = 1:(L ÷ 2)
    return entanglement_negativity(tab, left_half), mutual_information(tab, left_half)
end
nothing # hide
```

The next helper averages both quantities over trajectories and returns their
means with standard errors. Both functions take the random number generator as
an argument, so one seed reproduces the whole figure.

```@example boundary_dephasing
# Average the half-chain negativity and mutual information over trajectories.
function boundary_dephasing_averages(rng, d, L; p, depth=4L, trajectories=100)
    gates = [random_clifford(rng, d, [i, i + 1]) for i in 1:(L - 1)]
    negativity = zeros(trajectories)
    information = zeros(trajectories)

    for t in 1:trajectories
        negativity[t], information[t] = boundary_dephasing_trajectory(
            rng, d, L, gates; p=p, depth=depth
        )
    end

    standard_error(x) = std(x) / sqrt(length(x))
    return (
        mean(negativity), standard_error(negativity),
        mean(information), standard_error(information),
    )
end
nothing # hide
```

In every trajectory both quantities are whole numbers in log-`d` units: bits
for qubits, trits for qutrits. The paper does not state its logarithm base, so the figure is not
compared with its fitted coefficients. The plot shows qubits on the left and
qutrits, which the paper does not simulate, on the right, for no measurements,
for `p = 0.1`, below the qubit transition at ``p_c \approx 0.16``, and for
`p = 0.3`, above it. Every point averages 200 trajectories, and the error
bars are one standard error.

```@example boundary_dephasing
using CairoMakie

include(joinpath(pkgdir(QuditClifford), "docs", "makie_theme.jl")) # hide

rng = Xoshiro(2022)
dimensions = (2, 3)
probabilities = (0.0, 0.1, 0.3)
chain_lengths = [8, 16, 24, 32]
trajectories = 200

# results[(d, p)] holds one (E, ΔE, I, ΔI) tuple per chain length.
results = Dict(
    (d, p) => [
        boundary_dephasing_averages(rng, d, L; p=p, trajectories=trajectories)
        for L in chain_lengths
    ] for d in dimensions for p in probabilities
)

figure = Figure(size=(720, 420), figure_padding=9)
titles = (L"qubits, $d = 2$", L"qutrits, $d = 3$")
panels = [
    Axis(
        figure[row, column];
        title=row == 1 ? titles[column] : "",
        xlabel=row == 2 ? L"chain length $L$" : "",
        ylabel=column == 1 ? (
            row == 1 ? L"negativity $\langle E\rangle$ [$\log d$]" :
            L"mutual information $\langle I\rangle$ [$\log d$]"
        ) : "",
        xticks=chain_lengths,
    ) for row in 1:2, column in 1:2
]
linkyaxes!(panels[1, :]...)
linkyaxes!(panels[2, :]...)

for (column, d) in enumerate(dimensions), (index, p) in enumerate(probabilities)
    averages = results[(d, p)]
    # Row 1 plots the negativity (E, ΔE), row 2 the mutual information (I, ΔI).
    for (row, (value, error)) in enumerate(((1, 2), (3, 4)))
        means = getindex.(averages, value)
        errors = getindex.(averages, error)
        color = qudit_colors[index]
        label = L"p = %$p"
        lines!(panels[row, column], chain_lengths, means; color=color, label=label)
        errorbars!(panels[row, column], chain_lengths, means, errors; color=color,
            whiskerwidth=6)
        scatter!(panels[row, column], chain_lengths, means; color=color, label=label)
    end
end

axislegend(panels[1, 1]; position=:lt, merge=true)
figure
```

Without measurements both quantities vanish in every trajectory. The gates
carry each stabilizer generator to the ends, where the dephasing removes it,
nothing adds generators back, and after `4L` steps the chain is maximally
mixed. At `p = 0.1` the measurements keep restoring information, and both
quantities grow with `L` for qubits and qutrits alike. At `p = 0.3` the qubit
curves rise only over the smallest sizes and then stay flat within their error
bars, near half a bit of negativity: the area law above ``p_c``. The qutrit
curves at `p = 0.3` keep rising up to `L = 24` before they level off, which
suggests that the qutrit transition lies at a larger `p`; this example does not
locate it. The mutual information, which also counts classical correlations,
exceeds the negativity throughout but shows the same three regimes; the paper
reports it for the same two halves.

The paper's central result is that the negativity grows as ``L^{1/3}`` for
``0 < p < p_c``. At these sizes such a power law cannot be told apart from a
logarithm. The paper fits it over `L = 40` to `280` with 200 realizations;
raise `chain_lengths` toward those sizes for a scaling analysis rather than
reading an exponent off this figure.

## A deep measurement-only Ising circuit

As a larger example, consider the projective transverse-field Ising model on a
periodic qubit chain. Each layer independently measures
``Z_iZ_{i+1}^\dagger`` on every edge with probability `1-p`, followed by
``X_i`` on every site with probability `p`. For qubits,
``Z^\dagger=Z``. Starting from ``|+\rangle^{\otimes L}``, the two measurement
types grow and cut entangled clusters.

The first function generates one measurement trajectory.

```@example ising_circuit
using QuditClifford
using Random
using Statistics

# Simulate one trajectory of the periodic measurement-only Ising circuit.
# L is the number of qubits, p balances X against ZZ† measurements, and depth
# is the number of alternating measurement layers.
function measurement_only_ising(L;
    p=0.5,
    depth=8L,
)
    # Begin with |+⟩^L, an unentangled X-basis product state.
    state = DestabilizerTableau(2, L; state=:X)

    for _ in 1:depth
        # First measure ZZ† on every periodic edge.
        for i in 1:L
            if rand() < 1 - p
                # mod1 closes the edge from site L back to site 1.
                j = mod1(i + 1, L)
                zzdagger = DoublePauli(i, 0, 1, j, 0, 1)
                measure!(state, zzdagger)
            end
        end

        # Then measure X on every site.
        for i in 1:L
            if rand() < p
                xi = SinglePauli(i, 1, 0)
                measure!(state, xi)
            end
        end
    end

    return state
end
nothing # hide
```

In one dimension this circuit is critical at `p = 0.5`, where its steady-state
entanglement transition maps to bond percolation. The phases on either side are
area-law phases; at the transition, the trajectory-averaged entropy grows
logarithmically. See Lang and Büchler,
[Phys. Rev. B 102, 094204 (2020)](https://doi.org/10.1103/PhysRevB.102.094204).

The next helper averages the entropy arc over independent trajectories. A
single trajectory is noisy, whereas this ensemble average exposes the scaling
shared by typical trajectories. Qubit entropy is measured in bits.

```@example ising_circuit
# Estimate the mean entropy as a function of subsystem size.
# trajectories controls sampling accuracy; seed makes the ensemble reproducible.
function mean_entropy_arc(L;
    p=0.5,
    depth=8L,
    trajectories=500,
    seed=1234,
)
    Random.seed!(seed)

    # Include every nontrivial prefix cut, from one site through L - 1 sites.
    cuts = collect(1:(L - 1))
    entropies = zeros(length(cuts), trajectories)

    # Accumulate the entropy arc over measurement trajectories.
    for t in 1:trajectories
        state = measurement_only_ising(L; p=p, depth=depth)
        for j in cuts
            subsystem = collect(1:j)
            entropies[j, t] = entanglement_entropy(state, subsystem)
        end
    end

    mean_entropy = mean(entropies, dims=2)[:]
    return cuts, mean_entropy
end
nothing # hide
```

The following plot averages over ``250`` trajectories at the critical point. It calculates
all ``L-1`` nontrivial cuts. The line is a fit to the periodic chord-length form
``S(\ell)=S_0+(\widetilde c/3)\log_2[(L/\pi)\sin(\pi\ell/L)]``.

```@example ising_circuit
using CairoMakie

include(joinpath(pkgdir(QuditClifford), "docs", "makie_theme.jl")) # hide

# L sets the chain length; depth=8L reaches the steady-state regime.
# More trajectories reduce sample noise but increase documentation build time.
L = 32
cuts, entropy_arc = mean_entropy_arc(
    L; p=0.5, depth=8L, trajectories=250, seed=1234
)

# Fit every independently measured cut position.
log_chord = log2.((L / π) .* sinpi.(cuts ./ L))
offset, slope = hcat(ones(length(cuts)), log_chord) \ entropy_arc
c_tilde = 3 * slope
rounded_c_tilde = round(c_tilde; digits=2)

# Evaluate the fit on a dense grid so the dashed curve renders smoothly.
fit_positions = range(first(cuts), last(cuts); length=500)
fit_log_chord = log2.((L / π) .* sinpi.(fit_positions ./ L))
fitted_arc = offset .+ slope .* fit_log_chord

# Plot all measured cuts together with the chord-length fit.
figure = Figure()
axis = Axis(
    figure[1, 1];
    xlabel=L"subsystem size $\ell$",
    ylabel=L"mean entropy $S(\ell)$ [bits]",
)
lines!(
    axis,
    fit_positions,
    fitted_arc;
    label=L"chord-length fit, $\tilde{c} \approx %$rounded_c_tilde$",
    color=:black,
    linestyle=:dash,
)
scatter!(
    axis,
    cuts,
    entropy_arc;
    label=L"\mathrm{trajectory\ average}",
)
axislegend(axis; position=:cb)
figure
```

The points form the expected approximately symmetric entropy arc. Small
left-right differences remain because every cut is sampled independently with
a finite trajectory ensemble. Increase `L`, `depth`, and `trajectories` for a
scaling analysis rather than interpreting this one finite-size fit.

For increasing `L`, `depth`, and trajectory count, `c_tilde` should approach
the percolation effective entanglement central charge

```math
\widetilde c = \frac{3\sqrt{3}\ln 2}{2\pi} \approx 0.573.
```

This is an effective entanglement prefactor, not the ordinary central charge of
the underlying percolation conformal field theory, which is zero. Fit an
ensemble average rather than a single measurement trajectory, and check
several system sizes before drawing a scaling conclusion.
