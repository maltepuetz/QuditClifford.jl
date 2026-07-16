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

figure = Figure(size=(480, 160), figure_padding=6)
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
