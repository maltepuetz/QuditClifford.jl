```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Purification by random two-qudit measurements

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
The following helpers generate the measurements and run one trajectory. Each
takes the random number generator `rng` as an argument, which draws both the
Paulis and the measurement outcomes, so one seed reproduces every trajectory.

```@example purification
using QuditClifford
using Random
using Statistics

# Draw a nonidentity Pauli XˣZᶻ on one site.
function random_local_pauli(rng, d)
    x = rand(rng, 0:(d - 1))
    z = rand(rng, 0:(d - 1))

    while iszero(x) && iszero(z)
        x = rand(rng, 0:(d - 1))
        z = rand(rng, 0:(d - 1))
    end

    return x, z
end

# Measure one alternating nearest-neighbour matching of the periodic chain.
function random_brickwork_layer!(rng, tab, d, n, layer)
    first_site = isodd(layer) ? 1 : 2

    for offset in 0:(n ÷ 2 - 1)
        # mod1 closes the final even-layer bond from site n back to site 1.
        i = mod1(first_site + 2offset, n)
        j = mod1(i + 1, n)

        x_i, z_i = random_local_pauli(rng, d)
        x_j, z_j = random_local_pauli(rng, d)
        operator = DoublePauli(i, x_i, z_i, j, x_j, z_j)

        measure!(rng, tab, operator)
    end

    return nothing
end


# Follow one maximally mixed state until its tableau reaches full rank.
function purification_trajectory(
    rng,
    d,
    n;
    max_layers=10000,
)
    iseven(n) || throw(ArgumentError("n must be even"))

    # Phases are unnecessary because only the rank of the tableau is tracked.
    tab = DestabilizerTableau(d, n; state = :mixed, storephase = false)
    @assert tab.m == 0

    # Pre-filling with n extends a completed trajectory along its pure plateau.
    rank_history = fill(n, max_layers + 1)
    rank_history[1] = tab.m

    previous_rank = tab.m
    for layer in 1:max_layers
        random_brickwork_layer!(rng, tab, d, n, layer)

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
    rng,
    d,
    n;
    max_layers=10000,
    trajectories=1000,
)
    rank_histories = Matrix{Int}(undef, max_layers + 1, trajectories)
    purification_layers = Vector{Int}(undef, trajectories)

    for trajectory in 1:trajectories
        rank_history, purification_layers[trajectory] = purification_trajectory(
            rng,
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

rng = Xoshiro(1234)
results = [
    purification_ensemble(
        rng,
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
