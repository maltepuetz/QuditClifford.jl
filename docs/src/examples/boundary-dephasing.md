```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Measurements against boundary dephasing

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
and the mutual information between the left and right halves of the chain,
the two quantities the paper reports for this geometry. Phases do not affect either quantity, so the tableau stores none. The gates are
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
function boundary_dephasing_averages(rng, d, L; p, depth=4L, trajectories=200)
    # These draws only allocate the per-bond operators; every gate is refilled
    # before it is applied.
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
curves change little beyond `L = 16`: from `L = 16` to `L = 32` the negativity
gains about a tenth of a bit, against about seven tenths at `p = 0.1`. That is
consistent with the area law the paper finds above ``p_c``, although at these
sizes slow growth cannot be told apart from saturation. The qutrit curves at
`p = 0.3` rise more slowly beyond `L = 24`, which suggests that
the qutrit transition lies at a larger `p`; this example does not locate it.
The mutual information, which also counts classical correlations, exceeds the
negativity wherever either is nonzero, and it shows the same three regimes.

The paper's central result is that the negativity grows as ``L^{1/3}`` for
``0 < p < p_c``. At these sizes such a power law cannot be told apart from a
logarithm. The paper fits it over `L = 40` to `280` with 200 realizations;
raise `chain_lengths` toward those sizes for a scaling analysis rather than
reading an exponent off this figure.
