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
for qubits, trits for qutrits. These sizes are too small for the paper's fitted
coefficients; [At the paper's sizes](@ref) compares with them. The plot shows qubits on the left and
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
the qutrit transition lies at a larger `p`, which the next section confirms.
The mutual information, which also counts classical correlations, exceeds the
negativity wherever either is nonzero, and it shows the same three regimes.

The paper's central result is that the negativity grows as ``L^{1/3}`` for
``0 < p < p_c``. At these sizes such a power law cannot be told apart from a
logarithm.

## At the paper's sizes

The paper fits the power law over `L = 40` to `280` with 200 realizations.
Running that during every documentation build would take too long, so the
averages below were computed once by
[`docs/data/boundary-dephasing/reproduce.jl`](https://github.com/maltepuetz/QuditClifford.jl/blob/main/docs/data/boundary-dephasing/reproduce.jl)
and stored next to it as JLD2 files; the build only reads and plots them. The script runs
the same circuit as above, threaded over trajectories, for
`L = 40, 80, …, 280`, 200 trajectories per point and depth `4L`, with `p` in
steps of `0.025` up to `0.3` for qubits and `0.5` for qutrits. One dimension
up to `p = 0.3` takes about 17 minutes on 12 threads of an Apple M3 Pro:

```bash
julia --project=docs -t auto docs/data/boundary-dephasing/reproduce.jl 2
julia --project=docs -t auto docs/data/boundary-dephasing/reproduce.jl 3 0.5
```

```@example boundary_dephasing
using JLD2

# Read the stored averages into stored[d][(L, p)] = (E, ΔE, I, ΔI).
function read_stored(d)
    path = joinpath(pkgdir(QuditClifford), "docs", "data", "boundary-dephasing",
        "results_d$(d).jld2")
    data = JLD2.load(path)
    return Dict(
        (L, p) => (data["E"][i, j], data["dE"][i, j], data["I"][i, j], data["dI"][i, j])
        for (i, p) in enumerate(data["p"]) for (j, L) in enumerate(data["L"])
    )
end

stored = Dict(d => read_stored(d) for d in dimensions)
paper_lengths = 40:40:280
# Stop short of the colormap's pale end so the largest L stays visible on white.
length_colors = [cgrad(:batlow)[x] for x in range(0, 0.8; length=length(paper_lengths))]

figure = Figure(size=(720, 420), figure_padding=9)
panels = [
    Axis(
        figure[row, column];
        title=row == 1 ? titles[column] : "",
        xlabel=row == 2 ? L"measurement probability $p$" : "",
        ylabel=column == 1 ? (
            row == 1 ? L"negativity $\langle E\rangle$ [$\log d$]" :
            L"mutual information $\langle I\rangle$ [$\log d$]"
        ) : "",
    ) for row in 1:2, column in 1:2
]

for (column, d) in enumerate(dimensions), (index, L) in enumerate(paper_lengths)
    ps = sort([p for (ℓ, p) in keys(stored[d]) if ℓ == L])
    averages = [stored[d][(L, p)] for p in ps]
    for (row, (value, error)) in enumerate(((1, 2), (3, 4)))
        means = getindex.(averages, value)
        color = length_colors[index]
        lines!(panels[row, column], ps, means; color=color, label=L"L = %$L")
        errorbars!(panels[row, column], ps, means, getindex.(averages, error);
            color=color, whiskerwidth=4)
    end
end

Legend(figure[1:2, 3], panels[1, 1]; merge=true)
figure
```

For qubits the negativity is zero at `p = 0`, grows with `L` up to
``p \approx 0.15``, and the curves for all sizes meet between `p = 0.175`
and `0.2`, past which the negativity no longer depends on `L`: the area law.
That matches the paper's ``p_c \approx 0.16`` within the resolution of this
grid; locating it precisely needs a finite-size scaling collapse. The qutrit
curves keep growing with `L` up to `p = 0.3` and meet between `p = 0.325` and
`0.35`, so the qutrit transition sits at roughly twice the qubit measurement
rate. The mutual information follows the negativity at about twice its value.

At `p = 0.1` the paper fits ``E = c_1 L^{1/3} + c_2``. A weighted
least-squares fit of the stored averages gives:

```@example boundary_dephasing
# Fit y = c1 L^(1/3) + c2 with weights 1/σ²; return the coefficients and
# their standard errors.
function fit_cube_root(L, y, σ)
    X = hcat(L .^ (1 / 3), ones(length(L)))
    W = 1 ./ σ .^ 2
    A = X' * (W .* X)
    c = A \ (X' * (W .* y))
    covariance = inv(A)
    return c, sqrt.([covariance[1, 1], covariance[2, 2]])
end

fits = Dict(
    d => fit_cube_root(
        collect(paper_lengths),
        [stored[d][(L, 0.1)][1] for L in paper_lengths],
        [stored[d][(L, 0.1)][2] for L in paper_lengths],
    ) for d in dimensions
)

for d in dimensions
    (c1, c2), (σ1, σ2) = fits[d]
    println("d = $d: c1 = ", round(c1; digits=3), " ± ", round(σ1; digits=3),
        ", c2 = ", round(c2; digits=3), " ± ", round(σ2; digits=3))
end
```

```@example boundary_dephasing
figure = Figure(size=(720, 260), figure_padding=9)
for (column, d) in enumerate(dimensions)
    axis = Axis(
        figure[1, column];
        title=titles[column],
        xlabel=L"L^{1/3}",
        ylabel=column == 1 ? L"negativity $\langle E\rangle$ [$\log d$]" : "",
    )
    x = collect(paper_lengths) .^ (1 / 3)
    averages = [stored[d][(L, 0.1)] for L in paper_lengths]
    means, errors = getindex.(averages, 1), getindex.(averages, 2)
    (c1, c2), _ = fits[d]
    line = range(minimum(x), maximum(x); length=50)
    lines!(axis, line, c1 .* line .+ c2; color=qudit_colors[2], linestyle=:dash)
    errorbars!(axis, x, means, errors; color=qudit_colors[1], whiskerwidth=6)
    scatter!(axis, x, means; color=qudit_colors[1])
end
figure
```

For qubits the slope is ``c_1 = 0.72 \pm 0.03`` against the paper's `0.779`,
and the offset ``c_2 = -0.97 \pm 0.16`` against its `-1.307`. The paper does not
state its logarithm base; in natural logarithms the slope here would be
`0.50`, so the agreement in bits suggests base 2. The remaining difference of
about two standard errors may come from the depth, which the paper leaves at
``O(L)`` while this run fixes it at `4L`. Qutrits follow the same law with
``c_1 = 0.69 \pm 0.03`` trits.
