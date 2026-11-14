```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# A deep measurement-only Ising circuit

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
# is the number of alternating measurement layers. rng draws which operators
# are measured and the outcomes of the random measurements.
function measurement_only_ising(rng, L;
    p=0.5,
    depth=8L,
)
    # Begin with |+⟩^L, an unentangled X-basis product state.
    state = DestabilizerTableau(2, L; state = :X)

    for _ in 1:depth
        # First measure ZZ† on every periodic edge.
        for i in 1:L
            if rand(rng) < 1 - p
                # mod1 closes the edge from site L back to site 1.
                j = mod1(i + 1, L)
                zzdagger = DoublePauli(i, 0, 1, j, 0, 1)
                measure!(rng, state, zzdagger)
            end
        end

        # Then measure X on every site.
        for i in 1:L
            if rand(rng) < p
                xi = SinglePauli(i, 1, 0)
                measure!(rng, state, xi)
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
shared by typical trajectories. [`entanglement_profile!`](@ref) returns the
entropy of every prefix `1:l` from one elimination, so a whole arc costs about
as much as a single entropy. Qubit entropy is measured in bits.

```@example ising_circuit
# Estimate the mean entropy as a function of subsystem size.
# trajectories controls sampling accuracy; seed makes the ensemble reproducible.
function mean_entropy_arc(L;
    p=0.5,
    depth=8L,
    trajectories=500,
    seed=1234,
)
    rng = Xoshiro(seed)

    # Include every nontrivial prefix cut, from one site through L - 1 sites.
    cuts = collect(1:(L - 1))
    entropies = zeros(length(cuts), trajectories)

    # profile[l + 1] is the entropy of sites 1:l, for l = 0, …, L.
    profile = Vector{Int}(undef, L + 1)

    # Accumulate the entropy arc over measurement trajectories.
    for t in 1:trajectories
        state = measurement_only_ising(rng, L; p=p, depth=depth)
        entanglement_profile!(profile, state)
        entropies[:, t] .= @view profile[2:L]
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

# Fit all cut positions.
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
left-right differences remain because the trajectory ensemble is finite. Increase `L`, `depth`, and `trajectories` for a
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
