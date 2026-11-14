```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Random Sampling

Three operations draw random numbers: [`random_clifford`](@ref) and
[`random_clifford!`](@ref) sample Clifford unitaries, [`random_state!`](@ref)
samples stabilizer states, and [`measure!`](@ref) samples outcomes. Each accepts
an RNG as its first argument and otherwise uses the global RNG.

## Random Cliffords

[`random_clifford`](@ref) draws a Clifford uniformly from the Clifford group,
modulo global phase, on an ordered support: every symplectic matrix `F` is
equally likely, and so is each of its valid phase vectors. The result is a
stored [`CliffordOperator`](@ref), so it applies, conjugates, inverts and
composes like any other.

```jldoctest random
julia> using Random

julia> rng = Xoshiro(2026);

julia> U = random_clifford(rng, 3, 1:2);

julia> U.d, U.targets, size(U.F)
(3, [1, 2], (4, 4))

julia> E = CliffordOperator(3, 1:2, [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1], zeros(Int, 4));

julia> inv(U) ∘ U == E
true
```

Every coordinate comes from `rand(rng, 0:(d - 1))`, so the distribution is
exact for an ideal RNG. A seed reproduces a sample for the same RNG type, Julia
version and package version, but the draw stream is not a cross-version
guarantee, which is why these examples print only properties that every sample
shares.

[`random_clifford!`](@ref) refills an existing operator in place, on the same
dimension and support, and allocates nothing once warm. The new action never
depends on the old one, and this is the one package operation that changes an
operator in place. A refill changes `==` and `hash`, so an operator used as a
dictionary key should not be refilled. It borrows the same scratch as `apply!`,
so the no-concurrent-use rule above covers refilling too; and if the RNG
throws partway through, `U` is left unusable until a later refill succeeds,
with nothing rolled back.

## Random states

[`random_state!`](@ref) replaces a tableau's state with a uniformly random
stabilizer state that has `m` independent generators. The default, `m = n`,
gives a pure state:

```jldoctest random
julia> tab = DestabilizerTableau(2, 4);

julia> random_state!(rng, tab) === tab
true

julia> tab.m, is_pure(tab; verify = true)
(4, true)
```

A smaller `m` gives the normalized projector onto a uniformly random stabilizer
code: the stabilizer group has ``d^m`` elements and the density matrix has rank
``d^{n-m}``. This is not a random mixture of stabilizer states. `m = 0` is the
maximally mixed state.

```jldoctest random
julia> random_state!(rng, tab; m = 2);

julia> tab.m, is_pure(tab)
(2, false)
```

Each call with `m > 0` allocates a `2n × 2n` scratch matrix and costs
``O(n^2 m)``. Every draw happens before the tableau changes, so an exception from
the RNG leaves the tableau as it was. Without stored phases
(`storephase = false`) the state is uniform over isotropic subspaces, and no
phases are drawn. A [`DestabilizerTableau`](@ref) receives its dual basis
directly.

## Reproducible trajectories

Without `outcome`, a random branch of [`measure!`](@ref) draws its outcome
uniformly from `0:(d - 1)`, by default from the global RNG. Pass an RNG first,
as in `measure!(rng, tab, op)`, to draw from it instead: one seeded RNG shared
by every measurement and every [`random_clifford!`](@ref) call reproduces a
whole monitored-circuit trajectory. Only the random branches draw, so a
deterministic measurement leaves the RNG untouched.

### A monitored random circuit

Random two-qudit Cliffords interleaved with measurements are the standard model
of a monitored circuit. Keep one operator per bond, refill it for every layer,
and pass the same RNG to every measurement: `measure!` otherwise draws its
outcome from the global RNG, and one seed would no longer reproduce the whole
trajectory.

```jldoctest random
julia> function monitored!(rng, tab; layers = 8, p = 0.25)
           n, d = tab.n, tab.d
           bonds = [random_clifford(rng, d, [q, q + 1]) for q in 1:(n - 1)]
           for layer in 1:layers
               for q in (isodd(layer) ? 1 : 2):2:(n - 1)
                   apply!(tab, random_clifford!(rng, bonds[q]))
               end
               for q in 1:n
                   rand(rng) < p || continue
                   measure!(rng, tab, SinglePauli(q, 0, 1))
               end
           end
           return tab
       end;

julia> tab = monitored!(Xoshiro(7), StabilizerTableau(2, 8; state = :product));

julia> is_pure(tab; verify = true)
true
```

Projective measurement keeps a pure state pure, so the final state is pure
whatever the sample.
