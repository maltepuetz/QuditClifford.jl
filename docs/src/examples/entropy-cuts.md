```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Entropy across every cut

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
