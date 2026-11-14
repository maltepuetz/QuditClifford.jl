```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# Following entanglement through a measurement

A four-qubit GHZ state has one bit of entropy across every nontrivial cut. We
measure X on the first qubit and compare the entropy before and after. The local
measurement separates qubit 1 while leaving the remaining GHZ cluster
entangled.

```jldoctest ghz-measurement
julia> state = DestabilizerTableau(2, 4; state = :ghz)
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

Before the measurement, the cut between qubits `{1, 2}` and `{3, 4}` carries
one bit:

```jldoctest ghz-measurement
julia> entanglement_entropy(state, [1, 2])
1
```

``X_1`` does not commute with the ``Z_1 Z_2`` generator, so its outcome is
random and the measurement changes the state. `outcome = 0` selects the
eigenvalue `+1` so that the example is reproducible:

```jldoctest ghz-measurement
julia> x1 = SinglePauli(1, 1, 0)
X₁

julia> measure!(state, x1; outcome = 0)
0

julia> expect!(state, x1)
1.0 + 0.0im
```

Qubit 1 is now a product factor, while qubits 2 to 4 remain in a GHZ state:

```jldoctest ghz-measurement
julia> entanglement_entropy(state, [1]), entanglement_entropy(state, [2, 3])
(0, 1)
```

Entropy is an integer in log-`d` units. For qubits it is measured in bits; for
qutrits, one unit means ``\log(3)``.
