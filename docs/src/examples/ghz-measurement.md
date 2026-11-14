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
# Construct a four-qubit GHZ state.
# d = 2 selects qubits; n = 4 creates four sites; :ghz selects the GHZ preset.
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
# outcome = 0 selects the +1 eigenvalue and makes the walkthrough reproducible.
julia> measure!(state, x1; outcome = 0)
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
