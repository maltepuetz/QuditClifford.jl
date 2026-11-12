```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# A two-qutrit measurement

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
