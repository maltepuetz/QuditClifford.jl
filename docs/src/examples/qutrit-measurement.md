```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# A two-qutrit measurement

We begin with two independent qutrits in the X basis and measure the correlated
operator ``Z_1Z_2``. Each initial X stabilizer fails to commute with the
measurement, so this example exercises the multi-generator update path.

```jldoctest qutrit-measurement
julia> state = DestabilizerTableau(3, 2; state = :product, basis = :X)
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

The operator ``Z_1 Z_2`` is two `(qudit, x, z)` triplets. The outcome of the
measurement is random; `outcome = 2` selects the eigenvalue ``\omega^2`` so
that the example is reproducible:

```jldoctest qutrit-measurement
julia> z1z2 = DoublePauli(1, 0, 1, 2, 0, 1)
Z₁ Z₂

julia> measure!(state, z1z2; outcome = 2)
2
```

[`measure!`](@ref) changes the tableau in place. ``Z_1 Z_2`` is now a
generator, and its phase column records the outcome:

```jldoctest qutrit-measurement
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

The expectation value of ``Z_1 Z_2``, zero before the measurement, is now
``\omega^2``:

```jldoctest qutrit-measurement
julia> expect!(state, z1z2)
-0.5000000000000004 - 0.8660254037844385im
```

The phase entry `1` on the displayed ``Z_1Z_2`` generator means that
``\omega Z_1Z_2`` stabilizes the post-measurement state. Equivalently,
``Z_1Z_2`` has eigenvalue ``\omega^{-1}=\omega^2``, matching the returned
outcome and expectation value.
