```@meta
CurrentModule = QuditClifford
DocTestSetup = :(using QuditClifford)
```

# API Reference

```@docs
QuditClifford
```

## Tableaux

```@docs
AbstractTableau
DestabilizerTableau
StabilizerTableau
reset!
canonicalize!
is_pure
```

## Pauli operators

```@docs
AbstractPauli
FewQuditPauli
SinglePauli
DoublePauli
TriplePauli
NPauli
GeneralPauli
```

## Clifford unitaries

```@docs
AbstractClifford
Fourier
Phase
Multiplier
PauliGate
SUM
CPhase
SWAP
CliffordOperator
inv
∘
apply!
conjugate
```

## Random Cliffords and states

```@docs
random_clifford
random_clifford!
random_state!
```

## Measurements

```@docs
measure!
```

## Expectation values and entropy

```@docs
expect!
expect_int!
entanglement_entropy
```

## Inspecting states

```@docs
StabilizerKet
ket
state_vector
density_matrix
```
