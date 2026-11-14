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
Base.copy(::StabilizerTableau)
Base.copy!(::StabilizerTableau, ::StabilizerTableau)
Base.:(==)(::AbstractTableau, ::AbstractTableau)
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

## Measurements and channels

```@docs
measure!
expect!
expect_int!
dephase!
traceout!
```

## Entanglement measures

```@docs
entanglement_entropy
entanglement_profile
entanglement_profile!
entanglement_negativity
mutual_information
tripartite_information
conditional_mutual_information
```

## Inspecting states

```@docs
StabilizerKet
ket
state_vector
density_matrix
```
