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
