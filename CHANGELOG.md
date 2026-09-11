# Changelog

All notable changes to QuditClifford.jl are documented in this file.

## Unreleased

### Added

- Tableau constructors now warn when the qudit dimension is large enough that
  phase and symplectic dot products can overflow `Int` and silently corrupt
  results. The bound depends on both `d` and `n`; see the arithmetic envelope
  in the conventions page.

### Changed

- `InverseMod` strategies now always return an `Int`. Lookup tables may still
  be stored in any integer element type, including `Int32` and `UInt64`, and
  the value is converted when read. Previously an unsigned table silently
  corrupted odd-prime measurements and a narrow signed table threw a
  `MethodError`; both now behave exactly as an `Int` table.
- `PrecomputedInvMod` rejects lookup tables whose element type is not an
  `Integer`. A `Float64` table was previously accepted and worked only for
  exactly-valued entries.
- Internal: modular reduction in the tableau inner loops is now branch-free,
  which speeds up canonicalization and the measurement paths substantially.
  No user-visible behaviour change.

## 0.1.0 - 2026-07-16

- Initial public release.
- Stabilizer and destabilizer tableaux for prime-dimensional qudits.
- Pure and mixed-state initialization, projective Pauli measurements,
  expectation values, canonicalization, purity checks, and entanglement entropy.
- Continuous integration, API documentation, and automated release tagging.
