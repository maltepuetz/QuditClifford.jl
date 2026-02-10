"""
    expect_int!(tab::AbstractTableau, op)

Compute the expectation value ⟨P⟩ for a Pauli operator `op` in exponent form.

# Arguments
- `tab::AbstractTableau`: Stabilizer or destabilizer tableau.
- `op`: Pauli operator as `SinglePauli`, `DoublePauli`, `TriplePauli`, `NPauli`,
  `GeneralPauli`, or an `AbstractVector{<:Integer}` of length `2n` or `2n+1`.

# Returns
- `-1` if ⟨P⟩ = 0 (operator not in the stabilizer span).
- For odd prime `d`, returns `k` with ⟨P⟩ = ω^k (k mod `d`).
- For `d=2`, returns `k` with ⟨P⟩ = i^k (k mod 4).
- If `storephase=false` and `op` is in-span, returns `0` by convention.

# Notes
Uses shared span decomposition helpers:
1. `coeffs_from_generators!`
2. `residual_from_coeffs!`
3. `phase_exponent_from_coeffs!`
"""
function expect_int!(tab::AbstractTableau, op::AbstractPauli)
    d = tab.d
    kP = op.phase

    in_span_and_coeffs!(tab, op) || return -1
    tab.storephase || return 0

    kacc = phase_exponent_from_coeffs!(tab)
    d_phase = phase_modulus(d)
    return mod(mod(kP, d_phase) - kacc, d_phase)
end

function expect_int!(tab::AbstractTableau, op::AbstractVector{<:Integer})
    gop = GeneralPauli(tab.n, tab.d, op)
    return expect_int!(tab, gop)
end

"""
    expect!(tab::AbstractTableau, op)

Return the expectation value ⟨P⟩ as a `ComplexF64` for a Pauli operator `op`.

# Arguments
- `tab::AbstractTableau`: Tableau (canonicalized if needed for stabilizers).
- `op`: Pauli operator as `SinglePauli`, `DoublePauli`, `TriplePauli`, `NPauli`,
  or an `AbstractVector{<:Integer}` of length `2n` or `2n+1`.

# Returns
- `ComplexF64` expectation value. Returns `0.0 + 0.0im` if ⟨P⟩ = 0.

# Examples
```julia
tab = StabilizerTableau(3, 1; state=:product, basis=:Z)
val = expect!(tab, SinglePauli(1, 0, 1))  # ⟨Z⟩ = 1 + 0im
```

# Notes
Delegates to [`expect_int!`](@ref) and converts the exponent to a complex phase.
"""
function expect!(tab::AbstractTableau, op::AbstractPauli)
    k = expect_int!(tab, op)
    return _expect_from_exponent(tab, k)
end

function expect!(tab::AbstractTableau, op::AbstractVector{<:Integer})
    gop = GeneralPauli(tab.n, tab.d, op)
    return expect!(tab, gop)
end

@inline function _expect_from_exponent(tab::AbstractTableau, k::Int)
    k < 0 && return 0.0 + 0.0im

    d = tab.d
    d_phase = phase_modulus(d)

    if d == 2
        return cis((π / 2) * Float64(mod(k, d_phase)))
    else
        return cis(2π * (Float64(mod(k, d_phase)) / Float64(d)))
    end
end
