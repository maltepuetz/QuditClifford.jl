"""
    expect_int!(stabtab::StabilizerTableau, op)

Compute the expectation value ⟨P⟩ for a Pauli operator `op` in exponent form.

# Arguments
- `stabtab::StabilizerTableau`: Stabilizer tableau (canonicalized if needed).
- `op`: Pauli operator as `SinglePauli`, `DoublePauli`, `TriplePauli`, `NPauli`,
  or an `AbstractVector{<:Integer}` of length `2n` or `2n+1`.

# Returns
- `-1` if ⟨P⟩ = 0 (operator not in the stabilizer span).
- For odd prime `d`, returns `k` with ⟨P⟩ = ω^k (k mod `d`).
- For `d=2`, returns `k` with ⟨P⟩ = i^k (k mod 4).
- If `storephase=false` and `op` is in-span, returns `0` by convention.

# Examples
```julia
stab = StabilizerTableau(2, 2; state=:product, basis=:Z)
k = expect_int!(stab, SinglePauli(1, 0, 1))  # Z on qudit 1 -> 0 (⟨Z⟩ = 1)
k2 = expect_int!(stab, SinglePauli(1, 1, 0)) # X on qudit 1 -> -1 (⟨X⟩ = 0)
```

# Notes
- Canonicalizes `stabtab` if needed, which may permute generator columns and sets `iscanonical=true`.
- Uses the preallocated workspaces in `stabtab` to remain allocation-free.
"""
function expect_int!(stabtab::StabilizerTableau, op)
    tab = stabtab.tableau
    n = stabtab.n
    m = stabtab.m
    d = stabtab.d
    kP = op_phase_exponent(stabtab, op)

    # Fill res_workspace with the operator's XZ part (mod d).
    res = stabtab.res_workspace
    set_operator!(res, stabtab, op)

    # make sure the tableau is in canonical form (required for membership test)
    stabtab.iscanonical || canonicalize!(stabtab)

    piv = stabtab.pivcol_of_row

    @assert length(piv) >= 2n
    @assert length(stabtab.res_workspace) == 2n
    @assert length(stabtab.c_workspace) == n
    @assert length(stabtab.zacc_workspace) == n
    @assert length(stabtab.xdotz_cache) == n

    # ------------------------------------------------------------
    # Step 1: Membership test + coefficient readout in canonical basis
    # ------------------------------------------------------------
    cvec = stabtab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end

    # In RCEF, each pivot row r has a known pivot column pc = piv[r].
    # If res[r] != 0, that coefficient must be res[r], and we subtract it off.
    @inbounds for r in 1:2n
        pc = piv[r]
        pc == 0 && continue

        γ = res[r]
        γ == 0 && continue

        cvec[pc] = mod(γ, d)

        # res -= γ * tab[1:2n, pc]
        @inbounds @simd for i in 1:2n
            res[i] = mod(res[i] - mod(γ * tab[i, pc], d), d)
        end
    end

    # If residual is nonzero, operator not in stabilizer span ⇒ expectation 0, return -1.
    @inbounds for i in 1:2n
        if res[i] != 0
            return -1
        end
    end

    # If we don't store phase, we can only say "in span" ⇒ nonzero expectation,
    # but cannot determine the phase. Return 0 by convention (⟨P⟩ = 1).
    stabtab.storephase || return 0

    # ------------------------------------------------------------
    # Step 2: Compute phase exponent of Q = ∏_j g_j^{c[j]} with cross-terms
    # Uses cached xdotz_cache[j] for the binomial term (odd prime d).
    # xdotz_cache[j] == (x_j · z_j) mod d for each generator column j.
    # ------------------------------------------------------------
    prow = 2n + 1
    d_phase = phase_modulus(d)

    zacc = stabtab.zacc_workspace
    @turbo for q in 1:n
        zacc[q] = 0
    end
    kacc = 0

    if d == 2
        # qubits: coefficients are mod 2, and phase is mod 4 as i^k
        @inbounds for j in 1:m
            aj = cvec[j]
            aj == 0 && continue

            # cross = (x_j · zacc) mod 2
            cross = 0
            @turbo for q in 1:n
                cross += tab[q, j] * zacc[q]
            end
            cross = cross & 1  # mod 2

            # kacc += k_j + 2*cross   (mod 4)
            kacc = mod(kacc + tab[prow, j] + 2 * cross, d_phase)

            # zacc += z_j (mod 2)
            @inbounds @simd for q in eachindex(zacc)
                zacc[q] = (zacc[q] + tab[n+q, j]) & 1
            end
        end

        # ⟨P⟩ = i^(kP - kacc)
        δ = mod(mod(kP, d_phase) - kacc, d_phase)
        return δ

    else
        # odd prime d: phase is ω^k mod d
        inv2 = stabtab.inversemod(2, d)
        xdotz_cache = stabtab.xdotz_cache

        @inbounds for j in 1:m
            aj = mod(cvec[j], d)
            aj == 0 && continue

            # phase of g_j^aj: aj*k_j + C(aj,2)*(xj·zj)
            xdotz_j = xdotz_cache[j]  # cached
            kpow = mod(aj * tab[prow, j] + binom2_mod_oddprime(aj, d, inv2) * xdotz_j, d_phase)

            # cross = aj*(xj · zacc) mod d
            cross_base = dot_xz_col_vs_zacc(tab, n, j, zacc, d)
            cross = mod(aj * cross_base, d)

            kacc = mod(kacc + kpow + cross, d_phase)

            # zacc += aj*zj (mod d)
            @inbounds @simd for q in 1:n
                zacc[q] = mod(zacc[q] + mod(aj * tab[n+q, j], d), d)
            end
        end

        # ⟨P⟩ = ω^(kP - kacc)
        δ = mod(mod(kP, d_phase) - kacc, d_phase)
        return δ
    end
end

"""
    expect!(stabtab::StabilizerTableau, op)

Return the expectation value ⟨P⟩ as a `ComplexF64` for a Pauli operator `op`.

# Arguments
- `stabtab::StabilizerTableau`: Stabilizer tableau (canonicalized if needed).
- `op`: Pauli operator as `SinglePauli`, `DoublePauli`, `TriplePauli`, `NPauli`,
  or an `AbstractVector{<:Integer}` of length `2n` or `2n+1`.

# Returns
- `ComplexF64` expectation value. Returns `0.0 + 0.0im` if ⟨P⟩ = 0.

# Examples
```julia
stab = StabilizerTableau(3, 1; state=:product, basis=:Z)
val = expect!(stab, SinglePauli(1, 0, 1))  # ⟨Z⟩ = 1 + 0im
```

# Notes
Delegates to [`expect_int!`](@ref) and converts the exponent to a complex phase.
"""
function expect!(stabtab::StabilizerTableau, op)
    k = expect_int!(stabtab, op)
    return _expect_from_exponent(stabtab, k)
end

@inline function _expect_from_exponent(stabtab::StabilizerTableau, k::Int)
    k < 0 && return 0.0 + 0.0im

    d = stabtab.d
    d_phase = phase_modulus(d)

    if d == 2
        return cis((π / 2) * Float64(mod(k, d_phase)))
    else
        return cis(2π * (Float64(mod(k, d_phase)) / Float64(d)))
    end
end
