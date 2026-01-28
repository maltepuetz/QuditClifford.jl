############################################
# Allocation-free expectation value (RCEF) #
# Uses xdotz_cache for odd prime d         #
############################################

# ! ############################################## ! #
# ! ############################################## ! #
# ! ############################################## ! #
# TODO GO THROUGH THIS CAREFULLY AND TEST THOROUGHLY #
# ! ############################################## ! #
# ! ############################################## ! #
# ! ############################################## ! #


"""
    expectation_value!(stabtab::StabilizerTableau, op::Vector{Int})
    expectation_value!(stabtab::StabilizerTableau, op_xz::AbstractVector{Int}, kP::Int)

Allocation-free ⟨P⟩ for a Pauli operator P represented as a vector.

Operator format:
- If length(op) == 2n:    op[1:n]=x, op[n+1:2n]=z, and kP=0
- If length(op) == 2n+1:  op[1:n]=x, op[n+1:2n]=z, op[2n+1]=kP

Assumes `canonicalize!(stabtab)` was called so that:
- stabtab.tableau is in column-RCEF on rows 1:2n
- stabtab.pivcol_of_row[r] is the pivot column index for pivot row r (or 0)
- stabtab.xdotz_cache[j] == (x_j · z_j) mod d for each generator column j

Phase convention:
- odd prime d: phase exponent k is mod d and corresponds to ω^k
- d=2: phase exponent k is mod 4 and corresponds to i^k

Uses preallocated fields:
- res_workspace (length 2n)
- c_workspace   (length n)
- zacc_workspace (length n)

Returns ComplexF64.
"""
function expectation_value!(stabtab::StabilizerTableau, op::Vector{Int})
    n = stabtab.n
    @assert length(op) == 2n || length(op) == 2n + 1
    kP = (length(op) == 2n + 1) ? op[2n+1] : 0
    return expectation_value!(stabtab, view(op, 1:2n), kP)
end
function expectation_value!(stabtab::StabilizerTableau, op_xz, kP::Int)
    tab = stabtab.tableau
    n = stabtab.n
    d = stabtab.d
    piv = stabtab.pivcol_of_row

    @assert length(op_xz) == 2n
    @assert length(piv) >= 2n
    @assert length(stabtab.res_workspace) == 2n
    @assert length(stabtab.c_workspace) == n
    @assert length(stabtab.zacc_workspace) == n
    @assert length(stabtab.xdotz_cache) == n

    # ------------------------------------------------------------
    # Step 1: Membership test + coefficient readout in canonical basis
    # ------------------------------------------------------------
    res = stabtab.res_workspace
    @inbounds @simd for i in eachindex(res)
        res[i] = mod(op_xz[i], d)
    end

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

    # If residual is nonzero, operator not in stabilizer span ⇒ expectation 0.
    @inbounds for i in 1:2n
        if res[i] != 0
            return 0.0 + 0.0im
        end
    end
    
    # If we don't store phase, we can only say "in span" ⇒ nonzero expectation,
    # but cannot determine the phase. Return 1 by convention.
    stabtab.storephase || return 1.0 + 0.0im

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
        @inbounds for j in eachindex(cvec)
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
        return cis((π / 2) * Float64(δ))

    else
        # odd prime d: phase is ω^k mod d
        inv2 = stabtab.inversemod(2, d)
        xdotz_cache = stabtab.xdotz_cache

        @inbounds for j in eachindex(cvec)
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
        return cis(2π * (Float64(δ) / Float64(d)))
    end
end
