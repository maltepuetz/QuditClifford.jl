###########################################
# Commutation / symplectic inner products #
###########################################

@inline function commutation(s::AbstractVector{<:Integer}, v::AbstractVector{<:Integer})
    N = length(s) ÷ 2
    comm = 0

    @turbo for i in 1:N
        comm += s[i] * v[i+N]
    end
    @turbo for i in 1:N
        comm -= s[i+N] * v[i]
    end
    comm
end

@inline function commutation_col(A::AbstractMatrix{<:Integer}, col::Int, s::AbstractVector{<:Integer})
    N = length(s) ÷ 2
    comm = 0

    @turbo for i in 1:N
        comm += A[i, col] * s[i+N]
    end
    @turbo for i in 1:N
        comm -= A[i+N, col] * s[i]
    end
    comm
end
@inline function commutation_col(A::AbstractMatrix{<:Integer}, col::Int, op::SinglePauli)
    N = size(A, 1) ÷ 2
    comm = A[op.qudit, col] * op.z - A[op.qudit+N, col] * op.x
    comm
end
@inline function commutation_col(A::AbstractMatrix{<:Integer}, col::Int, op::DoublePauli)
    N = size(A, 1) ÷ 2
    comm = 0
    comm += A[op.qudit1, col] * op.z1 - A[op.qudit1+N, col] * op.x1
    comm += A[op.qudit2, col] * op.z2 - A[op.qudit2+N, col] * op.x2
    comm
end
@inline function commutation_col(A::AbstractMatrix{<:Integer}, col::Int, op::TriplePauli)
    N = size(A, 1) ÷ 2
    comm = 0
    comm += A[op.qudit1, col] * op.z1 - A[op.qudit1+N, col] * op.x1
    comm += A[op.qudit2, col] * op.z2 - A[op.qudit2+N, col] * op.x2
    comm += A[op.qudit3, col] * op.z3 - A[op.qudit3+N, col] * op.x3
    comm
end
@inline function commutation_col(A::AbstractMatrix{<:Integer}, col::Int, op::NPauli{D}) where D
    N = size(A, 1) ÷ 2
    comm = 0
    @turbo for i in 1:D
        comm += A[op.qudits[i], col] * op.zs[i] - A[op.qudits[i]+N, col] * op.xs[i]
    end
    comm
end

@inline function commutation_colcol(A::AbstractMatrix{<:Integer}, col1::Int, col2::Int)
    N = size(A, 1) ÷ 2
    comm = 0

    @turbo for i in 1:N
        comm += A[i, col1] * A[i+N, col2]
    end
    @turbo for i in 1:N
        comm -= A[i+N, col1] * A[i, col2]
    end
    comm
end


#########################################################
# Helpers: extract XZ / phase from op (no allocations)  #
#########################################################

"""
Fill `dst[1:2n]` with the XZ-vector of `op` reduced mod d.

Supported op types:
- SinglePauli, DoublePauli, TriplePauli, NPauli
- AbstractVector{<:Integer} of length 2n or 2n+1  (phase ignored)

Returns nothing. Does not allocate.
"""
@inline function fill_xz!(dst::AbstractVector{Int64}, stabtab::StabilizerTableau, op)
    n = stabtab.n
    d = stabtab.d
    @assert length(dst) >= 2n

    if op isa AbstractVector{<:Integer}
        @assert length(op) == 2n || length(op) == 2n + 1
        @inbounds @simd for i in 1:2n
            dst[i] = mod(op[i], d)
        end
        return nothing
    end

    @inbounds @simd for i in 1:2n
        dst[i] = 0
    end

    if op isa SinglePauli
        dst[op.qudit] = mod(op.x, d)
        dst[op.qudit+n] = mod(op.z, d)
    elseif op isa DoublePauli
        dst[op.qudit1] = mod(op.x1, d)
        dst[op.qudit1+n] = mod(op.z1, d)
        dst[op.qudit2] = mod(op.x2, d)
        dst[op.qudit2+n] = mod(op.z2, d)
    elseif op isa TriplePauli
        dst[op.qudit1] = mod(op.x1, d)
        dst[op.qudit1+n] = mod(op.z1, d)
        dst[op.qudit2] = mod(op.x2, d)
        dst[op.qudit2+n] = mod(op.z2, d)
        dst[op.qudit3] = mod(op.x3, d)
        dst[op.qudit3+n] = mod(op.z3, d)
    else
        # NPauli{D}
        @turbo for i in 1:length(op.qudits)
            q = op.qudits[i]
            dst[q] = mod(op.xs[i], d)
            dst[q+n] = mod(op.zs[i], d)
        end
    end
    return nothing
end

"""
Return the phase exponent kP of `op` reduced mod phase_modulus(d), if it exists.
If phase is not available (e.g. vector of length 2n), returns 0.

This is only used when `stabtab.storephase == true`.
"""
@inline function op_phase_exponent(stabtab::StabilizerTableau, op)::Int64
    stabtab.storephase || return 0
    d_phase = phase_modulus(stabtab.d)
    n = stabtab.n

    if op isa AbstractVector{<:Integer}
        return (length(op) == 2n + 1) ? mod(op[2n+1], d_phase) : 0
    end

    # Pauli structs: assume they have a `phase` field (your code does)
    return mod(op.phase, d_phase)
end


#########################################################
# Column update primitive used in noncommuting branch   #
#########################################################

"""
Multiply a tableau column by (generator_workspace)^a on the RIGHT, with correct phase updates.

This is used in the noncommuting measurement update:
    g_i <- g_i * (g_pivot)^a
where g_pivot is stored in generator_workspace.

- XZ rows are updated linearly mod d.
- If storephase=false: no phase bookkeeping.
- If storephase=true:
    odd prime d:
        (ω^k X^x Z^z)(ω^k' X^x' Z^z') = ω^(k+k' + x'·z) X^(x+x') Z^(z+z')
    d=2:
        use phase mod 4 as i^k and cross-term contributes 2*(x'·z) mod 4.
"""
function mul_col_by_workspace_power!(
    stabtab::StabilizerTableau,
    col::Int,
    a::Int,
)
    tab = stabtab.tableau
    genws = stabtab.generator_workspace
    n = stabtab.n
    d = stabtab.d

    a == 0 && return nothing

    nrows_block = 2n

    # If we store phase, compute cross term using the OLD target Z (before XZ update)
    if stabtab.storephase
        phase_row = 2n + 1
        d_phase = phase_modulus(d)

        if d == 2
            # a ∈ {0,1} and a != 0, so a == 1
            # cross = x_ws · z_col_old   (mod 2)
            cross = dot_xz_ws_vs_col(genws, tab, n, col, d)  # uses old tab Z entries
            tab[phase_row, col] = mod(tab[phase_row, col] + genws[phase_row] + 2 * cross, d_phase)
        else
            inv2 = stabtab.inversemod(2, d)

            # Phase of (ws)^a:
            # k_power = a*k_ws + C(a,2)*(x_ws · z_ws)  (mod d)
            xdotz_ws = dot_xz_ws(genws, n, d)
            k_power = mod(a * genws[phase_row] + binom2_mod_oddprime(a, d, inv2) * xdotz_ws, d_phase)

            # cross = a*(x_ws · z_col_old) (mod d)
            cross_base = dot_xz_ws_vs_col(genws, tab, n, col, d)  # x_ws · z_col_old
            cross = mod(a * cross_base, d)

            tab[phase_row, col] = mod(tab[phase_row, col] + k_power + cross, d_phase)
        end
    end

    # Now update XZ: col <- col + a*ws  (mod d)
    @inbounds @simd for i in 1:nrows_block
        tab[i, col] = mod(tab[i, col] + mod(a * genws[i], d), d)
    end

    return nothing
end


#############################################################
# Mixed-state membership test (in span) + phase reconstruction
#############################################################

# internal: allocation-free membership test and coefficient readout in canonical basis.
# returns true iff op ∈ span(S), and fills c_workspace[1:m] with coefficients.
@inline function in_span_and_coeffs!(stabtab::StabilizerTableau, op)
    tab = stabtab.tableau
    n = stabtab.n
    m = stabtab.m
    d = stabtab.d

    stabtab.iscanonical || canonicalize!(stabtab)
    piv = stabtab.pivcol_of_row

    res = stabtab.res_workspace
    set_operator!(res, stabtab, op) # TODO: write this method for all op types

    cvec = stabtab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end

    @inbounds for r in 1:2n
        pc = piv[r]
        pc == 0 && continue
        pc > m && continue

        γ = res[r]
        γ == 0 && continue

        cvec[pc] = mod(γ, d)

        @inbounds @simd for i in 1:2n
            res[i] = mod(res[i] - mod(γ * tab[i, pc], d), d)
        end
    end

    @inbounds for i in 1:2n
        res[i] != 0 && return false
    end
    return true
end

# internal: compute phase exponent of Q = ∏ g_j^{c[j]} (same logic as expectation_value!)
# requires canonicalized xdotz_cache to be valid.
@inline function phase_exponent_from_coeffs!(stabtab::StabilizerTableau)
    tab = stabtab.tableau
    n = stabtab.n
    m = stabtab.m
    d = stabtab.d
    stabtab.storephase || return 0

    prow = 2n + 1
    d_phase = phase_modulus(d)

    cvec = stabtab.c_workspace
    zacc = stabtab.zacc_workspace
    @turbo for q in 1:n
        zacc[q] = 0
    end
    kacc = 0

    if d == 2
        @inbounds for j in 1:m
            aj = cvec[j]
            aj == 0 && continue

            cross = 0
            @turbo for q in 1:n
                cross += tab[q, j] * zacc[q]
            end
            cross = cross & 1

            kacc = mod(kacc + tab[prow, j] + 2 * cross, d_phase)

            @inbounds @simd for q in 1:n
                zacc[q] = (zacc[q] + tab[n+q, j]) & 1
            end
        end
        return kacc
    else
        inv2 = stabtab.inversemod(2, d)
        xdotz_cache = stabtab.xdotz_cache

        @inbounds for j in 1:m
            aj = mod(cvec[j], d)
            aj == 0 && continue

            xdotz_j = xdotz_cache[j]
            kpow = mod(aj * tab[prow, j] + binom2_mod_oddprime(aj, d, inv2) * xdotz_j, d_phase)

            cross_base = dot_xz_col_vs_zacc(tab, n, j, zacc, d)
            cross = mod(aj * cross_base, d)

            kacc = mod(kacc + kpow + cross, d_phase)

            @inbounds @simd for q in 1:n
                zacc[q] = mod(zacc[q] + mod(aj * tab[n+q, j], d), d)
            end
        end
        return kacc
    end
end


############################################
# Projective measurement (mixed-state)     #
############################################

"""
Projective measurement of a Pauli operator.

Mixed-state stabilizer semantics (stabilizer density operators / codespace projectors):

### case 1: operator does NOT commute with all stabilizer generators
- the measurement outcome is uniformly random
- we replace the first generator that does not commute with the operator by the measured operator
  with a phase chosen so that the new generator stabilizes the post-measurement state
- we multiply all other generators that do not commute with the operator by powers of the
  replaced generator to restore commutation

### case 2: operator commutes with all stabilizer generators
- if the operator is in the stabilizer span: outcome is deterministic, state unchanged
- otherwise: outcome is uniformly random and we append the measured operator as a new generator
  (again with a phase chosen to stabilize the post-measurement state)

Returns the outcome as an integer:
- odd prime d: t ∈ 0:(d-1) meaning eigenvalue ω^t of the Pauli observable
- d=2: b ∈ {0,1,2,3} meaning eigenvalue (i)^b of the Pauli observable
"""
function measure!(stabtab::StabilizerTableau, op)
    d = stabtab.d
    n = stabtab.n
    m = stabtab.m

    # sample a random outcome (only used if non-deterministic)
    outcome = rand(0:d-1)
    
    # get phase of the operator (if available)
    kop = op_phase_exponent(stabtab, op)

    # phase exponent of the generator we should store so that it stabilizes the post-measurement state
    # If P_xz |ψ'> = ω^outcome |ψ'> then (ω^{-outcome} P_xz) |ψ'> = |ψ'>.
    # Thus if ω^{kop} P_xz |ψ'> = ω^outcome |ψ'> then (ω^{-outcome + kop} P_xz) |ψ'> = |ψ'>.
    # For d=2: (-1)^b = i^(2b), so the required stabilizer phase exponent is 2*b (mod 4).
    kgen = 0
    if stabtab.storephase
        if d == 2
            kgen = mod(2 * outcome + kop, phase_modulus(d))
        else
            kgen = mod(-outcome + kop, d)  # phase modulus is d for odd primes
        end
    end

    comm0 = 0  # the commutation with the first non-commuting generator
    for i in 1:m

        # get the commutator of generator i with the operator
        commutator = mod(commutation_col(stabtab.tableau, i, op), d)

        ### case a: if they commute we continue
        commutator == 0 && continue

        ### case b:
        # If the current generator is the first one that does not commute, we replace it
        # by the measured operator (with phase chosen to match the sampled outcome).
        # Else we multiply the generator with the generator that we replaced to some power.
        # The power is chosen, such that the new generator commutes with the operator.
        if comm0 == 0
            # store generator i in workspace and then replace it by the operator
            @turbo for j in eachindex(stabtab.generator_workspace)
                stabtab.generator_workspace[j] = stabtab.tableau[j, i]
            end

            # replace generator i by op (phase-aware if storephase=true)
            set_operator!(stabtab, i, op)

            # IMPORTANT for mixed-state measurement:
            # we override the phase row so that the new generator stabilizes the post-measurement state,
            # dependent on the sampled outcome, and the phase of the input operator.
            if stabtab.storephase
                stabtab.tableau[2n+1, i] = kgen
            end

            comm0 = commutator
        else
            # multiply generator i by the (generator that we replaced)^(a)
            # where a = mod(-commutator * invmod(comm0, d), d)
            # this ensures that the new generator commutes with the operator
            a = mod(-commutator * stabtab.inversemod(comm0, d), d)
            mul_col_by_workspace_power!(stabtab, i, a)
        end
    end

    ######################################################
    # If comm0 != 0, we were in the noncommuting branch. #
    # Measurement was random, tableau has been updated,  #
    # and `outcome` is the returned measurement result.  #
    ######################################################
    if comm0 != 0
        stabtab.iscanonical = false
        return outcome
    end

    #########################################################
    # Commuting branch (comm0 == 0): mixed-state semantics. #
    # Need to check if op is in the stabilizer span or not. #
    #########################################################

    # check whether op is in the stabilizer span and set coeffs in c_workspace
    in_span = in_span_and_coeffs!(stabtab, op)

    if in_span
        # deterministic outcome, state unchanged

        if stabtab.storephase
            # If Q = ∏ g_j^{c[j]} has the same XZ-part as op, then Q stabilizes the state:
            # Q = ω^{kacc} * op_xz, so op_xz has eigenvalue ω^{-kacc}.
            # Q = ω^{kacc} * op = ω^{kacc} * ω^{kop} * XZ(op), so op_xz has eigenvalue ω^{-kacc}.
            kacc = phase_exponent_from_coeffs!(stabtab)
            d_phase = phase_modulus(d)

            if d == 2
                # eigenvalue is i^{-kacc + kop}
                return mod(-kacc + kop, d_phase)
            else
                # eigenvalue is ω^{-kacc + kop}
                return mod(-kacc + kop, d)  # ω^t with t = -kacc mod d
            end
        else
            # Without phase tracking, we can only say it is deterministic, not which eigenvalue.
            return 0
        end
    else
        # commuting but not in span: random outcome, and we append a generator
        (m < n) || throw(ArgumentError("Cannot append generator: tableau at full capacity (m==n)."))

        newcol = m + 1
        stabtab.m = newcol

        # insert op as a new generator
        set_operator!(stabtab, newcol, op)

        # override phase so that the new generator stabilizes the post-measurement state
        if stabtab.storephase
            stabtab.tableau[2n+1, newcol] = kgen
        end

        stabtab.iscanonical = false
        return outcome
    end
end
