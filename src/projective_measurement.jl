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

@inline commutation_vec_op(v::AbstractVector{<:Integer}, op::AbstractVector{<:Integer}) = commutation(v, op)
@inline commutation_vec_op(v::AbstractVector{<:Integer}, op::GeneralPauli) = commutation(v, op.xz)
@inline function commutation_vec_op(v::AbstractVector{<:Integer}, op::SinglePauli)
    N = length(v) ÷ 2
    return v[op.qudit] * op.z - v[op.qudit+N] * op.x
end
@inline function commutation_vec_op(v::AbstractVector{<:Integer}, op::DoublePauli)
    N = length(v) ÷ 2
    comm = 0
    comm += v[op.qudit1] * op.z1 - v[op.qudit1+N] * op.x1
    comm += v[op.qudit2] * op.z2 - v[op.qudit2+N] * op.x2
    return comm
end
@inline function commutation_vec_op(v::AbstractVector{<:Integer}, op::TriplePauli)
    N = length(v) ÷ 2
    comm = 0
    comm += v[op.qudit1] * op.z1 - v[op.qudit1+N] * op.x1
    comm += v[op.qudit2] * op.z2 - v[op.qudit2+N] * op.x2
    comm += v[op.qudit3] * op.z3 - v[op.qudit3+N] * op.x3
    return comm
end
@inline function commutation_vec_op(v::AbstractVector{<:Integer}, op::NPauli{D}) where D
    N = length(v) ÷ 2
    comm = 0
    @turbo for i in 1:D
        q = op.qudits[i]
        comm += v[q] * op.zs[i] - v[q+N] * op.xs[i]
    end
    return comm
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
@inline function commutation_col(A::AbstractMatrix{<:Integer}, col::Int, op::GeneralPauli)
    return commutation_col(A, col, op.xz)
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
    tab::AbstractTableau,
    col::Int,
    a::Int,
)
    stab = tab.stab
    genws = tab.generator_workspace
    n = tab.n
    d = tab.d

    a == 0 && return nothing

    nrows_block = 2n

    # If we store phase, compute cross term using the OLD target Z (before XZ update)
    if tab.storephase
        phase_row = 2n + 1
        d_phase = phase_modulus(d)

        if d == 2
            # a ∈ {0,1} and a != 0, so a == 1
            # cross = x_ws · z_col_old   (mod 2)
            cross = dot_xz_ws_vs_col(genws, stab, n, col, d)  # uses old tab Z entries
            stab[phase_row, col] = mod(stab[phase_row, col] + genws[phase_row] + 2 * cross, d_phase)
        else
            inv2 = tab.inversemod(2, d)

            # Phase of (ws)^a:
            # k_power = a*k_ws + C(a,2)*(x_ws · z_ws)  (mod d)
            xdotz_ws = dot_xz_ws(genws, n, d)
            k_power = mod(a * genws[phase_row] + binom2_mod_oddprime(a, d, inv2) * xdotz_ws, d_phase)

            # cross = a*(x_ws · z_col_old) (mod d)
            cross_base = dot_xz_ws_vs_col(genws, stab, n, col, d)  # x_ws · z_col_old
            cross = mod(a * cross_base, d)

            stab[phase_row, col] = mod(stab[phase_row, col] + k_power + cross, d_phase)
        end
    end

    # Now update XZ: col <- col + a*ws  (mod d)
    @inbounds @simd for i in 1:nrows_block
        stab[i, col] = mod(stab[i, col] + mod(a * genws[i], d), d)
    end

    return nothing
end


############################################
# Projective measurement (mixed-state)     #
############################################

@inline function _effective_measurement_phase(op::AbstractPauli, d::Int, phase_policy::Int)::Int
    (0 <= phase_policy <= 2) || throw(ArgumentError("phase_policy must be 0, 1, or 2."))
    d != 2 && return op.phase

    valid = _is_valid_measurement(op, d)
    valid && return op.phase

    if phase_policy == 0
        @warn "Measuring a non-Hermitian Pauli for d=2; results may be unphysical. Set phase_policy=2 to silence or phase_policy=1 to auto-fix."
        return op.phase
    elseif phase_policy == 1
        parity = _xdotz_parity(op)
        return _fix_qubit_phase(op.phase, parity)
    else
        return op.phase
    end
end

#############################################################
# Type-specific hooks for generic projective measurement    #
#############################################################

@inline _on_noncommuting_col_updated!(tab::AbstractTableau, _col::Int) = nothing
@inline _on_noncommuting_col_updated!(tab::DestabilizerTableau, col::Int) = _update_xdotz_cache!(tab, col)

@inline function _after_noncommuting_measurement!(
    tab::AbstractTableau,
    _pivot::Int,
    _comm0::Int,
    _m::Int,
)
    return nothing
end

@inline function _after_noncommuting_measurement!(
    tab::DestabilizerTableau,
    pivot::Int,
    comm0::Int,
    m::Int,
)
    n = tab.n
    d = tab.d

    # noncommuting branch: update destabilizers
    inv_comm0 = tab.inversemod(comm0, d)
    @inbounds for r in 1:(2n)
        tab.destab[r, pivot] = mod(inv_comm0 * tab.generator_workspace[r], d)
    end

    @inbounds for j in 1:m
        j == pivot && continue
        t = mod(_symplectic_col_col(tab.destab, j, tab.stab, pivot, n), d)
        t == 0 && continue
        @inbounds @simd for r in 1:(2n)
            tab.destab[r, j] = mod(tab.destab[r, j] - mod(t * tab.destab[r, pivot], d), d)
        end
    end

    return nothing
end

@inline function _after_append_measurement!(
    tab::AbstractTableau,
    _newcol::Int,
    _op::AbstractPauli,
    _res::AbstractVector{Int},
)
    return nothing
end

@inline function _after_append_measurement!(
    tab::DestabilizerTableau,
    newcol::Int,
    op::AbstractPauli,
    res::AbstractVector{Int},
)
    n = tab.n
    d = tab.d
    m = newcol - 1

    # Construct a new destabilizer without elimination.
    # Use residual to pick a qudit.
    q = 0
    @inbounds for i in 1:n
        if res[i] != 0 || res[n+i] != 0
            q = i
            break
        end
    end
    q == 0 && throw(ArgumentError("Failed to construct destabilizer: residual is zero."))

    genws = tab.generator_workspace
    @turbo for j in eachindex(genws)
        genws[j] = 0
    end
    if res[q] != 0
        genws[n+q] = 1  # Z_q
    else
        genws[q] = 1    # X_q
    end
    v = view(genws, 1:(2n))

    # Project v to commute with old stabilizers.
    @inbounds for j in 1:m
        t = mod(_symplectic_vec_col(v, tab.stab, j, n), d)
        t == 0 && continue
        @inbounds @simd for r in 1:(2n)
            v[r] = mod(v[r] - mod(t * tab.destab[r, j], d), d)
        end
    end

    # β = <v, op>
    β = mod(commutation_vec_op(v, op), d)
    β == 0 && throw(ArgumentError("Failed to construct destabilizer: β = 0."))
    invβ = tab.inversemod(β, d)

    # D_new = invβ * v
    @inbounds @simd for r in 1:(2n)
        tab.destab[r, newcol] = mod(invβ * v[r], d)
    end

    # Orthogonalize old destabilizers to the new stabilizer.
    @inbounds for j in 1:m
        t = mod(_symplectic_col_col(tab.destab, j, tab.stab, newcol, n), d)
        t == 0 && continue
        @inbounds @simd for r in 1:(2n)
            tab.destab[r, j] = mod(tab.destab[r, j] - mod(t * tab.destab[r, newcol], d), d)
        end
    end

    _update_xdotz_cache!(tab, newcol)
    return nothing
end

"""
    measure!(tab::AbstractTableau, op;
        outcome::Int=rand(0:tab.d-1),
        phase_policy::Int=0
    )

Projectively measure a Pauli operator `op` and update `tab` in-place.

# Arguments
- `tab::AbstractTableau`: Tableau to update (`StabilizerTableau` or `DestabilizerTableau`).
- `op`: Pauli operator specified as `AbstractPauli` (e.g. `SinglePauli`, `DoublePauli`,
  `TriplePauli`, `NPauli`, `GeneralPauli`), or an `AbstractVector{<:Integer}` of length
  `2n` or `2n+1` (wrapped into `GeneralPauli` internally).

# Keyword Arguments
- `outcome::Int=rand(0:tab.d-1)`: Outcome used when the measurement is non-deterministic
  (uniform on `0:(d-1)`).
- `phase_policy::Int=0`: How to handle non-Hermitian Paulis when `d=2`.
  - `0`: warn (default), keep phase as-is.
  - `1`: auto-fix phase to a Hermitian one, no warning.
  - `2`: ignore and keep phase as-is, no warning.

# Returns
- Integer outcome `t`.
- For odd prime `d`, the measured eigenvalue is `ω^t` (mod `d`).
- For `d=2`, the eigenvalue exponent is returned in deterministic cases when `storephase=true`
  (values `0:3` for `i^t`). In non-deterministic branches, the returned value is exactly `outcome`.

# Examples
```julia
tab = StabilizerTableau(2, 2; state=:product, basis=:Z)
op = SinglePauli(1, 1, 0)           # X on qudit 1
t = measure!(tab, op)               # random outcome, tableau updated

tab = StabilizerTableau(3, 2; state=:ghz)
op3 = DoublePauli(1, 0, 1, 2, 0, 2) # Z1 * Z2^2
t3 = measure!(tab, op3)            # deterministic outcome t3=0, tableau unchanged
```

# Notes
- If `op` does not commute with all generators, the outcome is random and the first
  non-commuting generator is replaced; other non-commuting generators are adjusted
  to restore commutation.
- If `op` commutes with all generators, the outcome is deterministic when `op` lies
  in the stabilizer span; otherwise a new generator is appended (if `m < n`).
- If `storephase=false`, deterministic outcomes return `0` by convention.
"""
function measure!(tab::AbstractTableau, op::AbstractVector{<:Integer};
    outcome::Int=rand(0:tab.d-1),
    phase_policy::Int=0,
)
    gop = GeneralPauli(tab.n, tab.d, op)
    return measure!(tab, gop; outcome=outcome, phase_policy=phase_policy)
end

function measure!(tab::AbstractTableau, op::AbstractPauli;
    outcome::Int=rand(0:tab.d-1), # outcome used if non-deterministic
    phase_policy::Int=0,
)
    d = tab.d
    n = tab.n
    m = tab.m

    # get phase of the operator (if available)
    kop = tab.storephase ? _effective_measurement_phase(op, d, phase_policy) : op.phase

    # phase exponent of the generator we should store so that it stabilizes the post-measurement state
    # If P_xz |ψ'> = ω^outcome |ψ'> then (ω^{-outcome} P_xz) |ψ'> = |ψ'>.
    # Thus if ω^{kop} P_xz |ψ'> = ω^outcome |ψ'> then (ω^{-outcome + kop} P_xz) |ψ'> = |ψ'>.
    # For d=2: (-1)^b = i^(2b), so the required stabilizer phase exponent is 2*b (mod 4).
    kgen = 0
    if tab.storephase
        if d == 2
            kgen = mod(2 * outcome + kop, phase_modulus(d))
        else
            kgen = mod(-outcome + kop, d)  # phase modulus is d for odd primes
        end
    end

    comm0 = 0  # commutation with the first non-commuting generator
    pivot = 0

    for i in 1:m

        # commutator of generator i with the operator
        commutator = mod(commutation_col(tab.stab, i, op), d)

        ### case a: if they commute we continue
        commutator == 0 && continue

        ### case b:
        # If the current generator is the first one that does not commute, we replace it
        # by the measured operator (with phase chosen to match the sampled outcome).
        # Else we multiply the generator with the generator that we replaced to some power.
        # The power is chosen, such that the new generator commutes with the operator.
        if comm0 == 0 # first non-commuting generator becomes pivot
            # store generator i in workspace and then replace it by the operator
            @turbo for j in eachindex(tab.generator_workspace)
                tab.generator_workspace[j] = tab.stab[j, i]
            end

            # replace generator i by op (phase-aware if storephase=true)
            set_operator!(tab, i, op)

            # IMPORTANT for mixed-state measurement:
            # we override the phase row so that the new generator stabilizes the post-measurement state,
            # dependent on the sampled outcome, and the phase of the input operator.
            if tab.storephase
                tab.stab[2n+1, i] = kgen
            end

            comm0 = commutator
            pivot = i
            _on_noncommuting_col_updated!(tab, i)
        else
            # multiply generator i by the (generator that we replaced)^(a)
            # where a = mod(-commutator * invmod(comm0, d), d)
            # this ensures that the new generator commutes with the operator
            a = mod(-commutator * tab.inversemod(comm0, d), d)
            mul_col_by_workspace_power!(tab, i, a)
            _on_noncommuting_col_updated!(tab, i)
        end
    end

    ######################################################
    # If comm0 != 0, we were in the noncommuting branch. #
    # Measurement was random, tableau has been updated,  #
    # and `outcome` is the returned measurement result.  #
    ######################################################
    if comm0 != 0
        _after_noncommuting_measurement!(tab, pivot, comm0, m)
        tab.iscanonical = false
        return outcome
    end

    #########################################################
    # Commuting branch (comm0 == 0): mixed-state semantics. #
    # Need to check if op is in the stabilizer span or not. #
    #########################################################

    # check whether op is in the stabilizer span and set coeffs in c_workspace
    in_span = in_span_and_coeffs!(tab, op)
    res = tab.res_workspace

    if in_span
        # deterministic outcome, state unchanged
        if tab.storephase
            kacc = phase_exponent_from_coeffs!(tab)
            d_phase = phase_modulus(d)
            if d == 2
                return mod(-kacc + kop, d_phase)
            else
                return mod(-kacc + kop, d)
            end
        else
            return 0
        end
    end
    # commuting but not in span: random outcome, append a generator
    (m < n) || throw(ArgumentError("Cannot append generator: tableau at full capacity (m==n)."))

    newcol = m + 1
    tab.m = newcol
    set_operator!(tab, newcol, op)
    if tab.storephase
        tab.stab[2n+1, newcol] = kgen
    end
    _after_append_measurement!(tab, newcol, op, res)

    tab.iscanonical = false
    return outcome
end
