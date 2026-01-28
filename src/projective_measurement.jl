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


"""
Multiply a tableau column by (generator_workspace)^a on the RIGHT, with correct phase updates.

This is exactly what you want in measurement updates:
    g_i <- g_i * (g0)^a
where g0 is the generator you stored in generator_workspace.

- XZ rows are updated linearly mod d.
- If storephase=false: done.
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
            # a ∈ {0,1} and a != 0 here ⇒ a == 1
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

"""
Set a tableau column equal to an operator vector `op` that contains XZ (and optionally phase).
- XZ entries are reduced mod d.
- If storephase=true:
    - phase entry is reduced mod phase_modulus(d) (mod d for odd primes, mod 4 for d=2)
- If storephase=false:
    - ignores any phase entry (if provided).
"""
function set_operator!(
    stabtab::StabilizerTableau,
    col::Int,
    op::AbstractVector{<:Integer},
)
    tab = stabtab.tableau
    n = stabtab.n
    d = stabtab.d

    @assert length(op) == (stabtab.storephase ? (2n + 1) : (2n))

    # XZ
    @turbo for i in 1:(2n)
        tab[i, col] = mod(op[i], d)
    end

    # phase
    if stabtab.storephase
        d_phase = phase_modulus(d)
        tab[2n+1, col] = mod(op[2n+1], d_phase)
    end

    return nothing
end

############################################
# Projective measurement (PHASE-AWARE)     #
############################################

function measure!(stabtab::StabilizerTableau, op)
    n = stabtab.n
    d = stabtab.d

    ### case 1: operator commutes with all stabilizer generators
    # in that case the measurement outcome is deterministic and the state remains unchanged
    ### case 2: operator does not commute with all stabilizer generators
    # in that case we replace the first generator that anticommutes with the operator and
    # we multiply all other generators that do not commute with the generator that we
    # replaced to the power of a, where a is the a = mod(-commutator * invmod(comm0, d), d)

    comm0 = 0  # the commutation with the first non-commuting generator
    for i in 1:n

        # get the commutator of generator i with the operator
        commutator = mod(
            commutation_col(stabtab.tableau, i, op),
            d
        )

        ### case a: if they commute we continue
        commutator == 0 && continue

        ### case b:
        # if the current generator is the first one that does not commute, we replace it
        # by the operator. Else we multiply the generator with the generator that we
        # replaced to some power. The power is chosen, such that the new generator commutes
        # with the operator.
        if comm0 == 0
            # store generator i in workspace and then replace it by the operator
            @turbo for j in eachindex(stabtab.generator_workspace)
                stabtab.generator_workspace[j] = stabtab.tableau[j, i]
            end

            # replace generator i by op (phase-aware if storephase=true)
            set_operator!(stabtab, i, op)

            comm0 = commutator
        else
            # multiply generator i by the (generator that we replaced)^(a)
            # where a = mod(-commutator * invmod(comm0, d), d)
            # this ensures that the new generator commutes with the operator
            a = mod(-commutator * stabtab.inversemod(comm0, d), d)
            mul_col_by_workspace_power!(stabtab, i, a)
        end
    end

    nothing
end

# # measure a Pauli on a few qudits
# function measure!(stabtab::StabilizerTableau, op::FewQuditOperator)
#     n = stabtab.n
#     d = stabtab.d

#     ### case 1: operator commutes with all stabilizer generators
#     # in that case the measurement outcome is deterministic and the state remains unchanged
#     ### case 2: operator does not commute with all stabilizer generators
#     # in that case we replace the first generator that anticommutes with the operator and
#     # we multiply all other generators that do not commute with the generator that we
#     # replaced to the power of a, where a is the a = mod(-commutator * invmod(comm0, d), d)

#     comm0 = 0  # the commutation with the first non-commuting generator
#     for i in 1:n

#         # get the commutator of generator i with the operator
#         commutator = mod(
#             commutation_col(stabtab.tableau, i, op),
#             d
#         )

#         ### case a: if they commute we continue
#         commutator == 0 && continue

#         ### case b:
#         # if the current generator is the first one that does not commute, we replace it
#         # by the operator. Else we multiply the generator with the generator that we
#         # replaced to some power. The power is chosen, such that the new generator commutes
#         # with the operator.
#         if comm0 == 0
#             # store generator i in workspace and then replace it by the operator
#             @turbo for j in eachindex(stabtab.generator_workspace)
#                 stabtab.generator_workspace[j] = stabtab.tableau[j, i]
#             end

#             # replace generator i by op
#             # NOTE: Your existing set_operator! likely writes only XZ.
#             # If you want to include phase for operator measurements, you should extend it.
#             set_operator!(stabtab, i, op)

#             comm0 = commutator
#         else
#             # multiply generator i by the (generator that we replaced)^(a)
#             # where a = mod(-commutator * invmod(comm0, d), d)
#             # this ensures that the new generator commutes with the operator
#             a = mod(-commutator * stabtab.inversemod(comm0, d), d)

#             # IMPORTANT:
#             # This path uses generator_workspace as the "replaced generator", so we need the
#             # phase-aware update (not a naive linear update).
#             mul_col_by_workspace_power!(stabtab, i, a)
#         end
#     end
#     nothing
# end
