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


function measure!(stabtab::StabilizerTableau, op::Vector{Int})
    n = stabtab.n
    d = stabtab.d

    ### case 1: operator commutes with all stabilizer generators
    # in that case the measurement outcome is deterministic and the state remains unchanged
    ### case 2: operator does not commute with all stabilizer generators
    # in that case we replace the first generator that anticommutes with the operator and
    # we multiply all other generators that do not commute with the generator that we
    # replaced to the power of a, where a is the a = mod(-commutator * invmod(comm0, d), d)

    # @info "Measuring operator: ", op

    comm0 = 0  # the commutation with the first non-commuting generator
    indexxxx = 0
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
            @turbo for j in axes(stabtab.tableau, 1)
                stabtab.tableau[j, i] = op[j]
            end
            comm0 = commutator
            indexxxx = i
        else
            # multiply generator i by the (generator that we replaced)^(a)
            # where a = mod(-commutator * invmod(comm0, d), d)
            # this ensures that the new generator commutes with the operator
            @turbo for j in axes(stabtab.tableau, 1)
                stabtab.tableau[j, i] = mod(
                    stabtab.tableau[j, i] +
                    mod(
                        -commutator * stabtab.inversemod(comm0, d),
                        d
                    ) *
                    stabtab.generator_workspace[j],
                    d
                )
            end
        end
    end
    # if comm0 == 0
    #     @info "    Operator commutes with all stabilizers."
    # else
    #     @info "    Operator does not commute with all stabilizers. Replaced generator $indexxxx."
    # end

    nothing
end

# measure a Pauli on a single qudit
function measure!(stabtab::StabilizerTableau, op::FewQuditOperator)
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
            set_operator!(stabtab.tableau, i, op, n)
            # @turbo for j in axes(stabtab.tableau, 1)
            #     stabtab.tableau[j, i] = 0
            # end
            # stabtab.tableau[op.qudit, i] = op.x
            # stabtab.tableau[op.qudit+n, i] = op.z
            comm0 = commutator
        else
            # multiply generator i by the (generator that we replaced)^(a)
            # where a = mod(-commutator * invmod(comm0, d), d)
            # this ensures that the new generator commutes with the operator
            @turbo for j in axes(stabtab.tableau, 1)
                stabtab.tableau[j, i] = mod(
                    stabtab.tableau[j, i] +
                    mod(
                        -commutator * stabtab.inversemod(comm0, d),
                        d
                    ) *
                    stabtab.generator_workspace[j],
                    d
                )
            end
        end
    end
    nothing
end
