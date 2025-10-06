function measure!(stabtab::StabilizerTableau, op::Operator)

    ### case 1: operator commutes with all stabilizer generators
    # in that case the measurement outcome is deterministic and the state remains unchanged
    ### case 2: operator does not commute with all stabilizer generators
    # in that case we replace the first generator that anticommutes with the operator and
    # we multiply all other generators that do not commute with the generator that we
    # replaced to the power of a, where a is the a = mod(-commutator * invmod(comm0, d), d)

    comm0 = 0  # the commutation with the first non-commuting generator
    # @info "---------------------------------"
    # @info stabtab
    # @info "Measuring operator: ", op.string
    for i in 1:stabtab.n
        commutator = mod(
            commutation(Generator(stabtab, i), op),
            stabtab.d
        )

        # @info "Commutation of generator $i with the operator: $commutator"

        commutator == 0 && continue

        ## case 2:
        # if view(tableau.tableau, i, :) is the first generator we replace it
        # by the operator
        # else we multiply it by the operator to the power of the commutator
        if comm0 == 0
            copyto!(Generator(stabtab.generator_workspace), Generator(stabtab, i))
            copyto!(Generator(stabtab, i), op)
            comm0 = commutator
        else
            multiply!(
                Generator(stabtab, i),
                Generator(stabtab.generator_workspace),
                mod(
                    -commutator * stabtab.inversemod(comm0, stabtab.d), stabtab.d
                ),
                stabtab.d
            )
        end
    end
    nothing
    # @info "Measurement complete."
    # @info stabtab
end
