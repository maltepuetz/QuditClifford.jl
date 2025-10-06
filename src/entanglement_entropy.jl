

function entanglement_entropy(stabtab::StabilizerTableau, subsystem::T) where T<:AbstractVector

    if length(subsystem) > stabtab.n ÷ 2
        throw(
            ArgumentError(
                "Subsystem size must be less than or equal to half the number of qudits."
            )
        )
    end

    N_A = length(subsystem)
    # copy the subsystem tableau to the workspace
    @inbounds @simd for j in axes(stabtab.tableau, 2)
        for (i, qudit) in enumerate(subsystem)
            stabtab.workspace[i, j] = stabtab.tableau[qudit, j]
        end
        for (i, qudit) in enumerate(subsystem)
            stabtab.workspace[i+N_A, j] = stabtab.tableau[qudit+stabtab.n, j]
        end
    end


    ### compute the rank of the matrix stabtab.workspace[1:2*N_A, 1:stabtab.n]
    # implement Gauss-Jordan algorithm

    rank_A = rank_fp_cols!(
        view(stabtab.workspace, 1:2*N_A, 1:stabtab.n),
        stabtab.d,
        stabtab.inversemod
    )

    S_A = rank_A - N_A

    return S_A
end

function rank_fp_cols!(A::T, d::Int, inversemod::InverseMod) where T<:AbstractMatrix
    n, m = size(A)

    r = 1
    c = 1
    piv = 0
    while r <= n && c <= m
        # find a pivot in row r
        j = c
        while j <= m && A[r, j] == 0
            j += 1
        end
        if j > m
            r += 1
            continue
        end

        # swap into column c
        if j != c
            @inbounds @simd for i in axes(A, 1)
                A[i, c], A[i, j] = A[i, j], A[i, c]
            end
        end

        # scale pivot column so A[r,c] = 1
        α = inversemod(A[r, c], d)
        @inbounds @simd for i in axes(A, 1)
            A[i, c] = mod(A[i, c] * α, d)
        end

        # eliminate row r in every other column
        @inbounds for jj in axes(A, 2)
            jj == c && continue
            β = A[r, jj]
            β == 0 && continue
            for i in axes(A, 1)
                A[i, jj] = mod(A[i, jj] - mod(β * A[i, c], d), d)
            end
        end

        piv += 1
        r += 1
        c += 1
    end
    return piv
end
