function entanglement_entropy(
    stabtab::StabilizerTableau,
    subsystem::T
) where T<:AbstractVector

    # Entanglement entropy formula implemented here is for PURE stabilizer states.
    # For mixed stabilizer density operators (m < n) this does not return the von Neumann entropy.
    (stabtab.m == stabtab.n) || throw(ArgumentError("entanglement_entropy is only implemented for pure stabilizer states (m==n)."))

    N_A = length(subsystem)
    tab = stabtab.tableau
    ws = stabtab.workspace
    n = stabtab.n

    @turbo for j in axes(tab, 2), i in eachindex(subsystem)
        qudit = subsystem[i]
        ws[i, j] = tab[qudit, j]
        ws[i+N_A, j] = tab[qudit+n, j]
    end

    ### compute the rank of the matrix stabtab.workspace[1:2*N_A, 1:stabtab.n]
    # implement Gauss-Jordan algorithm

    rank_A = rank_fp_cols!(
        view(ws, 1:2*N_A, 1:n),
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

        if j != c
            @turbo for i in axes(A, 1)
                t = A[i, c]
                A[i, c] = A[i, j]
                A[i, j] = t
            end
        end

        # scale pivot column so A[r,c] = 1
        α = inversemod(A[r, c], d)
        @turbo for i in axes(A, 1)
            A[i, c] = mod(A[i, c] * α, d)
        end

        # eliminate row r in every other column
        for jj in axes(A, 2)
            jj == c && continue
            β = A[r, jj]
            β == 0 && continue
            @inbounds @simd for i in axes(A, 1)
                A[i, jj] = mod(A[i, jj] - mod(β * A[i, c], d), d)
            end
        end

        piv += 1
        r += 1
        c += 1
    end
    return piv
end
