"""
    entanglement_entropy(tab::StabilizerTableau, subsystem::AbstractVector)

Compute the stabilizer entanglement entropy of a subsystem.

# Arguments
- `tab::StabilizerTableau`: Stabilizer tableau describing the (possibly mixed) state.
- `subsystem::AbstractVector{<:Integer}`: 1-based qudit indices defining subsystem `A`.

# Returns
- An integer `S(A)` in log-`d` units (so `S=1` means entropy `log(d)`).

# Examples
```julia
tab = StabilizerTableau(2, 3; state=:ghz)
S1 = entanglement_entropy(tab, [1])      # 1
S2 = entanglement_entropy(tab, [1, 2])   # 1
```

# Notes
- For pure stabilizer states (`m == n`), the function uses `S(A) = S(B)` and computes the rank on the smaller side.
- For mixed stabilizer states (`m < n`), the function traces out the complement and uses the mixed-state stabilizer formula.
- `subsystem` should contain distinct indices; repeated indices lead to undefined results.
"""
function entanglement_entropy(
    tab::AbstractTableau,
    subsystem::T
) where T<:AbstractVector

    N_A = length(subsystem)
    n = tab.n
    m = tab.m

    if m == n
        # Pure stabilizer state. Use smaller side since S(A) = S(B).
        if N_A <= n ÷ 2
            rank_A = rank_subsystem_cols!(tab, subsystem, m)
            return rank_A - N_A
        end
        N_B = n - N_A
        rank_B = rank_subsystem_cols!(tab, subsystem, m; complement=true)
        return rank_B - N_B
    end

    # Mixed stabilizer state: trace out the complement B.
    # If R_B is the restriction of generators to B, then
    # m_A = m - rank(R_B), and S(ρ_A) = |A| - m_A (in log_d units).
    rank_B = rank_subsystem_cols!(tab, subsystem, m; complement=true)
    return N_A - m + rank_B
end

function rank_subsystem_cols!(
    tab::AbstractTableau,
    subsystem::AbstractVector,
    mcols::Int;
    complement::Bool=false,
)
    stab = tab.stab
    ws = tab.workspace
    n = tab.n

    if complement
        N = n - length(subsystem)
        N == 0 && return 0

        # Use c_workspace as a temporary membership mask (0/1).
        mask = tab.c_workspace
        @turbo for i in 1:n
            mask[i] = 0
        end
        @inbounds for i in eachindex(subsystem)
            mask[subsystem[i]] = 1
        end

        row = 1
        @inbounds for qudit in 1:n
            if mask[qudit] == 0
                @turbo for j in 1:mcols
                    ws[row, j] = stab[qudit, j]
                    ws[row+N, j] = stab[qudit+n, j]
                end
                row += 1
            end
        end
        return rank_fp_cols!(
            view(ws, 1:2*N, 1:mcols),
            tab.d,
            tab.inversemod,
        )
    end

    N = length(subsystem)
    N == 0 && return 0
    @turbo for j in 1:mcols, i in eachindex(subsystem)
        qudit = subsystem[i]
        ws[i, j] = stab[qudit, j]
        ws[i+N, j] = stab[qudit+n, j]
    end

    return rank_fp_cols!(
        view(ws, 1:2*N, 1:mcols),
        tab.d,
        tab.inversemod,
    )
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
