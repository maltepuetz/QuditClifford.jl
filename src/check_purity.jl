"""
    is_pure(tab::AbstractTableau) -> Bool

Return `true` when the active generators define a pure stabilizer state.

Purity requires exactly one independent, mutually commuting stabilizer
generator per qudit. The check does not change the represented state and does
not emit diagnostic output.
"""
function is_pure(tab::AbstractTableau)
    tab.m == tab.n || return false
    is_commuting(tab) || return false
    is_independent(tab) || return false
    return true
end

# Index of the first pair of active generators that fails to commute, as
# `(j, i)` with `j < i`, or `(0, 0)` when every pair commutes.
#
# The symplectic form is
#     <g_j, g_i> = sum_q (x_j[q] * z_i[q] - z_j[q] * x_i[q]),
# and with `P = X' * Z` that is exactly `P[j,i] - P[i,j]`. So one matrix
# product answers every pair at once. The pairwise form spends two length-`n`
# dot products per pair; this spends `P`, which is the SAME multiply count as
# one of them -- the other is already present as the transposed entry, because
# the form is antisymmetric. That is where the factor of two comes from, and it
# is arithmetic rather than cache behaviour.
#
# `P` is written over its whole `m x m` block and needs no initialization. It
# must not alias `stab`.
#
# The product cannot abandon itself early when a pair anticommutes; only the
# scan that follows it early-exits. That costs nothing worth recovering: the
# one caller that ever sees non-commuting input is construction validation,
# which raises immediately afterwards.
#
# `s` accumulates `n` terms of size up to `(d-1)^2`, so this sits inside the
# package-wide arithmetic envelope documented on `max_safe_dimension`.
@inline function _first_noncommuting_pair(
    stab::AbstractMatrix{Int},
    n::Int,
    m::Int,
    d::Int,
    P::AbstractMatrix{Int},
)
    m <= 1 && return (0, 0)
    @turbo for j in 1:m, i in 1:m
        s = 0
        for q in 1:n
            s += stab[q, j] * stab[n+q, i]
        end
        P[j, i] = s
    end
    @inbounds for j in 1:m, i in (j+1):m
        mod(P[j, i] - P[i, j], d) != 0 && return (j, i)
    end
    return (0, 0)
end

# Check if all active generators commute.
function is_commuting(tab::AbstractTableau)
    m = tab.m
    m <= 1 && return true
    # `workspace` is 2n x n and m <= n, so the m x m block fits. Borrowing it
    # is safe here: `is_pure` runs this to completion before `is_independent`
    # touches the same buffer, and no other consumer holds it across this call.
    return _first_noncommuting_pair(
        tab.stab, tab.n, m, tab.d, view(tab.workspace, 1:m, 1:m)) == (0, 0)
end

"""Check if the active stabilizer generators are linearly independent (rank m)."""
function is_independent(tab::AbstractTableau)
    n = tab.n
    m = tab.m
    d = tab.d
    m == 0 && return true
    ws = tab.workspace
    @turbo for j in 1:m, i in 1:(2n)
        ws[i, j] = tab.stab[i, j]
    end
    rank = rank_fp_cols!(view(ws, 1:2n, 1:m), d, tab.inversemod)
    return rank == m
end
