"""
    is_pure(tab::AbstractTableau; verify::Bool=false) -> Bool

Return `true` when the active generators define a pure stabilizer state.

A tableau is pure exactly when it carries one independent, mutually commuting
generator per qudit. Commutativity and independence hold by construction: the
raw-matrix constructors check them (see their `check` keyword), the presets
satisfy them by definition, and `measure!` and [`canonicalize!`](@ref) both
preserve them. So the default test is `m == n`, in constant time.

Pass `verify=true` to re-derive the full condition instead of resting on that
invariant. It costs `O(n^3)` and exists for debugging: it is the only form that
can see through a tableau built behind `check=false`, or an invariant broken by
a bug.

# Arguments
- `tab::AbstractTableau`: Tableau to test. Not modified.

# Keyword Arguments
- `verify::Bool=false`: Also confirm that the generators commute and are
  independent, rather than relying on the construction-time guarantee.

# Returns
- `Bool`: whether the represented state is pure.

# Examples
```jldoctest
julia> tab = StabilizerTableau(2, 3; state = :ghz);

julia> is_pure(tab)                 # from m == n alone
true

julia> is_pure(tab; verify = true)  # re-derived from the generators
true
```

# Notes
- Neither form changes the represented state or emits diagnostic output.
- `is_pure(tab)` reads only `tab.m` and `tab.n`; `verify=true` additionally
  borrows the rank workspace for `d` (`tab.bit_workspace` for `d = 2`, else
  `tab.rank_workspace`) for both the commutation and the independence check.
"""
function is_pure(tab::AbstractTableau; verify::Bool=false)
    tab.m == tab.n || return false
    verify || return true
    return _verify_pure(tab, _layout_marker(tab))
end

# Whether the active generators of a tableau with `m == n` commute and are
# independent.
function _verify_pure(tab::AbstractTableau, L::TableauLayout)
    _first_noncommuting_pair(tab, L) == (0, 0) || return false
    _generator_rank(tab, L) == tab.m || return false
    return true
end

# Index of the first pair of active generators that fails to commute, as
# `(j, i)` with `j < i`, or `(0, 0)` when every pair commutes.
#
# The symplectic form is
#     <g_j, g_i> = sum_q (x_j[q] * z_i[q] - z_j[q] * x_i[q]),
# which for `P = X' * Z` is exactly `P[j,i] - P[i,j]`. The form is therefore
# antisymmetric, and every pair can be read off the single product `P`: the
# `i < j` half of the answer is the `j < i` half transposed, so only `P` itself
# has to be computed, not both orientations of it.
#
# `P` is written over its whole `m x m` block and needs no initialization. It
# must not alias the block. An `Int` block holds the products as they are,
# any other element type the products reduced mod `d`, which it must hold.
#
# Only the scan early-exits; the product always runs in full. The one caller
# that sees non-commuting input is construction validation, which raises as
# soon as this returns, so there is nothing to recover by stopping sooner.
#
# `s` accumulates `n` terms of size up to `(d-1)^2`, so this sits inside the
# package-wide arithmetic envelope documented on `max_safe_dimension`.
@inline function _first_noncommuting_pair(
    b::_GMBlock,
    m::Int,
    d::Int,
    P::AbstractMatrix{T},
) where {T<:_StorageEltype}
    m <= 1 && return (0, 0)
    n = _nq(b)
    x = b.x
    z = b.z
    if T === Int
        @turbo for j in 1:m, i in 1:m
            s = 0
            for q in 1:n
                s += Int(x[q, j]) * Int(z[q, i])
            end
            P[j, i] = s
        end
    else
        # Not `@turbo`: it narrows `s` to `T` before reducing it.
        @inbounds for i in 1:m, j in 1:m
            s = 0
            @simd for q in 1:n
                s += Int(x[q, j]) * Int(z[q, i])
            end
            P[j, i] = mod(s, d) % T
        end
    end
    @inbounds for j in 1:m, i in (j+1):m
        mod(Int(P[j, i]) - Int(P[i, j]), d) != 0 && return (j, i)
    end
    return (0, 0)
end

# `_first_noncommuting_pair` for d = 2 on the columns 1:m of `W`, packed by
# `_pack_sites_gf2!` into words 1:nw: site k at bits 2k-2 (X) and 2k-1 (Z).
# The form x_j·z_i + z_j·x_i mod 2 of a pair is the parity of the even bits of
# the XOR over words of `a & (b >> 1)` and `(a >> 1) & b`, as in
# `_symplectic_gram_gf2!`. Returns the first `(j, i)`, `j < i`, in that order,
# whose form is 1, or `(0, 0)`.
function _first_noncommuting_pair_gf2(W::AbstractMatrix{UInt64}, nw::Int, m::Int)
    even = 0x5555555555555555
    checkbounds(W, 1:nw, 1:m)
    @inbounds for j in 1:m, i in (j + 1):m
        acc = zero(UInt64)
        @simd for l in 1:nw
            a = W[l, j]
            b = W[l, i]
            acc = xor(acc, xor(a & (b >> 1), (a >> 1) & b))
        end
        isodd(count_ones(acc & even)) && return (j, i)
    end
    return (0, 0)
end

# The first non-commuting pair of active generators of `tab`, borrowing the
# rank workspace for `tab.d`: the packed columns in `bit_workspace` at d = 2,
# else an `m x m` block of `rank_workspace` (2n x n, and m <= n) for the
# products. Every caller reads the result before the workspace is used again.
function _first_noncommuting_pair(tab::AbstractTableau, L::TableauLayout)
    m = tab.m
    m <= 1 && return (0, 0)
    if tab.d == 2
        nw = _pack_sites_gf2!(tab.bit_workspace, _stab(tab, L), m, Base.OneTo(tab.n), tab.n)
        return _first_noncommuting_pair_gf2(tab.bit_workspace, nw, m)
    end
    return _first_noncommuting_pair(_stab(tab, L), m, tab.d, view(tab.rank_workspace, 1:m, 1:m))
end

# Check if all active generators commute. Borrows the rank workspace for `d`,
# which `is_pure` then hands to the rank check; this check is finished with it
# by then.
is_commuting(tab::AbstractTableau) = _first_noncommuting_pair(tab, _layout_marker(tab)) == (0, 0)

# Rank of the active generator columns over GF(d). Consumes the rank
# workspace that `d` selects, which the elimination is free to leave in column
# echelon form -- nothing reads it back.
_generator_rank(tab::AbstractTableau, L::TableauLayout) =
    _rank_on_sites!(tab, L, Base.OneTo(tab.n), tab.n, tab.m)

"""Check if the active stabilizer generators are linearly independent (rank m)."""
is_independent(tab::AbstractTableau) = _generator_rank(tab, _layout_marker(tab)) == tab.m

# Reject generator columns that cannot represent a stabilizer group.
#
# Both halves are contract rather than preference: the tableau stores an abelian
# subgroup generated by `m` INDEPENDENT Paulis, as both constructor docstrings
# say. `measure!` decomposes an operator against those columns as a basis, which
# means nothing if they do not span what `m` claims.
#
# Construction is the only door into the contract -- presets satisfy it by
# definition, and `measure!` and `canonicalize!` both preserve it -- so checking
# here makes it an invariant everywhere else, which is what lets `is_pure` rest
# on `m == n`.
#
# Runs on the built tableau so it can borrow the rank workspace for `d` rather
# than allocate its own -- the two checks take it in turn, never at once.
function _validate_generators!(tab::AbstractTableau, L::TableauLayout)
    m = tab.m
    # A qubit generator must also be Hermitian, a ≡ x·z (mod 2): otherwise it
    # squares to -I and stabilizes nothing. Without stored phases there is no `a`
    # to check.
    if tab.d == 2 && tab.storephase
        b, phase = _stab(tab, L), _phase(tab)
        for j in 1:m
            xz = dot_xz_col(b, j, 2)
            mod(phase[j], 2) == xz && continue
            throw(ArgumentError(
                "Qubit stabilizer generators must be Hermitian, but generator $j " *
                "has phase exponent $(phase[j]) while x·z = $xz (mod 2), so " *
                "it squares to -I. Pass check=false to skip this check."))
        end
    end
    if m > 1
        pair = _first_noncommuting_pair(tab, L)
        if pair != (0, 0)
            throw(ArgumentError(
                "Stabilizer generators must commute, but generators $(pair[1]) and " *
                "$(pair[2]) do not. Pass check=false to skip this check."))
        end
    end
    if m > 0
        r = _generator_rank(tab, L)
        r == m || throw(ArgumentError(
            "Stabilizer generators must be independent, but the $m active " *
            "generators span rank $r. Pass check=false to skip this check."))
    end
    return nothing
end
