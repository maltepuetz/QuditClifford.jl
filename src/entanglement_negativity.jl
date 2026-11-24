"""
    entanglement_negativity(tab::AbstractTableau, A::AbstractVector, B::AbstractVector)
    entanglement_negativity(tab::AbstractTableau, A::AbstractVector)

Return the logarithmic negativity `ℰ(A:B) = log_d ‖ρ_ABᵀᴮ‖₁` between the
disjoint subsystems `A` and `B`, in log-`d` units (so `ℰ = 1` means `log(d)`).
`ρ_AB` is the state of `A ∪ B` with every other qudit traced out, and `ᵀᴮ` is
the partial transpose on `B`. With only `A` given, `B` is its complement.

Unlike the entropy of a mixed state, the negativity vanishes on classical
correlations: it measures the entanglement between `A` and `B` alone. For a
pure state and `B` the complement of `A` it equals
[`entanglement_entropy`](@ref)`(tab, A)`. It is symmetric in `A` and `B`, and
it is zero when either is empty.

# Arguments
- `tab::AbstractTableau`: Destabilizer or stabilizer tableau describing the
  (possibly mixed) state. Phase storage is not required.
- `A`, `B`: 1-based qudit indices. Each must contain distinct sites in
  `1:tab.n`, and no site may be in both; otherwise an `ArgumentError` is
  thrown. The check is always performed and costs `O(n)`.

# Returns
- An `Int`. The stabilizer state `ρ_AB` has a negativity spectrum that is flat,
  so `ℰ(A:B)` is always a whole number of `log(d)` units.

# Method
The generators are reduced to a basis of the stabilizer group of `ρ_AB` (the
elements supported on `A ∪ B`), and `ℰ(A:B)` is half the rank over `F_d` of
the matrix of their pairwise symplectic products restricted to `A`. Since the
elements commute, that matrix is minus the one restricted to `B`, so the
smaller side is used. This takes at most `O(n*m^2)` time, cubic in `n`, reads
only the X and Z rows, and leaves the generators, phases, destabilizers and
canonicalization caches unchanged. It overwrites the scratch arrays
`tab.c_workspace` and `tab.zacc_workspace` and the rank workspace for `d`
(`tab.bit_workspace` for `d = 2`, else `tab.rank_workspace`), and allocates
nothing unless `A` or `B` shares memory with `tab.c_workspace`, in which case
it is copied first.

# References
- S. Sang, Y. Li, T. Zhou, X. Chen, T. H. Hsieh and M. P. A. Fisher,
  *Entanglement Negativity at Measurement-Induced Criticality*,
  PRX Quantum **2**, 030313 (2021), arXiv:2012.00031. Derives
  `ℰ = rank(K)/2` for qubit stabilizer states, with `K` the anticommutation
  matrix over `F_2`.
- D. Fattal, T. S. Cubitt, Y. Yamamoto, S. Bravyi and I. L. Chuang,
  *Entanglement in the stabilizer formalism*, arXiv:quant-ph/0406168 (2004).
  The local reduction of bipartite stabilizer states to EPR pairs, which
  underlies the formula.

The qudit form used here, with the symplectic product `x_i·z_j - z_i·x_j`
over `F_d`, is the same construction for prime `d`. The package tests check
it against a dense partial transpose for `d = 2, 3, 5, 7`.

# Examples
```jldoctest
julia> bell = StabilizerTableau(3, 2; state = :ghz);

julia> entanglement_negativity(bell, [1], [2])
1

julia> ghz = StabilizerTableau(3, 3; state = :ghz);

julia> entanglement_negativity(ghz, [1], [2])   # tracing out qudit 3 leaves only classical correlations
0

julia> entanglement_negativity(ghz, [1]) == entanglement_entropy(ghz, [1]) == 1
true
```
"""
function entanglement_negativity(
    tab::AbstractTableau,
    A::AbstractVector,
    B::AbstractVector,
)
    NA, NB = _mark_subsystems!(tab, (A, B))
    return _negativity_from_mask!(tab, _layout_marker(tab), NA, NB)
end

function entanglement_negativity(tab::AbstractTableau, A::AbstractVector)
    A = Base.unalias(tab.c_workspace, A)
    _mark_with_complement!(tab, A)
    NA = Int(length(A))
    return _negativity_from_mask!(tab, _layout_marker(tab), NA, tab.n - NA)
end

# Validate `A` and label it 1 in `c_workspace`, and every other site 2.
function _mark_with_complement!(tab::AbstractTableau, A::AbstractVector)
    _check_subsystem!(tab, A)
    mask = tab.c_workspace
    @inbounds for q in 1:tab.n
        mask[q] = mask[q] == 1 ? 1 : 2
    end
    return nothing
end

# Group the sites by their label in `c_workspace`, given `counts[l + 1]`
# sites with each label `l` in 0:3: write them to `tab.zacc_workspace`, label
# by label and increasing within a label, and return the index at which each
# label's block starts. Copying rows through this list goes column by column,
# like the copy in `rank_subsystem_cols!`, instead of one strided row at a time.
function _group_sites_by_label!(tab::AbstractTableau, counts::NTuple{4,Int})
    mask = tab.c_workspace
    sites = tab.zacc_workspace
    first = (1, 1 + counts[1], 1 + counts[1] + counts[2],
             1 + counts[1] + counts[2] + counts[3])
    p0, p1, p2, p3 = first
    @inbounds for q in 1:tab.n
        label = mask[q]
        if label == 0
            sites[p0] = q
            p0 += 1
        elseif label == 1
            sites[p1] = q
            p1 += 1
        elseif label == 2
            sites[p2] = q
            p2 += 1
        else
            sites[p3] = q
            p3 += 1
        end
    end
    return first
end

# Copy the X and Z rows of the `N` sites listed in `tab.zacc_workspace` from
# index `first` on into the rank workspace `ws` after row `offset`: X exponents
# to rows offset+1:offset+N and Z exponents to offset+N+1:offset+2N, columns
# 1:m. Returns the last row written, offset + 2N.
function _copy_site_rows!(tab::AbstractTableau, L::TableauLayout, ws::AbstractMatrix, first::Int,
                          N::Int, offset::Int)
    sites = view(tab.zacc_workspace, first:(first + N - 1))
    return _copy_sites!(ws, _stab(tab, L), tab.m, sites, N, offset)
end

# The dot products x_i·z_j and z_i·x_j of columns `ci` and `cj` of `ws`, with
# X exponents on rows xS+1:xS+len and Z exponents on rows zS+1:zS+len, each
# reduced mod `d` into `0:d-1`. On `Int` each sum has len <= n terms below
# (d-1)^2, inside the package's arithmetic envelope (see
# `max_safe_dimension`); the narrow methods are exact for every `len` and `d`.
# The `Int` method needs `len >= 1`: on Julia 1.10 its `@turbo` loop reads
# memory even for an empty range, which can crash. The caller has len >= 1.
@inline function _cross_dots(ws::AbstractMatrix{Int}, xS::Int, zS::Int, len::Int, ci::Int, cj::Int,
                             d::Int)
    xz = 0
    zx = 0
    @turbo for t in 1:len
        xz += ws[xS + t, ci] * ws[zS + t, cj]
        zx += ws[zS + t, ci] * ws[xS + t, cj]
    end
    return rem(xz, d), rem(zx, d)
end

# The sums run in `UInt64`, which holds len·(d-1)^2 for any `len` below 2^32
# at `d <= 65535`, so they are exact on 32-bit too, where an `Int` product
# alone overflows past d = 46341.
@inline function _cross_dots(ws::AbstractMatrix{UInt16}, xS::Int, zS::Int, len::Int, ci::Int,
                             cj::Int, d::Int)
    xz = zero(UInt64)
    zx = zero(UInt64)
    @inbounds @simd for t in 1:len
        xz += UInt64(ws[xS + t, ci]) * UInt64(ws[zS + t, cj])
        zx += UInt64(ws[zS + t, ci]) * UInt64(ws[xS + t, cj])
    end
    return Int(xz % UInt64(d)), Int(zx % UInt64(d))
end

# A `UInt32` sum fits twice as many lanes in a vector as the `UInt64` lanes
# an `Int` accumulator takes, and an `Int` sum of `UInt8` products does not
# vectorize at all. Runs of `chunk` terms (at most `_UINT8_DOT_CHUNK`)
# are summed in `UInt32` and added into `UInt64` totals.
@inline function _cross_dots(ws::AbstractMatrix{UInt8}, xS::Int, zS::Int, len::Int, ci::Int,
                             cj::Int, d::Int, chunk::Int=_UINT8_DOT_CHUNK)
    xz = zero(UInt64)
    zx = zero(UInt64)
    t0 = 0
    while t0 < len
        t1 = min(t0 + chunk, len)
        a = zero(UInt32)
        b = zero(UInt32)
        @inbounds @simd for t in (t0 + 1):t1
            a += UInt32(ws[xS + t, ci]) * UInt32(ws[zS + t, cj])
            b += UInt32(ws[zS + t, ci]) * UInt32(ws[xS + t, cj])
        end
        xz += a
        zx += b
        t0 = t1
    end
    return Int(xz % UInt64(d)), Int(zx % UInt64(d))
end

# `c_workspace[q]` labels site `q`: 1 for A, 2 for B and 0 for a traced-out
# site. `NA` and `NB` are the sizes of A and B.
function _negativity_from_mask!(tab::AbstractTableau, L::TableauLayout, NA::Int, NB::Int)
    (NA == 0 || NB == 0 || tab.m == 0) && return 0
    return _negativity_from_mask!(tab, L, _rank_workspace(tab), NA, NB)
end

function _negativity_from_mask!(tab::AbstractTableau, L::TableauLayout, ws::AbstractMatrix, NA::Int,
                                NB::Int)
    n, m, d = tab.n, tab.m, tab.d

    # ℰ(A:B) = ℰ(B:A), so the generators are restricted to the smaller side S.
    side = NA <= NB ? 1 : 2
    NS = min(NA, NB)
    NC = n - NA - NB

    # Rows 1:R0 hold the X and Z exponents on the traced-out sites, rows
    # R0+1:R those on S (see `_copy_sites!` for the layout).
    first = _group_sites_by_label!(tab, (NC, NA, NB, 0))
    R0 = _copy_site_rows!(tab, L, ws, first[1], NC, 0)
    R = _copy_site_rows!(tab, L, ws, first[side + 1], NS, R0)

    # Pivoting on the traced-out rows alone leaves columns piv+1:m zero there,
    # and those columns are a basis of the stabilizer group of ρ_AB.
    piv = _block_rank!(ws, 0, R, 1:m, d, tab.inversemod, R0)
    k = m - piv
    k <= 1 && return 0

    # K[i, j] is the symplectic product on S of basis elements i and j. It is
    # alternating, so its rank is even. K occupies rows R+1:R+k and columns
    # 1:k, disjoint from the rows 1:R it is computed from. It fits: ρ_AB has
    # at most NA + NB independent stabilizers, so k <= NA + NB, and
    # 2NS <= NA + NB, so 2NC + 2NS + k <= 2n. On the bit-packed workspace the
    # three blocks each start on a fresh word, which takes at most two words
    # more than cld(2n, 64), as `_bit_rank_workspace` provides.
    _symplectic_gram!(ws, R0, NS, R, piv, k, d)
    return _block_rank!(ws, R, R + k, 1:k, d, tab.inversemod) ÷ 2
end

# Write K (see `_negativity_from_mask!`) to rows R+1:R+k and columns 1:k of
# `ws`, from the basis columns piv+1:piv+k restricted to the `NS` sites whose
# X exponents are on rows R0+1:R0+NS and Z exponents on R0+NS+1:R0+2NS.
function _symplectic_gram!(ws::AbstractMatrix, R0::Int, NS::Int, R::Int, piv::Int, k::Int, d::Int)
    xS = R0
    zS = R0 + NS
    T = eltype(ws)
    @inbounds for j in 1:k
        cj = piv + j
        ws[R + j, j] = zero(T)
        for i in 1:(j - 1)
            xz, zx = _cross_dots(ws, xS, zS, NS, piv + i, cj, d)
            v = xz - zx
            v < 0 && (v += d)
            ws[R + i, j] = v % T
            ws[R + j, i] = (v == 0 ? 0 : d - v) % T
        end
    end
    return ws
end

# On the bit-packed workspace the sites of S fill the words after bit R0, and
# K over GF(2) is packed into the words after bit R.
_symplectic_gram!(ws::AbstractMatrix{UInt64}, R0::Int, NS::Int, R::Int, piv::Int, k::Int, ::Int) =
    _symplectic_gram_gf2!(ws, R0 >> 6, cld(2NS, 64), piv, k, R >> 6)
