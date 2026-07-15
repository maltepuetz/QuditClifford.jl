"""
    DestabilizerTableau(d::Int, n::Int; state::Symbol=:mixed, basis=:Z, storephase::Bool=true)
    DestabilizerTableau(d::Int, n::Int, state::Symbol; kwargs...)
    DestabilizerTableau(d::Int, tableau::AbstractMatrix{<:Integer}; m::Union{Int,Nothing}=nothing, storephase::Union{Bool,Nothing}=nothing)

Construct a stabilizer tableau augmented with destabilizers (dual generators).
Destabilizers enable fast membership tests and measurements without canonicalization.
"""
mutable struct DestabilizerTableau{T<:InverseMod} <: AbstractTableau
    d::Int                             # qudit dimension
    n::Int                             # number of qudits
    m::Int                             # number of active generator columns (0 ≤ m ≤ n)
    stab::Matrix{Int}                 # tableau has dimensions (2n + storephase) × n
    destab::Matrix{Int}                # destabilizers: 2n × n (no phase row)
    storephase::Bool                   # whether phase information is stored
    iscanonical::Bool                  # true if the tableau is in canonical (RCEF) form
    inversemod::T                      # inverse mod function for dimension d

    # workspaces for destabilizer construction
    destab_A::Matrix{Int}              # n × 2n
    destab_Awork::Matrix{Int}          # n × 2n
    destab_Ap::Matrix{Int}             # n × n
    destab_inv::Matrix{Int}            # n × n
    destab_pivots::Vector{Int}         # length n

    # workspaces
    workspace::Matrix{Int}             # 2n×n (used by entanglement_entropy)
    generator_workspace::Vector{Int}   # length 2n + storephase

    # workspace for canonicalize!
    pivcol_of_row::Vector{Int}         # length 2n (pivot column for each row if canonical)
    xdotz_cache::Vector{Int}           # length n (stores x·z values for each generator)

    # workspace for expectation value calculations
    res_workspace::Vector{Int}         # length 2n
    c_workspace::Vector{Int}           # length n
    zacc_workspace::Vector{Int}        # length n
end

function DestabilizerTableau(d::Int, n::Int;
    state::Symbol=:mixed,
    basis=:Z,
    storephase::Bool=true,
    inversemod::T=PrecomputedInvMod(d)
) where {T<:InverseMod}
    state_norm, basis_spec = _normalize_state_and_basis(state, basis, n)
    tab, m = _preset_tableau(d, n, state_norm, basis_spec, storephase)
    return _build_destabilizer_tableau(d, n, m, tab, storephase, inversemod)
end

DestabilizerTableau(d::Int, n::Int, state::Symbol; kwargs...) = DestabilizerTableau(d, n; state=state, kwargs...)

function DestabilizerTableau(d::Int, tableau::AbstractMatrix{<:Integer};
    m::Union{Int,Nothing}=nothing,
    storephase::Union{Bool,Nothing}=nothing,
    inversemod::T=PrecomputedInvMod(d)
) where {T<:InverseMod}
    m !== nothing && m < 0 && throw(ArgumentError("m must satisfy 0 ≤ m ≤ n."))
    tab_in = (tableau isa Matrix{Int}) ? tableau : Matrix{Int}(tableau)
    layout = _infer_tableau_layout(size(tab_in, 1), size(tab_in, 2), m, storephase)
    layout === nothing && throw(ArgumentError(
        "Tableau must have shape (2n + storephase) × m or m × (2n + storephase) with m ≤ n."
    ))

    if layout.transpose
        tab_in = permutedims(tab_in)
    end

    return _build_destabilizer_tableau(d, layout.n, layout.m, tab_in, layout.storephase, inversemod)
end

function _build_destabilizer_tableau(
    d::Int,
    n::Int,
    m::Int,
    stab_in::AbstractMatrix{<:Integer},
    storephase::Bool,
    inversemod::T,
) where {T<:InverseMod}
    !Primes.isprime(d) && throw(ArgumentError("Qudit dimension d must be a prime number."))
    (0 ≤ m ≤ n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ n."))

    nrows = 2n + (storephase ? 1 : 0)
    size(stab_in, 1) == nrows || throw(ArgumentError("Tableau row count must be 2n (+1 if storephase=true)."))

    # We keep capacity n columns, but accept either n columns or m columns and pad.
    stab_in = (stab_in isa Matrix{Int}) ? stab_in : Matrix{Int}(stab_in)
    stab = if size(stab_in, 2) == n
        stab_in
    elseif size(stab_in, 2) == m
        tmp = zeros(Int, nrows, n)
        tmp[:, 1:m] .= stab_in
        tmp
    else
        throw(ArgumentError("Tableau must have either n columns (capacity) or m columns (active generators)."))
    end

    # Reduce entries mod d / mod phase modulus
    @turbo for j in 1:n
        for i in 1:(2n)
            stab[i, j] = mod(stab[i, j], d)
        end
    end
    if storephase
        d_phase = phase_modulus(d)
        prow = 2n + 1
        @turbo for j in 1:n
            stab[prow, j] = mod(stab[prow, j], d_phase)
        end
    end

    # Ensure unused columns are zeroed.
    if m < n
        @turbo for j in (m+1):n, i in axes(stab, 1)
            stab[i, j] = 0
        end
    end

    tab = DestabilizerTableau{T}(
        d,
        n,
        m,
        stab,
        zeros(Int, 2n, n),
        storephase,
        false,
        inversemod,
        zeros(Int, n, 2n),
        zeros(Int, n, 2n),
        zeros(Int, n, n),
        zeros(Int, n, n),
        zeros(Int, n),
        zeros(Int, 2n, n),
        zeros(Int, nrows),
        zeros(Int, 2n),
        zeros(Int, n),
        zeros(Int, 2n),
        zeros(Int, n),
        zeros(Int, n),
    )

    rebuild_destabilizers!(tab)
    _recompute_xdotz_cache!(tab)

    return tab
end

#############################
# Destabilizer construction #
#############################

@inline function _pivot_columns!(
    A::AbstractMatrix{Int},
    m::Int,
    ncols::Int,
    d::Int,
    inversemod::InverseMod,
    pivots::Vector{Int},
)
    r = 1
    c = 1
    pivcount = 0
    while r <= m && c <= ncols
        pivot = r
        while pivot <= m && A[pivot, c] == 0
            pivot += 1
        end
        if pivot > m
            c += 1
            continue
        end

        if pivot != r
            @turbo for j in 1:ncols
                t = A[r, j]
                A[r, j] = A[pivot, j]
                A[pivot, j] = t
            end
        end

        α = inversemod(A[r, c], d)
        @turbo for j in c:ncols
            A[r, j] = mod(A[r, j] * α, d)
        end

        for rr in (r+1):m
            β = A[rr, c]
            β == 0 && continue
            @turbo for j in c:ncols
                A[rr, j] = mod(A[rr, j] - mod(β * A[r, j], d), d)
            end
        end

        pivcount += 1
        pivots[pivcount] = c
        r += 1
        c += 1
    end
    return pivcount
end

function _inv_matrix_mod!(
    invA::AbstractMatrix{Int},
    Awork::AbstractMatrix{Int},
    m::Int,
    d::Int,
    inversemod::InverseMod,
)
    (size(Awork, 1) >= m && size(Awork, 2) >= m) || throw(ArgumentError("Matrix workspace too small."))
    (size(invA, 1) >= m && size(invA, 2) >= m) || throw(ArgumentError("Inverse workspace too small."))

    fill!(invA, 0)
    @inbounds for i in 1:m
        invA[i, i] = 1
    end

    for c in 1:m
        pivot = c
        while pivot <= m && Awork[pivot, c] == 0
            pivot += 1
        end
        pivot > m && throw(ArgumentError("Matrix is singular modulo d."))

        if pivot != c
            @turbo for j in 1:m
                t = Awork[c, j]
                Awork[c, j] = Awork[pivot, j]
                Awork[pivot, j] = t

                t2 = invA[c, j]
                invA[c, j] = invA[pivot, j]
                invA[pivot, j] = t2
            end
        end

        α = inversemod(Awork[c, c], d)
        @turbo for j in 1:m
            Awork[c, j] = mod(Awork[c, j] * α, d)
            invA[c, j] = mod(invA[c, j] * α, d)
        end

        for rr in 1:m
            rr == c && continue
            β = Awork[rr, c]
            β == 0 && continue
            @turbo for j in 1:m
                Awork[rr, j] = mod(Awork[rr, j] - mod(β * Awork[c, j], d), d)
                invA[rr, j] = mod(invA[rr, j] - mod(β * invA[c, j], d), d)
            end
        end
    end

    return invA
end

function rebuild_destabilizers!(tab::DestabilizerTableau)
    n = tab.n
    m = tab.m
    d = tab.d
    stab = tab.stab
    destab = tab.destab

    fill!(destab, 0)
    m == 0 && return destab

    # Build A = Sᵀ J (size m × 2n)
    A = tab.destab_A
    @inbounds for k in 1:m
        for q in 1:n
            A[k, q] = mod(stab[n+q, k], d)
            A[k, n+q] = mod(-stab[q, k], d)
        end
    end

    Awork = tab.destab_Awork
    @turbo for j in 1:(2n), i in 1:m
        Awork[i, j] = A[i, j]
    end

    pivots = tab.destab_pivots
    pivcount = _pivot_columns!(Awork, m, 2n, d, tab.inversemod, pivots)
    # If generators are not independent, destabilizers are undefined; leave destab zeroed.
    pivcount == m || return destab

    A_P = tab.destab_Ap
    @inbounds for j in 1:m
        pj = pivots[j]
        @turbo for i in 1:m
            A_P[i, j] = A[i, pj]
        end
    end

    @turbo for j in 1:m, i in 1:m
        Awork[i, j] = A_P[i, j]
    end

    A_inv = tab.destab_inv
    _inv_matrix_mod!(A_inv, Awork, m, d, tab.inversemod)

    @inbounds for i in 1:m
        p = pivots[i]
        @turbo for j in 1:m
            destab[p, j] = mod(A_inv[i, j], d)
        end
    end
    return destab
end

#############################
# Cache maintenance helpers #
#############################

@inline function _recompute_xdotz_cache!(tab::DestabilizerTableau)
    n = tab.n
    m = tab.m
    d = tab.d
    xdotz_cache = tab.xdotz_cache
    stab = tab.stab
    @inbounds for j in 1:m
        xdotz_cache[j] = dot_xz_col(stab, n, j, d)
    end
    @inbounds for j in (m+1):n
        xdotz_cache[j] = 0
    end
    return nothing
end

@inline function _update_xdotz_cache!(tab::DestabilizerTableau, col::Int)
    tab.xdotz_cache[col] = dot_xz_col(tab.stab, tab.n, col, tab.d)
    return nothing
end

#############################
# Symplectic helpers        #
#############################

@inline function _symplectic_col_col(A::AbstractMatrix{<:Integer}, colA::Int, B::AbstractMatrix{<:Integer}, colB::Int, n::Int)
    s = 0
    @turbo for q in 1:n
        s += A[q, colA] * B[n+q, colB]
    end
    @turbo for q in 1:n
        s -= A[n+q, colA] * B[q, colB]
    end
    return s
end

@inline function _symplectic_vec_col(vec::AbstractVector{<:Integer}, B::AbstractMatrix{<:Integer}, colB::Int, n::Int)
    s = 0
    @turbo for q in 1:n
        s += vec[q] * B[n+q, colB]
    end
    @turbo for q in 1:n
        s -= vec[n+q] * B[q, colB]
    end
    return s
end

#############################
# Destabilizer-specific API #
#############################

function reset!(tab::DestabilizerTableau; state::Symbol=:mixed, basis=:Z)
    state_norm, basis_spec = _normalize_state_and_basis(state, basis, tab.n)
    m = _preset_tableau!(tab.stab, tab.d, tab.n, state_norm, basis_spec, tab.storephase)
    tab.m = m
    tab.iscanonical = false
    rebuild_destabilizers!(tab)
    _recompute_xdotz_cache!(tab)
    return tab
end

function reset!(tab::DestabilizerTableau, state::Symbol; basis=:Z)
    return reset!(tab; state=state, basis=basis)
end

# give the struct a nice standard presentation (stabilizers + destabilizers)
function Base.show(io::IO, tab::DestabilizerTableau)
    println(io, "Destabilizer Tableau:")
    println(io, "    Qudit dimension:  d = ", tab.d)
    println(io, "    Number of Qudits: n = ", tab.n)
    println(io, "    Generators:       m = ", tab.m)

    tab.n >= max_qudits_display[] && (println(io, "    Tableau is too large to display."); return)

    if tab.m < tab.n
        println(io, "    Mixed tableau (m < n generators):")
    else
        println(io, "    Tableau:")
    end

    if tab.m == 0
        println(io, "    (no generators; maximally mixed on the full space)")
        return
    end

    tabview = view(tab.stab, :, 1:tab.m)
    destview = view(tab.destab, 1:(2*tab.n), 1:tab.m)
    maxstab = maximum(abs, tabview)
    maxdest = maximum(abs, destview)
    N = ndigits(max(maxstab, maxdest)) - 1
    extraspace = 0
    isodd(N) && isodd(tab.n) && (extraspace += 1)
    dash_len = max(0, tab.n * (N + 2) ÷ 2 - 2 + extraspace)

    header = if tab.n > 1
        left = "-"^dash_len * " X " * "-"^dash_len
        sep = extraspace == 1 ? "  " : "   "
        right = "-"^dash_len * " Z " * "-"^dash_len
        left * sep * right
    else
        left = "-"^dash_len * "X " * "-"^dash_len
        right = "-"^dash_len * " Z " * "-"^dash_len
        left * " " * right
    end

    # Stabilizers
    println(io, "     Stabilizers:")
    println(io, "     ", header)
    for col in 1:tab.m
        print(io, "    ")
        for row in 1:size(tab.stab, 1)
            ((row == tab.n + 1) || (row == 2 * tab.n + 1)) && print(io, " |")
            print(io, lpad(tab.stab[row, col], N + 2))
        end
        println(io)
    end

    # Separator + Destabilizers
    println(io, "     ", "-"^length(header))
    println(io, "     Destabilizers:")
    println(io, "     ", header)
    for col in 1:tab.m
        print(io, "    ")
        for row in 1:(2*tab.n)
            (row == tab.n + 1) && print(io, " |")
            print(io, lpad(tab.destab[row, col], N + 2))
        end
        println(io)
    end
end
