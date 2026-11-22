# Access to the tableau storage. A tableau holds its stabilizers in the fields
# `stab_x`, `stab_z` (`n × n`) and `phase` (length `n`, or 0 without
# `storephase`), and a `DestabilizerTableau` its destabilizers in `destab_x` and
# `destab_z`. Each of the four blocks is a view into a padded parent of its own,
# laid out by `_padded_block` (src/xz_block.jl), or packed bits laid out by
# `_padded_bit_block` (src/bit_block.jl). The field `layout` says whether
# generator `j` is column `j` (`GeneratorMajor`) or row `j` (`QuditMajor`) of
# every block (src/layout.jl). Only this file, the two constructor files and
# src/layout.jl touch those fields; algorithms go through `_stab`, `_destab` and
# `_phase`.
#
# The type parameters `{IM, E, W, V}` of both tableau types: `E` is the element
# type of the unpacked workspaces and, unless packed, of the blocks; `W` is the
# block word (`E`, or `UInt64` when packed); `V` is the type of `phase` and
# `xdotz_cache` (`Vector{Int}`, or `_BitPlanes` when packed). A packed tableau
# is `{IM, UInt8, UInt64, _BitPlanes}` and exists only at d = 2.

# The bounds are spelled out: with `E` left free, `StabilizerTableau{<:Any,
# <:Any, UInt64}` is not a subtype of `StabilizerTableau`, and its methods
# would lose to the generic ones.
const _BitStab = StabilizerTableau{<:InverseMod, UInt8, UInt64, _BitPlanes}
const _BitDestab = DestabilizerTableau{<:InverseMod, UInt8, UInt64, _BitPlanes}
const _BitTableau = Union{_BitStab, _BitDestab}

# The storage element type a user chose: `UInt8`, `UInt16`, `Int` or `Bits`.
# It is what `show`, `copy!` and `tableau_eltype` report.
_storage_eltype(::StabilizerTableau{IM, E}) where {IM, E} = E
_storage_eltype(::DestabilizerTableau{IM, E}) where {IM, E} = E
_storage_eltype(::_BitStab) = Bits
_storage_eltype(::_BitDestab) = Bits

# The element type of the unpacked workspaces, and of the blocks unless packed.
_entry_eltype(::StabilizerTableau{IM, E}) where {IM, E} = E
_entry_eltype(::DestabilizerTableau{IM, E}) where {IM, E} = E

"""
    tableau_eltype(tab::AbstractTableau) -> Type

The storage element type of the X and Z exponents of `tab`: `UInt8`, `UInt16`,
`Int`, or [`Bits`](@ref) for packed qubit storage. It is the `eltype` keyword
the constructors take.

# Examples
```jldoctest
julia> tableau_eltype(StabilizerTableau(3, 2))
UInt8

julia> tableau_eltype(DestabilizerTableau(2, 2; eltype = Bits))
Bits
```
"""
tableau_eltype(tab::AbstractTableau) = _storage_eltype(tab)

# Whether `E` can store a tableau at dimension `d`: `Int` always, `UInt8` and
# `UInt16` when they hold every residue mod `d` and mod `phase_modulus(d)`,
# `Bits` at d = 2, any other type never, `Union{UInt8, UInt16}` and `Union{}`
# included.
function _eltype_admissible(E::Type, d::Int)
    E === Int && return true
    E === Bits && return d == 2
    (E === UInt8 || E === UInt16) || return false
    return d - 1 <= typemax(E) && phase_modulus(d) - 1 <= typemax(E)
end

# The element type constructors use at d = 2 when no `eltype` keyword is given.
# A constant function, so the default constructor infers a concrete tableau
# type; the test suite may redefine it to run every d = 2 test on `Bits`.
_default_qubit_eltype() = UInt8

# The default storage element type for dimension `d`: `_default_qubit_eltype()`
# at d = 2, otherwise the narrowest admissible one.
function _default_eltype(d::Int)
    d == 2 && return _default_qubit_eltype()
    _eltype_admissible(UInt8, d) && return UInt8
    _eltype_admissible(UInt16, d) && return UInt16
    return Int
end

function _check_eltype(E::Type, d::Int)
    _eltype_admissible(E, d) && return nothing
    ok = join(map(nameof, filter(T -> _eltype_admissible(T, d), (UInt8, UInt16, Int, Bits))), ", ")
    throw(ArgumentError("eltype = $E cannot hold entries mod d = $d; use one of $ok."))
end

# The stabilizer and destabilizer blocks and the phase vector of a tableau,
# with `L` the tableau's layout marker. They share the tableau's arrays, so
# writing through them changes the tableau.
_stab(tab::AbstractTableau, ::L) where {L<:TableauLayout} = XZBlock{L}(tab.stab_x, tab.stab_z)
_destab(tab::DestabilizerTableau, ::L) where {L<:TableauLayout} = XZBlock{L}(tab.destab_x, tab.destab_z)
_stab(tab::_BitTableau, ::L) where {L<:TableauLayout} = BitBlock{L}(tab.stab_x, tab.stab_z, tab.n)
_destab(tab::_BitDestab, ::L) where {L<:TableauLayout} =
    BitBlock{L}(tab.destab_x, tab.destab_z, tab.n)
_phase(tab::AbstractTableau) = tab.phase

@noinline _throw_removed_field(s::Symbol, accessor::Symbol) =
    throw(ArgumentError("tab.$s was removed; use $accessor(tab)."))

@inline function Base.getproperty(tab::StabilizerTableau, s::Symbol)
    s === :stab && _throw_removed_field(s, :stabilizer_matrix)
    return getfield(tab, s)
end

@inline function Base.getproperty(tab::DestabilizerTableau, s::Symbol)
    s === :stab && _throw_removed_field(s, :stabilizer_matrix)
    s === :destab && _throw_removed_field(s, :destabilizer_matrix)
    return getfield(tab, s)
end

# A block matrix of layout `L` seen generator-major (generator `j` in column
# `j`): the matrix itself, or its lazy transpose.
_generator_major_view(::GeneratorMajor, A::AbstractMatrix) = A
_generator_major_view(::QuditMajor, A::AbstractMatrix) = transpose(A)

# The blocks of `b` stacked generator-major as a fresh `Int` matrix, X rows
# first, with `extra` further rows left for the caller to fill.
function _stacked_int_matrix(b::XZBlock{L}, extra::Int) where {L}
    n = _nq(b)
    S = Matrix{Int}(undef, 2n + extra, _ncap(b))
    S[1:n, :] .= _generator_major_view(L(), b.x)
    S[n+1:2n, :] .= _generator_major_view(L(), b.z)
    return S
end
function _stacked_int_matrix(b::XZBlock{QuditMajor}, extra::Int)
    n = _nq(b)
    S = Matrix{Int}(undef, 2n + extra, _ncap(b))
    _transpose_into!(S, 0, b.x)
    _transpose_into!(S, n, b.z)
    return S
end

# The stacked generator matrix of `tab`, generator-major in either storage layout.
function _stabilizer_generator_matrix(tab::AbstractTableau, L::TableauLayout)
    S = _stacked_int_matrix(_stab(tab, L), tab.storephase ? 1 : 0)
    tab.storephase && (S[2tab.n+1, :] .= _phase(tab))
    return S
end

# The stacked destabilizer matrix of `tab`, generator-major in either storage layout.
_destabilizer_generator_matrix(tab::DestabilizerTableau, L::TableauLayout) =
    _stacked_int_matrix(_destab(tab, L), 0)

function _check_layout(layout::Symbol)
    layout === :generator || layout === :qudit ||
        throw(ArgumentError("layout must be :generator or :qudit, got :$layout"))
    return nothing
end

"""
    stabilizer_matrix(tab::AbstractTableau; layout::Symbol=:generator) -> Matrix{Int}

Return the stabilizer generators of `tab` as a fresh `Int` matrix.

```julia
tab = StabilizerTableau(3, 2; state = :product, basis = :Z)
S = stabilizer_matrix(tab)                    # 5×2, generators in columns
Sq = stabilizer_matrix(tab; layout = :qudit)  # 2×5, generators in rows
```

With `layout = :generator` (the default) the matrix is `(2n + storephase) × n`:
generator `j` is column `j`, rows `1:n` hold the X exponents, rows `n+1:2n` the Z
exponents, and row `2n+1` the phase exponent (only when `storephase` is `true`).
All `n` columns are returned, including the zeroed capacity columns `m+1:n`.
With `layout = :qudit` the result is the `permutedims` of that matrix, with
generator `j` in row `j`.

The result is a copy; changing it does not change `tab`. Any other symbol for
`layout` throws an `ArgumentError`.
"""
function stabilizer_matrix(tab::AbstractTableau; layout::Symbol = :generator)
    _check_layout(layout)
    S = _stabilizer_generator_matrix(tab, _layout_marker(tab))
    return layout === :generator ? S : permutedims(S)
end

"""
    destabilizer_matrix(tab::DestabilizerTableau; layout::Symbol=:generator) -> Matrix{Int}

Return the destabilizer generators of `tab` as a fresh `Int` matrix.

```julia
tab = DestabilizerTableau(3, 2; state = :product, basis = :Z)
D = destabilizer_matrix(tab)                    # 4×2, generators in columns
Dq = destabilizer_matrix(tab; layout = :qudit)  # 2×4, generators in rows
```

With `layout = :generator` (the default) the matrix is `2n × n`: destabilizer
`j` is column `j`, rows `1:n` hold the X exponents and rows `n+1:2n` the Z
exponents. Destabilizers carry no phase row. With `layout = :qudit` the result
is the `permutedims` of that matrix.

The result is a copy; changing it does not change `tab`. Any other symbol for
`layout` throws an `ArgumentError`.
"""
function destabilizer_matrix(tab::DestabilizerTableau; layout::Symbol = :generator)
    _check_layout(layout)
    D = _destabilizer_generator_matrix(tab, _layout_marker(tab))
    return layout === :generator ? D : permutedims(D)
end

function _check_generator_matrix_size(src::AbstractMatrix, nrows::Int, ncols::Int)
    size(src) == (nrows, ncols) || throw(ArgumentError(
        "expected a $(nrows)×$(ncols) matrix, got $(size(src, 1))×$(size(src, 2))"))
    return nothing
end

# Entry `s` reduced mod `d` in `Int` and stored as `E`, which holds every
# residue mod `d` of an admissible `d`.
@inline _reduced_entry(::Type{E}, s::Integer, d::Int) where {E} = mod(Int(s), d) % E

# Overwrite the generators of a tableau with a generator-major matrix of exactly
# the shape `stabilizer_matrix` or `destabilizer_matrix` returns, reducing its
# entries mod `d` and its phases mod `phase_modulus(d)`, in the tableau's
# storage layout. `m`, `iscanonical` and the caches are left alone.
_load_stabilizer_matrix!(tab::AbstractTableau, S::AbstractMatrix{<:Integer}) =
    _load_stabilizer_matrix!(tab, _layout_marker(tab), S)

function _load_stabilizer_matrix!(tab::AbstractTableau, L::TableauLayout, S::AbstractMatrix{<:Integer})
    n, d, E = tab.n, tab.d, _entry_eltype(tab)
    _check_generator_matrix_size(S, 2n + (tab.storephase ? 1 : 0), n)
    _generator_major_view(L, tab.stab_x) .= _reduced_entry.(E, view(S, 1:n, :), d)
    _generator_major_view(L, tab.stab_z) .= _reduced_entry.(E, view(S, n+1:2n, :), d)
    tab.storephase && (tab.phase .= _reduced_entry.(Int, view(S, 2n + 1, :), phase_modulus(d)))
    return tab
end

_load_destabilizer_matrix!(tab::DestabilizerTableau, D::AbstractMatrix{<:Integer}) =
    _load_destabilizer_matrix!(tab, _layout_marker(tab), D)

function _load_destabilizer_matrix!(tab::DestabilizerTableau, L::TableauLayout, D::AbstractMatrix{<:Integer})
    n, d, E = tab.n, tab.d, _entry_eltype(tab)
    _check_generator_matrix_size(D, 2n, n)
    _generator_major_view(L, tab.destab_x) .= _reduced_entry.(E, view(D, 1:n, :), d)
    _generator_major_view(L, tab.destab_z) .= _reduced_entry.(E, view(D, n+1:2n, :), d)
    return tab
end

# The packed forms of the two loaders, through the bit leaves, so the padding
# bits stay zero.
function _load_stabilizer_matrix!(tab::_BitTableau, L::TableauLayout, S::AbstractMatrix{<:Integer})
    _check_generator_matrix_size(S, 2tab.n + (tab.storephase ? 1 : 0), tab.n)
    return _load_bit_generators!(tab, L, S, tab.n)
end

# Generators `1:ncols` of the packed `tab`, and their phases, from the columns
# `1:ncols` of the stacked generator-major matrix `S`; the other generators
# are left alone.
function _load_bit_generators!(tab::_BitTableau, L::TableauLayout, S::AbstractMatrix{<:Integer},
                               ncols::Int)
    n, d = tab.n, tab.d
    _load_bit_block!(_stab(tab, L), S, d, ncols)
    if tab.storephase
        dp = phase_modulus(d)
        for j in 1:ncols
            tab.phase[j] = _reduced_entry(Int, S[2n + 1, j], dp)
        end
    end
    return tab
end

function _load_destabilizer_matrix!(tab::_BitDestab, L::TableauLayout,
                                    D::AbstractMatrix{<:Integer})
    _check_generator_matrix_size(D, 2tab.n, tab.n)
    _load_bit_block!(_destab(tab, L), D, tab.d)
    return tab
end

function _load_bit_block!(b::BitBlock, S::AbstractMatrix{<:Integer}, d::Int, ncols::Int = _nq(b))
    n = _nq(b)
    for j in 1:ncols, q in 1:n
        _xset!(b, _reduced_entry(Int, S[q, j], d), q, j)
        _zset!(b, _reduced_entry(Int, S[n + q, j], d), q, j)
    end
    return b
end

##### packed construction #####

# The packed constructors build every array of the tableau at its final size
# and fill the packed blocks in place, without an unpacked copy of the blocks.

# A packed tableau of type `T` for `n` qubits in the layout `layout` (already
# checked by `_check_layout`), with `m = 0`, zero blocks, planes and
# workspaces, `iscanonical` false.
function _zero_packed_tableau(::Type{StabilizerTableau}, n::Int, storephase::Bool,
                              inversemod::IM, layout::Symbol) where {IM<:InverseMod}
    return StabilizerTableau{IM, UInt8, UInt64, _BitPlanes}(
        2, n, 0, _stab_bit_blocks(n)..., _BitPlanes(storephase ? n : 0, 2),
        storephase, false, _layout_of(layout), inversemod,
        zeros(Int, 2n + (storephase ? 1 : 0)), zeros(UInt8, 2n), zeros(UInt8, n, 3),
        _word_workspace(n),
        zeros(Int, 2n), _BitPlanes(n, 1),
        zeros(Int, 2n), zeros(Int, n), zeros(Int, n),
        _storage_rank_workspace(UInt8, 2, n), _bit_rank_workspace(2, n),
    )
end

function _zero_packed_tableau(::Type{DestabilizerTableau}, n::Int, storephase::Bool,
                              inversemod::IM, layout::Symbol) where {IM<:InverseMod}
    return DestabilizerTableau{IM, UInt8, UInt64, _BitPlanes}(
        2, n, 0, _stab_bit_blocks(n)..., _BitPlanes(storephase ? n : 0, 2),
        _destab_bit_blocks(n)...,
        storephase, false, _layout_of(layout), inversemod,
        zeros(Int, 2n + (storephase ? 1 : 0)), zeros(UInt8, 2n), zeros(UInt8, n, 3),
        _word_workspace(n),
        zeros(Int, 2n), _BitPlanes(n, 1),
        zeros(Int, 2n), zeros(Int, n), zeros(Int, n), zeros(Int, n),
        _storage_rank_workspace(UInt8, 2, n), _bit_rank_workspace(2, n),
    )
end

_layout_of(layout::Symbol) = layout === :qudit ? QuditMajor() : GeneratorMajor()

# The packed tableau of type `T` in the preset state `state_norm` (normalized
# by `_normalize_state_and_basis`), filled by `reset!`'s preset path.
function _packed_preset_tableau(::Type{T}, n::Int, state_norm::Symbol, basis_spec,
                                storephase::Bool, inversemod::InverseMod,
                                layout::Symbol) where {T<:AbstractTableau}
    tab = _zero_packed_tableau(T, n, storephase, inversemod, layout)
    _reset_preset!(tab, _layout_marker(tab), state_norm, basis_spec)
    _check_inversemod(inversemod, 2)
    _warn_if_dimension_unsafe(2, n)
    return tab
end

# The packed tableau of type `T` holding the generators of the generator-major
# raw matrix `tab_in`, whose shape `shape` comes from `_infer_tableau_layout`:
# entries reduced mod 2, phases mod 4, columns `m+1:n` zero, checked as the
# unpacked constructors check, and for a `DestabilizerTableau` with the
# destabilizers and `xdotz_cache` rebuilt.
function _packed_matrix_tableau(::Type{T}, d::Int, shape, tab_in::AbstractMatrix{<:Integer},
                                inversemod::InverseMod, check::Bool,
                                layout::Symbol) where {T<:AbstractTableau}
    n, m = shape.n, shape.m
    _check_inversemod(inversemod, d)
    (0 ≤ m ≤ n) || throw(ArgumentError("m must satisfy 0 ≤ m ≤ n."))
    _warn_if_dimension_unsafe(d, n)
    S = _generator_input(tab_in, n, m, shape.storephase)
    tab = _zero_packed_tableau(T, n, shape.storephase, inversemod, layout)
    tab.m = m
    L = _layout_marker(tab)
    _load_bit_generators!(tab, L, S, m)
    _init_packed_generators!(tab, L, check)
    return tab
end

_init_packed_generators!(tab::_BitStab, L::TableauLayout, check::Bool) =
    check && _validate_generators!(tab, L)
_init_packed_generators!(tab::_BitDestab, L::TableauLayout, check::Bool) =
    _init_destabilizers!(tab, L, nothing, nothing, check)

# The destabilizers of a packed tableau, the same as the unpacked method
# computes. The rows `[z_k x_k]` of `A = Sᵀ J` (the signs vanish mod 2) are
# packed, each followed by row `k` of an `m × m` identity, and reduced by
# Gauss-Jordan elimination over GF(2). Once every column of `A` is processed
# with `m` pivots, `A` is the identity on its pivot columns, so the identity
# part holds the inverse of `A` restricted to them.
function rebuild_destabilizers!(tab::_BitDestab, L::TableauLayout)
    n, m = tab.n, tab.m
    stab, destab = _stab(tab, L), _destab(tab, L)
    _zero!(destab)
    m == 0 && return destab

    nw = _nwords(n)
    W = 2nw + _nwords(m)
    # Column `k` of `R` is packed row `k`: Z bits in words `1:nw`, X bits in
    # words `nw+1:2nw`, the identity part after them.
    R = zeros(UInt64, W, m)
    @inbounds for k in 1:m
        for q in 1:n
            w, bit = _word_index(q), UInt64(1) << _bit_offset(q)
            _zget(stab, q, k) == 0 || (R[w, k] |= bit)
            _xget(stab, q, k) == 0 || (R[nw + w, k] |= bit)
        end
        R[2nw + _word_index(k), k] |= UInt64(1) << _bit_offset(k)
    end

    pivots = zeros(Int, m)
    r = 1
    @inbounds for c in 1:2n
        r > m && break
        cq = c <= n ? c : c - n
        w = (c <= n ? 0 : nw) + _word_index(cq)
        bit = UInt64(1) << _bit_offset(cq)
        p = r
        while p <= m && R[w, p] & bit == 0
            p += 1
        end
        p > m && continue
        if p != r
            for v in 1:W
                R[v, p], R[v, r] = R[v, r], R[v, p]
            end
        end
        # Row `r` is zero in every column before `c`, so the words before `w`
        # need no update.
        for k in 1:m
            (k == r || R[w, k] & bit == 0) && continue
            @simd for v in w:W
                R[v, k] ⊻= R[v, r]
            end
        end
        pivots[r] = c
        r += 1
    end
    # Dependent generators have no destabilizers; they stay zero.
    r - 1 == m || return destab

    @inbounds for i in 1:m, j in 1:m
        (R[2nw + _word_index(j), i] >> _bit_offset(j)) & UInt64(1) == 0 ||
            _rset!(destab, 1, pivots[i], j)
    end
    return destab
end

# The packed scratch of a packed tableau for `n` qudits.
_word_workspace(n::Int) = zeros(UInt64, max(_nwords(n), 32), 4)
