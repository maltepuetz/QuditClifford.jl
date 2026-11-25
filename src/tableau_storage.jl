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
# `UInt16` when `d <= typemax(E)`, `Bits` at d = 2, any other type never,
# `Union{UInt8, UInt16}` and `Union{}` included.
#
# `d <= typemax(E)` is what the narrow modular primitives need; it also holds
# every residue mod `d` and mod `phase_modulus(d)` (4 at d = 2, `d` otherwise).
# The largest primes it admits are 251 for `UInt8` and 65521 for `UInt16`.
function _eltype_admissible(E::Type, d::Int)
    E === Int && return true
    E === Bits && return d == 2
    (E === UInt8 || E === UInt16) || return false
    return d <= typemax(E)
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
    need = (E === UInt8 || E === UInt16) ? " ($E needs d ≤ $(Int(typemax(E))))" : ""
    throw(ArgumentError("eltype = $E cannot store a tableau at d = $d$need; use one of $ok."))
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

# Entry `s` of any integer type reduced mod `d` and stored as `E`, which holds
# every residue mod `d` of an admissible `d`. The reduction runs in the type
# `mod` promotes `s` and `d` to, so no entry is converted to `Int` first.
@inline _reduced_entry(::Type{E}, s::Integer, d::Int) where {E} = mod(s, d) % E

# Overwrite the generators of a tableau with a generator-major matrix of exactly
# the shape `stabilizer_matrix` or `destabilizer_matrix` returns, reducing its
# entries mod `d` and its phases mod `phase_modulus(d)`, in the tableau's
# storage layout. `m`, `iscanonical` and the caches are left alone. The package
# itself never calls these two loaders: the tests and the benchmark workloads
# (benchmark/workloads.jl) build tableaux in arbitrary states with them, so
# they stay next to the storage they write.
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

##### construction #####

# Both tableau types are built the same way: every array at its final size,
# zeroed, with the layout field set (`_new_tableau`), then filled in place. A
# preset is written by `reset!`'s preset path in the requested layout. Raw
# generators are loaded generator-major into an unpacked tableau, by a loop that
# vectorizes for a `Matrix{Int}`, which is switched to the requested layout
# after validation (and, for a `DestabilizerTableau`, the rebuild of the
# duals); a packed tableau is loaded in its own layout through the bit leaves.

# The arrays of a new tableau for the storage type `E` the user chose (`Bits`
# or an element type), all zero: the stabilizer and destabilizer blocks at
# their slots, the phase, the workspaces in field order from
# `generator_workspace` to `zacc_workspace`, and the two rank workspaces.
# `_copied_*` give the same tuples for `copy`.
_new_stab_blocks(::Type{E}, n::Int) where {E<:_StorageEltype} = _stab_blocks(E, n)
_new_stab_blocks(::Type{Bits}, n::Int) = _stab_bit_blocks(n)
_new_destab_blocks(::Type{E}, n::Int) where {E<:_StorageEltype} = _destab_blocks(E, n)
_new_destab_blocks(::Type{Bits}, n::Int) = _destab_bit_blocks(n)
_new_phase(::Type{<:_StorageEltype}, n::Int) = zeros(Int, n)
_new_phase(::Type{Bits}, n::Int) = _BitPlanes(n, 2)
_new_cache(::Type{<:_StorageEltype}, n::Int) = zeros(Int, n)
_new_cache(::Type{Bits}, n::Int) = _BitPlanes(n, 1)
_new_word_workspace(::Type{<:_StorageEltype}, ::Int) = zeros(UInt64, 0, 0)
_new_word_workspace(::Type{Bits}, n::Int) = _word_workspace(n)
# The element type of the unpacked workspaces: `UInt8` for a packed tableau.
_workspace_eltype(::Type{E}) where {E<:_StorageEltype} = E
_workspace_eltype(::Type{Bits}) = UInt8

function _new_workspaces(::Type{E}, n::Int, storephase::Bool) where {E}
    U = _workspace_eltype(E)
    return (zeros(Int, 2n + (storephase ? 1 : 0)), zeros(U, 2n), zeros(U, n, 3),
            _new_word_workspace(E, n), zeros(Int, 2n), _new_cache(E, n),
            zeros(Int, 2n), zeros(Int, n), zeros(Int, n))
end
_new_rank_workspaces(::Type{E}, d::Int, n::Int) where {E} =
    (_storage_rank_workspace(_workspace_eltype(E), d, n), _bit_rank_workspace(d, n))

_copied_stab(tab::AbstractTableau) =
    (_copy_gen_block(tab.stab_x, tab.n), _copy_gen_block(tab.stab_z, tab.n), copy(tab.phase))
_copied_workspaces(tab::AbstractTableau) =
    (zero(tab.generator_workspace), zero(tab.pivot_workspace), zero(tab.coeff_workspace),
     zero(tab.word_workspace), copy(tab.pivcol_of_row), copy(tab.xdotz_cache),
     zero(tab.res_workspace), zero(tab.c_workspace), zero(tab.zacc_workspace))
_copied_rank_workspaces(tab::AbstractTableau) = (zero(tab.rank_workspace), zero(tab.bit_workspace))

# A tableau of type `T` with storage `E` for `n` qudits in layout `L`: `m = 0`,
# zero blocks, phase, cache and workspaces, `iscanonical` false.
function _new_tableau(::Type{StabilizerTableau}, ::Type{E}, d::Int, n::Int, storephase::Bool,
                      inversemod::InverseMod, L::TableauLayout) where {E}
    return StabilizerTableau(d, n, 0, _new_stab_blocks(E, n)...,
                             _new_phase(E, storephase ? n : 0), storephase, false, L, inversemod,
                             _new_workspaces(E, n, storephase)..., _new_rank_workspaces(E, d, n)...)
end

function _new_tableau(::Type{DestabilizerTableau}, ::Type{E}, d::Int, n::Int, storephase::Bool,
                      inversemod::InverseMod, L::TableauLayout) where {E}
    return DestabilizerTableau(d, n, 0, _new_stab_blocks(E, n)...,
                               _new_phase(E, storephase ? n : 0), _new_destab_blocks(E, n)...,
                               storephase, false, L, inversemod, _new_workspaces(E, n, storephase)...,
                               zeros(Int, n), _new_rank_workspaces(E, d, n)...)
end

_layout_of(layout::Symbol) = layout === :qudit ? QuditMajor() : GeneratorMajor()

# The checks on `d` and `inversemod` both constructors make once the shape of
# the tableau is known, and the warning for a dimension past the overflow
# envelope.
function _check_dimension(d::Int, n::Int, inversemod::InverseMod)
    Primes.isprime(d) || throw(ArgumentError("Qudit dimension d must be a prime number."))
    _check_inversemod(inversemod, d)
    _warn_if_dimension_unsafe(d, n)
    return nothing
end

# The `(d, n)` constructors of both tableau types: the tableau of type `T` with
# storage `E` in the preset state `state` (with `basis`), built in the layout
# `layout`.
function _preset_tableau(::Type{T}, ::Type{E}, d::Int, n::Int, state::Symbol, basis,
                         storephase::Bool, inversemod::InverseMod,
                         layout::Symbol) where {T<:AbstractTableau, E}
    n >= 0 || throw(ArgumentError("n must be nonnegative, got $n"))
    _check_eltype(E, d)
    _check_layout(layout)
    state_norm, basis_spec = _normalize_state_and_basis(state, basis, n)
    tab = _new_tableau(T, E, d, n, storephase, inversemod, _layout_of(layout))
    _reset_preset!(tab, _layout_marker(tab), state_norm, basis_spec, true)
    _check_dimension(d, n, inversemod)
    return tab
end

# The raw-matrix constructors of both tableau types: the tableau of type `T`
# with storage `E` holding the generators of `tableau`, stacked
# generator-major or transposed (`_infer_tableau_shape`): entries reduced mod
# `d`, phases mod `phase_modulus(d)`, columns `m+1:n` zero, checked with
# `check`, and for a `DestabilizerTableau` with the destabilizers and
# `xdotz_cache` rebuilt. The input is never written.
function _matrix_tableau(::Type{T}, ::Type{E}, d::Int, tableau::AbstractMatrix{<:Integer},
                         m::Union{Int,Nothing}, storephase::Union{Bool,Nothing},
                         inversemod::InverseMod, check::Bool,
                         layout::Symbol) where {T<:AbstractTableau, E}
    _check_eltype(E, d)
    _check_layout(layout)
    m !== nothing && m < 0 && throw(ArgumentError("m must satisfy 0 ≤ m ≤ n."))
    # The input is read entry by entry with 1-based indices.
    Base.require_one_based_indexing(tableau)
    shape = _infer_tableau_shape(size(tableau, 1), size(tableau, 2), m, storephase)
    shape === nothing && throw(ArgumentError(
        "Tableau must have shape (2n + storephase) × m or m × (2n + storephase) with m ≤ n."
    ))
    _check_dimension(d, shape.n, inversemod)

    # A transposed input is read through a lazy `transpose`, with no copy.
    S = shape.transpose ? transpose(tableau) : tableau
    packed = E === Bits
    tab = _new_tableau(T, E, d, shape.n, shape.storephase, inversemod,
                       packed ? _layout_of(layout) : GeneratorMajor())
    tab.m = shape.m
    L = _layout_marker(tab)
    packed ? _load_bit_generators!(tab, L, S, shape.m) : _load_generators!(tab, S, shape.m)
    _init_from_generators!(tab, L, check)
    return packed ? tab : _apply_layout!(tab, layout)
end

# Generators `1:m` of the unpacked generator-major `tab`, and their phases,
# from the stacked generator-major matrix `S`.
function _load_generators!(tab::AbstractTableau, S::AbstractMatrix{<:Integer}, m::Int)
    n, d = tab.n, tab.d
    _load_generator_entries!(tab.stab_x, tab.stab_z, S, n, m, d)
    if tab.storephase
        d_phase = phase_modulus(d)
        @inbounds for j in 1:m
            tab.phase[j] = _reduced_entry(Int, S[2n + 1, j], d_phase)
        end
    end
    return tab
end

# A `StabilizerTableau` built from raw generators is only validated (with
# `check`); a `DestabilizerTableau` also rebuilds its duals.
_init_from_generators!(tab::StabilizerTableau, L::TableauLayout, check::Bool) =
    (check && _validate_generators!(tab, L); tab)

# `copy!` for both tableau types: a raw block copy of the stabilizers, then of
# the destabilizers of a `DestabilizerTableau`, so `dst` takes the layout of
# `src` without a transpose.
function _copy_tableau!(dst::AbstractTableau, src::AbstractTableau)
    _check_copy_compatible(dst, src)
    dst === src && return dst
    _copy_block!(dst.stab_x, src.stab_x)
    _copy_block!(dst.stab_z, src.stab_z)
    copyto!(dst.phase, src.phase)
    _copy_duals!(dst, src)
    copyto!(dst.pivcol_of_row, src.pivcol_of_row)
    copyto!(dst.xdotz_cache, src.xdotz_cache)
    dst.m = src.m
    dst.iscanonical = src.iscanonical
    dst.layout = src.layout
    return dst
end

_copy_duals!(::StabilizerTableau, ::StabilizerTableau) = nothing
function _copy_duals!(dst::DestabilizerTableau, src::DestabilizerTableau)
    _copy_block!(dst.destab_x, src.destab_x)
    _copy_block!(dst.destab_z, src.destab_z)
    return nothing
end

# The packed scratch of a packed tableau for `n` qudits: four columns of one
# word per 64 qudits, and at least the `2·_WORD_BITS` words a layout switch
# needs (`_transpose_bit_block!`).
_word_workspace(n::Int) = zeros(UInt64, max(_nwords(n), cld(2 * _WORD_BITS, 4)), 4)

##### canonical copies for `==` and `hash` #####

# `_canonical_stab` (src/tableau_equality.jl) for the layouts that have a
# canonicalization kernel of their own: the stabilizer blocks are copied, the
# kernel runs on the copies with a copy of the phases and fresh scratch, and
# the tableau itself (blocks, phases, `iscanonical`, pivot table, x·z cache,
# workspaces) is only read. Without `storephase` the tableau's own empty phase
# is returned, since nothing reads it.

# A packed tableau: canonicalized generator-major by the packed kernel, the
# copies of a qudit-major tableau transposed first.
_canonical_stab(tab::_BitTableau, L::GeneratorMajor) = _canonical_stab_bits(tab, L)
_canonical_stab(tab::_BitTableau, L::QuditMajor) = _canonical_stab_bits(tab, L)

function _canonical_stab_bits(tab::_BitTableau, L::TableauLayout)
    phase = _phase(tab)
    tab.iscanonical && return _stab(tab, L), phase
    n = tab.n
    x = _copy_bit_block(tab.stab_x, n)
    z = _copy_bit_block(tab.stab_z, n)
    if L isa QuditMajor
        buf = Vector{UInt64}(undef, 2 * _WORD_BITS)
        _transpose_bit_block!(x, buf)
        _transpose_bit_block!(z, buf)
    end
    S = BitBlock{GeneratorMajor}(x, z, n)
    Sphase = tab.storephase ? copy(phase) : phase
    _canonicalize_gm_bits!(S, tab.storephase ? Sphase.planes : nothing, nothing,
                           tab.m, Vector{Int}(undef, 2n))
    return S, Sphase
end

# An unpacked qudit-major tableau: canonicalized in its own layout by the
# qudit-major kernel, so that the canonical copies of two qudit-major tableaux
# are compared in storage order.
function _canonical_stab(tab::AbstractTableau, L::QuditMajor)
    phase = _phase(tab)
    tab.iscanonical && return _stab(tab, L), phase
    n = tab.n
    S = XZBlock{QuditMajor}(_copy_block(tab.stab_x), _copy_block(tab.stab_z))
    Sphase = tab.storephase ? copy(phase) : phase
    _canonicalize_qm!(S, Sphase, nothing, nothing, tab.m, tab.d, tab.storephase,
                      tab.inversemod, similar(tab.coeff_workspace, n, 2),
                      Vector{Int}(undef, n), Vector{Int}(undef, 2n), _QM_CANON_SPARSE_DIV)
    return S, Sphase
end
