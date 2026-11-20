############################################
# Channels on mixed stabilizer states      #
############################################

#############################################################
# Type-specific hook for removing a generator               #
#############################################################

@inline _on_generator_dropped!(::AbstractTableau, ::TableauLayout, ::Int, ::Int) = nothing

# The dual of the last generator takes the place of the dual of the removed one.
# Every other dual keeps its pairings: the generators it pairs with are unchanged
# or, after `dephase!`'s sweep, differ from the old ones only by multiples of the
# removed generator, which is orthogonal to all of them.
@inline function _on_generator_dropped!(tab::DestabilizerTableau, L::TableauLayout, j::Int, last::Int)
    dest = _destab(tab, L)
    if j != last
        _move_gen!(dest, j, last)
        tab.xdotz_cache[j] = tab.xdotz_cache[last]
    end
    _zero_gen!(dest, last)
    tab.xdotz_cache[last] = 0
    return nothing
end

# Remove active generator `j`: the last active column takes its place, the last
# column is zeroed and `m` shrinks by one.
function _drop_generator!(tab::AbstractTableau, L::TableauLayout, j::Int)
    stab = _stab(tab, L)
    last = tab.m
    _move_gen!(stab, j, last)
    _zero_gen!(stab, last)
    if tab.storephase
        phase = _phase(tab)
        phase[j] = phase[last]
        phase[last] = 0
    end
    _on_generator_dropped!(tab, L, j, last)
    tab.m = last - 1
    return nothing
end

"""
    dephase!(tab::AbstractTableau, op)

Measure the Pauli operator `op` and forget the outcome: apply the channel
`ρ → Σₖ Πₖ ρ Πₖ`, where `Πₖ` are the eigenprojectors of `op`, and return `tab`.

`op` is an `AbstractPauli` or an `AbstractVector{<:Integer}` of length `2n` or
`2n+1` (wrapped into a `GeneralPauli`). Its phase is ignored and its exponents
are reduced mod `d`; a nonzero power of `op` gives the same state as `op`. Pauli
operators acting outside `1:n`, or a dense operator of the wrong length, throw an
`ArgumentError` before `tab` is modified.

If `op` commutes with every generator, `tab` is unchanged, `m == 0` included.
Otherwise the state loses the generator that carries the randomness of the
measurement: `m` drops by one, the remaining generators are adjusted to commute
with `op`, and the generator basis is no longer canonical.

Unlike [`measure!`](@ref), nothing is sampled, so `dephase!` draws no random
numbers, adds no generator and needs no phase handling, which makes it valid for
`storephase=false` and non-Hermitian qubit operators alike. It allocates nothing
for an `AbstractPauli`; a raw exponent vector is wrapped in a `GeneralPauli`
first, as in `measure!`, which allocates a small wrapper.
The commutation pass costs `O(km)` for a `k`-qudit sparse Pauli, and removing a
generator adds `O(nm)`.

# Examples
```jldoctest
julia> tab = DestabilizerTableau(3, 2; state = :ghz);

julia> dephase!(tab, SinglePauli(1, 0, 1));   # Z on qudit 1: the GHZ state becomes classical

julia> tab.m
1
```
"""
function dephase!(tab::AbstractTableau, op::AbstractVector{<:Integer})
    return dephase!(tab, GeneralPauli(tab.n, tab.d, op))
end

dephase!(tab::AbstractTableau, op::AbstractPauli) = _dephase!(tab, _layout_marker(tab), op)

function _dephase!(tab::AbstractTableau, L::TableauLayout, op::AbstractPauli)
    _check_pauli_support(tab, op)
    d = tab.d
    m = tab.m

    # A dense operand lives in res_workspace, which nothing below writes.
    opc = _commutation_operand!(tab, op)

    stab = _stab(tab, L)
    pivot = 0
    comm0 = 0
    kws = 0
    for i in 1:m
        c = mod(commutation_col(stab, i, opc), d)
        c == 0 && continue
        if pivot == 0
            # The pivot is kept in pivot_workspace, its phase in `kws`, and it
            # is removed afterwards; the other noncommuting generators absorb a
            # power of it.
            _copy_gen!(tab.pivot_workspace, stab, i)
            kws = tab.storephase ? _phase(tab)[i] : 0
            pivot = i
            comm0 = c
        else
            a = mod(-c * tab.inversemod(comm0, d), d)
            mul_col_by_workspace_power!(tab, L, i, a, kws)
            _on_noncommuting_col_updated!(tab, L, i)
        end
    end

    pivot == 0 && return tab
    _drop_generator!(tab, L, pivot)
    tab.iscanonical = false
    return tab
end

"""
    traceout!(tab::AbstractTableau, sites)

Discard the qudits `sites` and return `tab`: the register keeps its `n` qudits,
the sites are left maximally mixed, and the state becomes `Tr_A ρ ⊗ I/d^|A|` for
`A = sites`. The stabilizer group shrinks to its subgroup supported on the
complement of `A`.

`sites` is a single qudit index or an `AbstractVector` of them (a range such as
`2:3` included). Every site must be an integer in `1:n` and no site may repeat;
otherwise an `ArgumentError` is thrown before `tab` is modified. An empty
collection leaves `tab` unchanged.

Tracing out qudit `q` equals dephasing by `X_q` and then by `Z_q`, which is how
it is computed, so it supports both tableau types, every prime `d` and
`storephase=false`, and allocates nothing. The cost is `O(|sites|·nm)`.

# Examples
```jldoctest
julia> tab = DestabilizerTableau(3, 3; state = :ghz);

julia> traceout!(tab, 1);               # qudit 1 becomes maximally mixed

julia> entanglement_entropy(tab, [1])
1
```
"""
function traceout!(tab::AbstractTableau, sites::AbstractVector)
    _check_subsystem!(tab, sites)
    return _traceout!(tab, _layout_marker(tab), sites)
end

function traceout!(tab::AbstractTableau, site::Integer)
    1 <= site <= tab.n || throw(ArgumentError(
        "The site must be in 1:$(tab.n); got $site."))
    return _traceout!(tab, _layout_marker(tab), (Int(site),))
end

# Trace out the checked `sites` by dephasing each by X and then by Z.
function _traceout!(tab::AbstractTableau, L::TableauLayout, sites)
    for q in sites
        _dephase!(tab, L, SinglePauli(Int(q), 1, 0))
        _dephase!(tab, L, SinglePauli(Int(q), 0, 1))
    end
    return tab
end

# The canonical basis symbol for the k-th site; `basis` is one symbol for all
# sites or a vector with one symbol per site.
@inline _reset_basis_at(basis::Symbol, ::Int) = _canonical_basis_symbol(basis)
@inline _reset_basis_at(basis::Union{AbstractVector,Tuple}, k::Int) =
    _canonical_basis_symbol(basis[k])

_check_reset_basis(basis::Symbol, ::Int) = (_canonical_basis_symbol(basis); nothing)
function _check_reset_basis(basis::Union{AbstractVector,Tuple}, nsites::Int)
    basis isa AbstractVector && Base.require_one_based_indexing(basis)
    length(basis) == nsites || throw(ArgumentError(
        "basis has length $(length(basis)) but there are $nsites sites."))
    for b in basis
        b isa Symbol || throw(ArgumentError("basis entries must be Symbols."))
        _canonical_basis_symbol(b)
    end
    return nothing
end
_check_reset_basis(::Any, ::Int) = throw(ArgumentError(
    "basis must be a Symbol, or a vector or tuple of Symbols."))

# Give each of `sites`, already traced out, a fresh generator of its target
# basis. No generator touches a traced-out site, so the new column index is at
# most n, and the dual built for it needs no projection.
function _prepare_sites!(tab::AbstractTableau, L::TableauLayout, sites, basis)
    isempty(sites) && return tab
    stab = _stab(tab, L)
    phase = _phase(tab)
    # The new generator is copied into res_workspace, which `_after_append_measurement!`
    # reads and never writes. Nothing else keeps data there: `reset!` has finished
    # its `traceout!`, which dephases with sparse operators that need no operand
    # vector, and a dense operand of `dephase!` is dead once that call returns.
    res = tab.res_workspace
    k = 0
    for q in sites
        k += 1
        newcol = tab.m + 1
        _set_site_column!(stab, phase, tab.d, newcol, Int(q),
                          _reset_basis_at(basis, k), tab.storephase)
        tab.m = newcol
        _copy_gen!(res, stab, newcol)
        _after_append_measurement!(tab, L, newcol, res)
    end
    tab.iscanonical = false
    return tab
end

"""
    reset!(tab::AbstractTableau, sites; basis=:Z)

Reset the qudits `sites` to a fixed eigenstate without recording an outcome, and
return `tab`: the channel traces out `sites` and prepares each one in the state
that `state=:product` gives it for the same basis, so the rest of the register
becomes mixed if it was entangled with `sites`.

`sites` is a single qudit index or an `AbstractVector` of them (a range such as
`2:3` included). `basis` is `:X`, `:Y` or `:Z` (lower case accepted), applied to
every site, or a vector or tuple of them with one entry per site. Sites outside
`1:n`, repeated sites, an unknown basis symbol, a `basis` collection whose length
differs from the number of sites and a `basis` vector with offset axes throw an `ArgumentError` before `tab` is modified. An
empty collection leaves `tab` unchanged.

Unlike [`measure!`](@ref) followed by a correction, no outcome is sampled or
returned, so `reset!` draws no random numbers and needs no phase handling beyond
the product preset's own. It supports both tableau types, every prime `d` and
`storephase=false`, and allocates nothing. The generator basis is no longer
canonical afterwards, and `tab.m` equals the rank after tracing out `sites` plus
`length(sites)`.

# Examples
```jldoctest
julia> tab = DestabilizerTableau(3, 3; state = :ghz);

julia> reset!(tab, 1);                         # qudit 1 is |0⟩, the other two are correlated but mixed

julia> expect!(tab, SinglePauli(1, 0, 1)), is_pure(tab)
(1.0 + 0.0im, false)

julia> reset!(tab, [2, 3]; basis = [:X, :Y]);  # per-site bases

julia> is_pure(tab)
true
```
"""
function reset!(tab::AbstractTableau, sites::AbstractVector; basis=:Z)
    _check_subsystem!(tab, sites)
    _check_reset_basis(basis, length(sites))
    return _reset_sites!(tab, _layout_marker(tab), sites, basis)
end

function reset!(tab::AbstractTableau, site::Integer; basis=:Z)
    1 <= site <= tab.n || throw(ArgumentError(
        "The site must be in 1:$(tab.n); got $site."))
    _check_reset_basis(basis, 1)
    return _reset_sites!(tab, _layout_marker(tab), (Int(site),), basis)
end

# Trace out the checked `sites` and prepare each in its target basis.
function _reset_sites!(tab::AbstractTableau, L::TableauLayout, sites, basis)
    _traceout!(tab, L, sites)
    return _prepare_sites!(tab, L, sites, basis)
end
