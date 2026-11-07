############################################
# Channels on mixed stabilizer states      #
############################################

# Throw an `ArgumentError` unless `op` acts inside the register of `tab`. The
# check is O(k) for a k-qudit sparse Pauli and does not allocate.
function _check_pauli_support(tab::AbstractTableau, op::_BuiltinFewQuditPauli)
    qudits = _few_qudit_pauli_data(op)[1]
    for q in qudits
        1 <= q <= tab.n || throw(ArgumentError(
            "Pauli acts on qudit $q, outside the register of n = $(tab.n) qudits."))
    end
    return nothing
end

function _check_pauli_support(tab::AbstractTableau, op::GeneralPauli)
    length(op.xz) == 2 * tab.n || throw(ArgumentError(
        "Pauli vector has length $(length(op.xz)), expected 2n = $(2 * tab.n)."))
    return nothing
end

_check_pauli_support(::AbstractTableau, ::AbstractPauli) = nothing

#############################################################
# Type-specific hook for removing a generator               #
#############################################################

@inline _on_generator_dropped!(::AbstractTableau, ::Int, ::Int) = nothing

# The dual of the last generator takes the place of the dual of the removed one.
# Every other dual keeps its pairings: the generators it pairs with are unchanged
# or, after `dephase!`'s sweep, differ from the old ones only by multiples of the
# removed generator, which is orthogonal to all of them.
@inline function _on_generator_dropped!(tab::DestabilizerTableau, j::Int, last::Int)
    dest = tab.destab
    if j != last
        # j != last, so these are disjoint columns of destab.
        @turbo for r in 1:(2 * tab.n)
            dest[r, j] = dest[r, last]
        end
        tab.xdotz_cache[j] = tab.xdotz_cache[last]
    end
    @turbo for r in 1:(2 * tab.n)
        dest[r, last] = 0
    end
    tab.xdotz_cache[last] = 0
    return nothing
end

# Remove active generator `j`: the last active column takes its place, the last
# column is zeroed and `m` shrinks by one.
function _drop_generator!(tab::AbstractTableau, j::Int)
    stab = tab.stab
    last = tab.m
    rows = size(stab, 1)
    if j != last
        # j != last, so these are disjoint columns of stab.
        @turbo for r in 1:rows
            stab[r, j] = stab[r, last]
        end
    end
    @turbo for r in 1:rows
        stab[r, last] = 0
    end
    _on_generator_dropped!(tab, j, last)
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
`storephase=false` and non-Hermitian qubit operators alike. It allocates nothing.
The commutation pass costs `O(km)` for a `k`-qudit sparse Pauli, and removing a
generator adds `O(nm)`.

# Examples
```julia
tab = DestabilizerTableau(3, 2; state=:ghz)
dephase!(tab, SinglePauli(1, 0, 1))   # Z on qudit 1: the GHZ state becomes classical
tab.m                                  # 1
```
"""
function dephase!(tab::AbstractTableau, op::AbstractVector{<:Integer})
    return dephase!(tab, GeneralPauli(tab.n, tab.d, op))
end

function dephase!(tab::AbstractTableau, op::AbstractPauli)
    _check_pauli_support(tab, op)
    d = tab.d
    m = tab.m

    # A dense operand lives in res_workspace, which nothing below writes.
    opc = _commutation_operand!(tab, op)

    pivot = 0
    comm0 = 0
    for i in 1:m
        c = mod(commutation_col(tab.stab, i, opc), d)
        c == 0 && continue
        if pivot == 0
            # The pivot is kept in the workspace and removed afterwards; the
            # other noncommuting generators absorb a power of it.
            @turbo for j in eachindex(tab.generator_workspace)
                tab.generator_workspace[j] = tab.stab[j, i]
            end
            pivot = i
            comm0 = c
        else
            a = mod(-c * tab.inversemod(comm0, d), d)
            mul_col_by_workspace_power!(tab, i, a)
            _on_noncommuting_col_updated!(tab, i)
        end
    end

    pivot == 0 && return tab
    _drop_generator!(tab, pivot)
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
```julia
tab = DestabilizerTableau(3, 3; state=:ghz)
traceout!(tab, 1)                # qudit 1 becomes maximally mixed
entanglement_entropy(tab, [1])   # 1
```
"""
function traceout!(tab::AbstractTableau, sites::AbstractVector)
    _check_subsystem!(tab, sites)
    for q in sites
        dephase!(tab, SinglePauli(Int(q), 1, 0))
        dephase!(tab, SinglePauli(Int(q), 0, 1))
    end
    return tab
end

function traceout!(tab::AbstractTableau, site::Integer)
    1 <= site <= tab.n || throw(ArgumentError(
        "The site must be in 1:$(tab.n); got $site."))
    dephase!(tab, SinglePauli(Int(site), 1, 0))
    dephase!(tab, SinglePauli(Int(site), 0, 1))
    return tab
end
