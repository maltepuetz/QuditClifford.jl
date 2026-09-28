##### Dense and exact state representations #####
#
# `ket`, `state_vector` and `density_matrix` read a tableau and never write to
# it: they canonicalize a private copy, allocate their own scratch, and never
# borrow the shared workspaces. Hence no `!`, and no DestabilizerTableau hook --
# there is no `destab` duality or `xdotz_cache` to keep in step.
#
# `maxentries` bounds the scalar storage a call returns, never peak memory, and
# every bound is tested without forming a value that could overflow.
#
# No `@inbounds` anywhere in this file. Every write here goes through an index
# computed from loop state -- a basis label, a carried row, a term counter --
# and under `@inbounds` a slip in that arithmetic corrupts memory instead of
# raising. Bounds checks cost 11-31% on the density-matrix inner loop at the
# default size limit (32 ms on a 4096 × 4096 qubit matrix), which is nothing
# next to using a 256 MiB result. The lone unchecked loop is the `@turbo` dot
# product in state_expansion.jl, which indexes only by its loop variable.

const DEFAULT_MAX_ENTRIES = 1 << 24

# Key for the one inner constructor. Unexported: a (labels, phases) pair with
# the right shape need not be a stabilizer state at all (a minus sign on |111⟩
# alone in a uniform 3-qubit superposition gives CCZ|+++⟩), so construction is
# reserved for the paths below, which derive the data from a valid tableau.
struct _TrustedKet end

"""
    StabilizerKet

Exact expansion of a pure stabilizer state in the computational basis, as
returned by [`ket`](@ref).

A stabilizer state with `T = d^k` nonzero amplitudes is

```math
|\\psi\\rangle = \\frac{1}{\\sqrt{T}} \\sum_{t=1}^{T} \\zeta^{\\,\\mathrm{phases}[t]}\\, |\\mathrm{labels}[:, t]\\rangle,
```

with ``\\zeta = i`` for `d = 2` and ``\\zeta = \\omega = e^{2\\pi i/d}`` for odd
prime `d`. Every amplitude has the same modulus and a phase that is an integer
power of ``\\zeta``, so the expansion is exact.

# Fields
- `d::Int`: qudit dimension.
- `n::Int`: number of qudits.
- `labels::Matrix{Int}`: `n × T` digits; column `t` is the basis label of term
  `t`, qudit 1 in row 1.
- `phases::Vector{Int}`: the `T` exponents of ``\\zeta``, reduced modulo
  `phase_modulus(d)`.

# Canonical form
Labels are strictly increasing in lexicographic order and `phases[1] == 0`, so
the global phase is fixed by making the first amplitude real and positive. Two
kets of the same state are therefore `==`, and hash equal, whichever generators
they were computed from.

# Notes
- Labels are digits rather than linear indices, so a ket stays representable
  when `d^n` does not fit in an `Int`: a 100-qutrit GHZ ket is 300 digits and
  three phases. This is an exact support expansion, not a compressed form —
  `|+⟩^{⊗n}` still has `d^n` terms.
- A ket owns its arrays, but Julia permits mutating arrays held by an immutable
  struct. Treat `labels` and `phases` as read-only: mutating them breaks the
  canonical form and any `Dict` holding the ket as a key.
- There is no public constructor from raw data, because a correctly shaped
  `(labels, phases)` pair need not be a stabilizer state.

See also [`state_vector`](@ref), [`density_matrix`](@ref).
"""
struct StabilizerKet
    d::Int
    n::Int
    labels::Matrix{Int}
    phases::Vector{Int}

    function StabilizerKet(::_TrustedKet, d::Int, n::Int,
                           labels::Matrix{Int}, phases::Vector{Int})
        @assert size(labels) == (n, length(phases))
        @assert !isempty(phases) && phases[1] == 0
        return new(d, n, labels, phases)
    end
end

Base.:(==)(a::StabilizerKet, b::StabilizerKet) =
    a.d == b.d && a.n == b.n && a.labels == b.labels && a.phases == b.phases

# Integer contents only, so `isequal` and `==` coincide.
Base.isequal(a::StabilizerKet, b::StabilizerKet) = a == b

# Not cached: the arrays are read-only by contract only, and a cached hash
# would outlive a violation of it.
Base.hash(k::StabilizerKet, h::UInt) =
    hash(k.phases, hash(k.labels, hash(k.n, hash(k.d, hash(:StabilizerKet, h)))))

################
# Input checks #
################

function _check_budget(maxentries::Int)
    maxentries > 0 && return nothing
    throw(ArgumentError("maxentries must be positive, got $maxentries."))
end

# Cheap contract checks shared by all three conversions, in the order they run.
function _check_convertible(tab::AbstractTableau, fname::String, needs_pure::Bool)
    tab.storephase || throw(ArgumentError(
        "$fname needs a tableau with storephase=true: without stored phases the " *
        "tableau does not determine a state. Track phases from state preparation " *
        "onward; adding zero phases afterwards does not recover the lost state."))
    if needs_pure && !is_pure(tab)
        throw(ArgumentError(
            "$fname needs a pure state (m == n); got m = $(tab.m), n = $(tab.n). " *
            "Use density_matrix for mixed states."))
    end
    n, d = tab.n, tab.d
    if n > 0 && d > max_safe_dimension(n)
        throw(ArgumentError(
            "$fname needs exact phase arithmetic, which Int supports at n = $n " *
            "qudits only for d <= $(max_safe_dimension(n)); got d = $d."))
    end
    d == 2 && _check_qubit_hermitian(tab, fname)
    return nothing
end

# A qubit generator is an involution only if Hermitian, a ≡ x·z (mod 2);
# otherwise P^2 = -I and the exponent sweep 0:1 does not walk a group. Checked
# on the caller's original columns, so the reported index is theirs, and
# computed locally rather than read from `xdotz_cache`.
function _check_qubit_hermitian(tab::AbstractTableau, fname::String)
    n, stab = tab.n, tab.stab
    for j in 1:tab.m
        xz = dot_xz_col(stab, n, j, 2)
        mod(stab[2n+1, j], 2) == xz && continue
        throw(ArgumentError(
            "$fname needs Hermitian qubit generators, but active column $j has " *
            "phase exponent $(stab[2n+1, j]) while x·z = $xz (mod 2), so it squares " *
            "to -I. measure! with phase_policy=2 or a raw-matrix constructor can " *
            "produce such a generator; phase_policy=1 repairs it at measurement."))
    end
    return nothing
end

_size_error(fname::String, what::String, maxentries::Int) = ArgumentError(
    "$fname would store $what entries, which exceeds maxentries = $maxentries. " *
    "Pass a larger maxentries to allow it.")

# The byte count must be an Int too, even when the caller passes
# maxentries = typemax(Int); a larger budget never overrides this.
function _check_bytes(count::Int, elsize::Int, fname::String)
    count <= typemax(Int) ÷ elsize && return nothing
    throw(ArgumentError(
        "$fname cannot allocate $count entries of $elsize bytes: the byte count " *
        "does not fit in an Int."))
end

#######
# ket #
#######

"""
    ket(tab::AbstractTableau; maxentries::Int=QuditClifford.DEFAULT_MAX_ENTRIES) -> StabilizerKet

Return the exact computational-basis expansion of the pure state `tab`
represents.

# Arguments
- `tab::AbstractTableau`: A pure tableau (`m == n`) with `storephase=true`.
  Not modified.

# Keyword Arguments
- `maxentries::Int`: Largest number of integers the result may store, counting
  `n` label digits and one phase per term, `(n+1)·d^k` in all. The default is
  `2^24`.

# Returns
- [`StabilizerKet`](@ref) in canonical form: labels in ascending lexicographic
  order, first amplitude real and positive.

# Examples
```julia
tab = StabilizerTableau(3, 100; state=:ghz)
ket(tab)          # (|00…0⟩ + |11…1⟩ + |22…2⟩)/√3, three terms at any n
```

# Notes
- Throws `ArgumentError` for a mixed tableau, for `storephase=false`, for a
  non-Hermitian qubit generator (naming its column), for `d` beyond the exact
  `Int` phase arithmetic at this `n`, or when the result exceeds `maxentries`.
- Costs `O(n^3 + (n+1) d^k)`; the cubic term is canonicalizing a private copy.
- Works on the stabilizer group, so it needs no destabilizers and reads no
  workspaces; `DestabilizerTableau` and `StabilizerTableau` give identical kets.
"""
function ket(tab::AbstractTableau; maxentries::Int=DEFAULT_MAX_ENTRIES)
    _check_budget(maxentries)
    _check_convertible(tab, "ket", true)
    n, d = tab.n, tab.d
    # n + 1 cannot overflow: the tableau already holds (2n + 1) × n entries.
    per_term = n + 1
    # Zero qudits: one empty label with phase 0. Bypasses the vector kernels,
    # which are not exercised on empty ranges.
    n == 0 && return StabilizerKet(_TrustedKet(), d, 0, zeros(Int, 0, 1), [0])

    G, k, cstar = _prepare_support(tab)
    limit = maxentries ÷ per_term
    T = limit == 0 ? -1 : _bounded_pow(d, k, limit)
    T < 0 && throw(_size_error("ket", "($n+1)·$d^$k", maxentries))
    _check_bytes(per_term * T, sizeof(Int), "ket")

    labels = Matrix{Int}(undef, n, T)
    phases = Vector{Int}(undef, T)
    t = Ref(0)
    _emit_support(G, n, k, d, cstar) do label, phase
        i = (t[] += 1)
        for q in 1:n
            labels[q, i] = label[q]
        end
        phases[i] = phase
    end
    return StabilizerKet(_TrustedKet(), d, n, labels, phases)
end
