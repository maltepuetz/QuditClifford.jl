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

####################
# Basis and phases #
####################

# Qudit 1 is most significant: index(c) = 1 + Σ_q c_q d^(n-q). Only called
# after a guard has established d^n <= maxentries, so no stride overflows.
function _kron_strides(d::Int, n::Int)
    strides = Vector{Int}(undef, n)
    acc = 1
    for q in n:-1:1
        strides[q] = acc
        q > 1 && (acc *= d)
    end
    return strides
end

@inline function _kron_index(label::AbstractVector{Int}, strides::Vector{Int})
    idx = 1
    for q in eachindex(strides)
        idx += label[q] * strides[q]
    end
    return idx
end

# ζ^e for a reduced exponent. Exact for qubits; for odd d, the conversion
# `_expect_from_exponent` uses. Evaluated per call rather than tabulated, so no
# buffer ever scales with an unrestricted d.
@inline function _zeta_power(e::Int, d::Int)
    if d == 2
        return e == 0 ? complex(1.0, 0.0) :
               e == 1 ? complex(0.0, 1.0) :
               e == 2 ? complex(-1.0, 0.0) : complex(0.0, -1.0)
    end
    return cis(2π * (Float64(e) / Float64(d)))
end

# The one scatter both state_vector methods share.
@inline function _scatter_term!(psi::Vector{ComplexF64}, label::AbstractVector{Int},
                                phase::Int, strides::Vector{Int}, d::Int, amp::Float64)
    psi[_kron_index(label, strides)] = _zeta_power(phase, d) * amp
    return nothing
end

################
# state_vector #
################

"""
    state_vector(tab::AbstractTableau; maxentries::Int=QuditClifford.DEFAULT_MAX_ENTRIES) -> Vector{ComplexF64}
    state_vector(k::StabilizerKet; maxentries::Int=QuditClifford.DEFAULT_MAX_ENTRIES) -> Vector{ComplexF64}

Return the dense state vector, of length `d^n`, of a pure tableau or a
[`StabilizerKet`](@ref).

Qudit 1 is the most significant digit, so basis label `(c₁, …, cₙ)` sits at
index `1 + Σ_q c_q d^(n-q)` and product states agree with `kron`. The global
phase is the one [`ket`](@ref) fixes: the first nonzero entry is real and
positive.

# Arguments
- `tab::AbstractTableau`: A pure tableau (`m == n`) with `storephase=true`.
  Not modified.
- `k::StabilizerKet`: An exact ket.

# Keyword Arguments
- `maxentries::Int`: Largest number of complex entries the result may hold,
  here `d^n`. The default is `2^24`, 256 MiB of `ComplexF64`.

# Returns
- `Vector{ComplexF64}` of length `d^n`.

# Examples
```julia
tab = StabilizerTableau(2, 2; state=:product, basis=[:Z, :X])
state_vector(tab)       # |0⟩ ⊗ |+⟩ = [1, 1, 0, 0] / √2
```

# Notes
- The tableau method checks `d^n` against `maxentries` before canonicalizing,
  and scatters the support directly: it never builds a `StabilizerKet`, so its
  budget is `d^n` alone even when the exact ket would need more.
- Throws `ArgumentError` under the same conditions as [`ket`](@ref), with
  `d^n` as the size.
"""
function state_vector(tab::AbstractTableau; maxentries::Int=DEFAULT_MAX_ENTRIES)
    _check_budget(maxentries)
    _check_convertible(tab, "state_vector", true)
    n, d = tab.n, tab.d
    D = _bounded_pow(d, n, maxentries)
    D < 0 && throw(_size_error("state_vector", "$d^$n", maxentries))
    _check_bytes(D, sizeof(ComplexF64), "state_vector")
    n == 0 && return ComplexF64[1]

    G, k, cstar = _prepare_support(tab)
    psi = zeros(ComplexF64, D)
    strides = _kron_strides(d, n)
    amp = 1 / sqrt(Float64(_bounded_pow(d, k, D)))   # d^k <= d^n = D
    _emit_support(G, n, k, d, cstar) do label, phase
        _scatter_term!(psi, label, phase, strides, d, amp)
    end
    return psi
end

function state_vector(k::StabilizerKet; maxentries::Int=DEFAULT_MAX_ENTRIES)
    _check_budget(maxentries)
    n, d = k.n, k.d
    D = _bounded_pow(d, n, maxentries)
    D < 0 && throw(_size_error("state_vector", "$d^$n", maxentries))
    _check_bytes(D, sizeof(ComplexF64), "state_vector")

    psi = zeros(ComplexF64, D)
    strides = _kron_strides(d, n)
    T = length(k.phases)
    amp = 1 / sqrt(Float64(T))
    for t in 1:T
        _scatter_term!(psi, view(k.labels, :, t), k.phases[t], strides, d, amp)
    end
    return psi
end

##################
# density_matrix #
##################

"""
    density_matrix(tab::AbstractTableau; maxentries::Int=QuditClifford.DEFAULT_MAX_ENTRIES) -> Matrix{ComplexF64}

Return the dense `d^n × d^n` density matrix of the pure or mixed state `tab`
represents.

For the stabilizer group ``S`` generated by the `m` active columns,

```math
\\rho = \\frac{1}{d^n} \\sum_{g \\in S} g ,
```

so ``\\mathrm{tr}\\,\\rho = 1``, ``\\rho`` has `d^(n-m)` eigenvalues equal to
``1/d^{\\,n-m}`` and the rest zero, and ``\\rho^2 = \\rho`` exactly when the
state is pure. Basis ordering is the one [`state_vector`](@ref) uses.

# Arguments
- `tab::AbstractTableau`: Any tableau with `storephase=true`, pure or mixed,
  including the maximally mixed `m = 0`. Not modified.

# Keyword Arguments
- `maxentries::Int`: Largest number of complex entries the result may hold,
  here `d^(2n)`. The default is `2^24`, which is 12 qubits.

# Returns
- `Matrix{ComplexF64}` of size `d^n × d^n`.

# Examples
```julia
tab = StabilizerTableau(2, 2; state=:ghz)
density_matrix(tab)     # (|00⟩ + |11⟩)(⟨00| + ⟨11|) / 2
```

# Notes
- Sums the `d^m` group elements of the original generators directly, without
  canonicalizing: `Θ(d^(2n))` in all, dominated by zeroing the output even when
  the state is maximally mixed.
- Entries carry floating-point round-off, and cancellation can leave small
  residuals where the exact entry is zero; compare with an absolute tolerance.
- Throws `ArgumentError` under the same conditions as [`ket`](@ref) except
  purity, with `d^(2n)` as the size.
"""
function density_matrix(tab::AbstractTableau; maxentries::Int=DEFAULT_MAX_ENTRIES)
    _check_budget(maxentries)
    _check_convertible(tab, "density_matrix", false)
    n, d, m = tab.n, tab.d, tab.m
    D = _bounded_pow(d, n, maxentries)
    # D * D is formed only after D <= maxentries ÷ D, so it cannot overflow.
    (D < 0 || D > maxentries ÷ D) &&
        throw(_size_error("density_matrix", "$d^$n × $d^$n", maxentries))
    _check_bytes(D * D, sizeof(ComplexF64), "density_matrix")

    rho = zeros(ComplexF64, D, D)
    if n == 0
        rho[1, 1] = 1
        return rho
    end

    p = phase_modulus(d)
    s = p ÷ d
    strides = _kron_strides(d, n)
    c = zeros(Int, n)
    r = zeros(Int, n)
    xz = zeros(Int, 2n)
    u = zeros(Int, m)
    invD = 1 / D
    _walk_group(tab.stab, n, m, d, xz, u) do xz, a
        _add_pauli!(rho, xz, a, n, d, p, s, strides, c, r, invD)
    end
    return rho
end

# rho += ζ^a X^x Z^z / D for the Pauli P(xz[1:n], xz[n+1:2n], a).
#
# Column `col` is basis label c in kron order and ⟨c + x| P |c⟩ = ζ^(a + s z·c),
# so one pass over the D columns places every nonzero. A base-d counter walks c,
# last qudit fastest; along it the column index rises by exactly one, and the
# row index and the phase exponent are carried rather than recomputed, which
# keeps the pass amortized O(1) per entry instead of O(n).
function _add_pauli!(rho::Matrix{ComplexF64}, xz::Vector{Int}, a::Int,
                     n::Int, d::Int, p::Int, s::Int, strides::Vector{Int},
                     c::Vector{Int}, r::Vector{Int}, invD::Float64)
    D = size(rho, 1)
    fill!(c, 0)
    row = 1
    for q in 1:n
        r[q] = xz[q]                      # r = c + x with c = 0
        row += r[q] * strides[q]
    end
    e = a
    col = 1
    while true
        rho[row, col] += _zeta_power(e, d) * invD
        col == D && return nothing        # c is all d-1; there is no next column
        col += 1
        q = n
        while true
            # Advance digit q of c by one; the row digit r[q] = c[q] + x[q]
            # moves with it. The row index changes by the CHANGE in r[q] times
            # its stride, not by its new value: r[q] wraps from d-1 to 0 where
            # c[q] does not, offset by x[q], so this wrap is not synchronized
            # with the carry that triggered the step. At d = 3 with x[q] = 1,
            # c[q] going 1 -> 2 is no carry, yet r[q] goes 2 -> 0 and the row
            # moves by -2 strides, not +1.
            rq = r[q] + 1
            rq == d && (rq = 0)
            row += (rq - r[q]) * strides[q]
            r[q] = rq
            # The phase needs no such care. z·c rises by z[q], so e rises by
            # s z[q]; when c[q] wraps, the true change is -(d-1) z[q], and the
            # difference d s z[q] = p z[q] ≡ 0 (mod p). Since e < p and
            # s z[q] < p, one conditional subtract reduces it.
            e += s * xz[n+q]
            e >= p && (e -= p)
            c[q] += 1
            c[q] < d && break
            c[q] = 0
            q -= 1                        # carry; col < D keeps q >= 1
        end
    end
end

###########
# Display #
###########

# Display caps come from IOContext properties rather than a global `Ref` like
# `max_qudits_display`. That is a deliberate departure: a property is scoped to
# one `show` call, composes with the caller's own IOContext, and leaves no
# mutable global for concurrent callers to race on. `max_qudits_display` is
# left as it is.
function _ket_display_cap(io::IO, key::Symbol, default::Int)
    v = get(io, key, default)
    (v isa Integer && v > 0) || throw(ArgumentError(
        "IOContext property :$key must be a positive integer, got $(repr(v))."))
    return Int(v)
end

# |c₁c₂…cₙ⟩, digits joined for d <= 10 and comma separated above that, with the
# middle elided past `maxlabel` coordinates. Counting coordinates rather than
# characters never splits a multi-digit coordinate.
function _show_ket_label(io::IO, k::StabilizerKet, t::Int, maxlabel::Int)
    n = k.n
    sep = k.d > 10 ? "," : ""
    print(io, '|')
    if n <= maxlabel
        for q in 1:n
            q > 1 && print(io, sep)
            print(io, k.labels[q, t])
        end
    else
        head = cld(maxlabel, 2)
        tail = maxlabel - head
        for q in 1:head
            q > 1 && print(io, sep)
            print(io, k.labels[q, t])
        end
        print(io, sep, '…')
        for q in (n-tail+1):n
            print(io, sep, k.labels[q, t])
        end
    end
    print(io, '⟩')
    return nothing
end

# One term with its leading separator; the first term is bare, since its phase
# is 0 by the canonical form. Qubits use signs and i; odd primes use ω_d with
# the dimension as a subscript, so an expression printed without its header
# still names its root.
function _show_ket_term(io::IO, k::StabilizerKet, t::Int, maxlabel::Int)
    if t > 1
        e = k.phases[t]
        if k.d == 2
            print(io, e >= 2 ? " − " : " + ")
            isodd(e) && print(io, 'i')
        else
            print(io, " + ")
            if e != 0
                print(io, 'ω')
                _show_script_integer(io, k.d, _PAULI_SUBSCRIPT_DIGITS, '₋')
                e != 1 && _show_script_integer(io, e, _PAULI_SUPERSCRIPT_DIGITS, '⁻')
            end
        end
    end
    _show_ket_label(io, k, t, maxlabel)
    return nothing
end

# The expression alone: `|c⟩` for one term, otherwise `(t₁ + t₂ + …)/√T`. At
# most `:max_ket_terms` terms are printed, fewer when `:limit` is set and the
# line would pass the display width less `indent`; the full count T always
# stays in the divisor. Only printed terms are formatted, each bounded by the
# label cap.
function _show_ket_expression(io::IO, k::StabilizerKet, indent::Int)
    T = length(k.phases)
    maxterms = _ket_display_cap(io, :max_ket_terms, 16)
    maxlabel = _ket_display_cap(io, :max_ket_label, 32)
    if T == 1
        _show_ket_label(io, k, 1, maxlabel)
        return nothing
    end
    budget = get(io, :limit, false) === true ? displaysize(io)[2] - indent : typemax(Int)
    tail = string(")/√", T)
    ellipsis = " + …"
    print(io, '(')
    width = 1 + textwidth(tail)
    shown = 0
    for t in 1:min(T, maxterms)
        term = sprint(_show_ket_term, k, t, maxlabel; context=io)
        w = textwidth(term)
        # Always print the first term. A later one must fit together with the
        # ellipsis that follows it when terms remain; the last term needs only
        # its own width.
        more = t < T
        t > 1 && width + w + (more ? textwidth(ellipsis) : 0) > budget && break
        print(io, term)
        width += w
        shown += 1
    end
    shown < T && print(io, ellipsis)
    print(io, tail)
    return nothing
end

Base.show(io::IO, k::StabilizerKet) = _show_ket_expression(io, k, 0)

function Base.show(io::IO, ::MIME"text/plain", k::StabilizerKet)
    T = length(k.phases)
    print(io, "StabilizerKet (d = ", k.d, ", n = ", k.n, ", ", T,
          T == 1 ? " term" : " terms", "):\n  ")
    _show_ket_expression(io, k, 2)
    return nothing
end
