##### some helper functions used in multiple files #####


#################################
# Phase-aware generator algebra #
#################################

# phase modulus:
# - odd prime d: ω^k with k mod d
# - qubits d=2: i^k with k mod 4   (needed for Clifford closure)
@inline phase_modulus(d::Int) = (d == 2 ? 4 : d)

# Every phase and symplectic dot product below accumulates `n` terms of size up
# to `(d-1)^2`, and the odd-`d` phase update sums two such terms, so the whole
# package is exact only while `max(n, 2) * (d-1)^2` fits in an `Int`. Past that
# the accumulators wrap and results are silently wrong -- `dot_xz_col` at n = 4,
# d = 3037000507 returns 581896576 where the answer is 4.
#
# Warn rather than throw: the bound is the worst case, every entry equal to
# `d-1`, and real tableaux are sparse enough that plenty of workloads past it
# still compute correctly. Refusing them would be wrong; saying nothing is
# worse.
#
# `maxlog=1` keeps a construction loop from drowning the session, but Julia keys
# that budget by the log message's `id`, which defaults to the call site -- one
# budget for every dimension there will ever be. The first unsafe tableau a
# session builds would then silence every later one, including a strictly worse
# `(d, n)`, which defeats the point of a warning whose whole subject is that
# the results are otherwise silently wrong. So the `id` carries `(d, n)`.
#
# The policy that buys is "every distinct `(d, n)` warns once", NOT "a worse
# one warns": a pair that is unsafe by less than one already reported still
# gets its own warning, because the comparison is identity, not severity.
# Accepted deliberately, with its cost stated rather than discovered:
#
#   - a sweep over `n` at one unsafe `d` warns once per `n`, so an `n = 1:10_000`
#     scaling study past the bound prints 10_000 warnings, not one;
#   - each distinct pair permanently interns a `Symbol` and adds an entry to the
#     logger's `message_limits`, neither of which is ever freed.
#
# Both are unbounded in principle. Neither is reachable without already being
# outside the range where this package returns correct numbers, which is the
# situation the warning exists to make loud -- a quieter policy would spend its
# one budget on whichever configuration happened to come first. If that trade
# ever needs revisiting, bucketing `n` (say by `floor(log2(n))`) keeps "a new
# order of magnitude speaks up" while bounding the id set.
#
# `isqrt` keeps the test itself from overflowing, as in `_barrett_ok`.
@inline max_safe_dimension(n::Int) = isqrt(typemax(Int) ÷ max(n, 2)) + 1

function _warn_if_dimension_unsafe(d::Int, n::Int)
    n == 0 && return nothing
    dmax = max_safe_dimension(n)
    d <= dmax && return nothing
    id = Symbol("qc_dimension_unsafe_", d, "_", n)
    @warn "Qudit dimension is large enough that phase and symplectic dot products \
           can overflow Int, giving silently incorrect results." d n max_safe_d = dmax _id = id maxlog = 1
    return nothing
end

# Helper: dot product x_i ⋅ z_j for a given column i and j, mod d.
# useful when updating phases while multiplying stabilizer generators
@inline function dot_xz_col(tab::AbstractMatrix{Int}, n::Int, i::Int, j::Int, d::Int)
    s = 0
    @turbo for q in 1:n
        s += tab[q, i] * tab[n+q, j]
    end
    return mod(s, d)
end
@inline dot_xz_col(tab::AbstractMatrix{Int}, n::Int, j::Int, d::Int) = dot_xz_col(tab, n, j, j, d)

# dot(x_src, z_tgt) where x_src from generator_workspace and z_tgt from tableau column
@inline function dot_xz_ws_vs_col(genws::Vector{Int}, tab::AbstractMatrix{Int}, n::Int, tgt::Int, d::Int)
    s = 0
    @turbo for q in 1:n
        s += genws[q] * tab[n+q, tgt]
    end
    return mod(s, d)
end

# dot(x_ws, z_ws) for a generator stored in workspace
@inline function dot_xz_ws(genws::Vector{Int}, n::Int, d::Int)
    s = 0
    @turbo for q in 1:n
        s += genws[q] * genws[n+q]
    end
    return mod(s, d)
end

# dot(x_col(j), zacc) mod d
@inline function dot_xz_col_vs_zacc(tab::AbstractMatrix{Int}, n::Int, j::Int, zacc::Vector{Int}, d::Int)
    s = 0
    @turbo for q in 1:n
        s += tab[q, j] * zacc[q]
    end
    return mod(s, d)
end
# Helper: C(t,2) mod d = t*(t-1)/2 mod d for odd prime d (needs inv2 = inv(2) mod d).
@inline function binom2_mod_oddprime(t::Int, d::Int, inv2::Int)
    return mod(mod(t * (t - 1), d) * inv2, d)
end
