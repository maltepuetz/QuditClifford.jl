##### overflow-safe scalar arithmetic for Clifford data #####
#
# `src/modular.jl`'s vector primitives are NOT a general fallback here. Both
# of their multiplying tiers -- Barrett and divide-twice -- form `a * src[i]`
# as an unguarded `Int` product, so they are exact only inside the package's
# documented envelope `max(n, 2)(d-1)^2 <= typemax(Int)`. (The multiply-free
# tiers are not the problem: the add-two-residues form is exact to
# `d <= 2^62`, far outside that envelope, and the subtract form always.)
#
# A Clifford dot sums `2k` terms rather than `n`, so it can leave that envelope
# even at n = 2: with A = [-1 -1; 0 -1] and the symplectic F = [A A; 0 A^-T] at
# d = 2147483647, the four-term first output coordinate is 4, while an
# unreduced `Int` dot wraps to 0 -- with no warning, because n = 2 is inside
# the envelope. Hence a separate, self-contained arithmetic tier here.

# All three take canonical residues `0 <= a, b < M` and return one.

@inline function add_mod(a::Int, b::Int, M::Int)
    # `M - b >= 1` since `b < M`, so the subtraction cannot overflow, and the
    # `a + b` branch is taken only when `a + b < M`.
    return a >= M - b ? a - (M - b) : a + b
end

@inline function sub_mod(a::Int, b::Int, M::Int)
    return a >= b ? a - b : M - (b - a)
end

@inline function mul_mod(a::Int, b::Int, M::Int)
    # `widemul` promotes BEFORE multiplying. `widen(a * b)` would widen an
    # already-overflowed value and is not equivalent.
    if M - 1 <= isqrt(typemax(Int))
        return mod(a * b, M)
    else
        return Int(mod(widemul(a, b), M))
    end
end

# Resolve the accumulation tier ONCE per preparation, never per column or per
# element. `S = 2k` is the longest dot in the subsystem; every product is
# bounded by `B^2`. `B` takes the phase modulus into account as well as `d`,
# because at d = 2 exponents are bits but phase coefficients reach 3.
# `T` is the accumulator type; `Int` everywhere except `_clifford_barrett`'s
# word-size-independent derivation.
@inline function clifford_fast_dots(S::Int, d::Int, ::Type{T} = Int) where {T<:Signed}
    S == 0 && return true
    B = max(d - 1, phase_modulus(d) - 1)
    return B <= isqrt(typemax(T) ÷ S)
end

# Barrett constant `(M, s)` for the tuple kernels, or `(0, 0)` when there is
# none: `div(x, d) == (x * M) >> s` and `x * M` fits in `Int` for every
# `0 <= x <= X`, with `X = 2S·d·(d-1)` the largest unreduced value those
# kernels form for a support of `S = 2k` coordinates (each call site states its
# own bound). This is the constant of `src/modular.jl`'s Barrett tier with the
# bound raised from `(d-1)^2` to `X`, and the shift chosen per modulus.
@inline clifford_barrett(S::Int, d::Int) = _clifford_barrett(Int, S, d)

# The derivation in a `W`-bit signed type `T`, so it holds for `Int` on both
# 32- and 64-bit hosts and can be checked for either on one machine. `M` is
# returned as a `T`, and `M == 0` means there is no constant.
#
# `clifford_fast_dots(4S, d, T)` bounds `4S(d-1)^2 >= X` by `typemax(T)`, so
# `X` is formed without overflow. The shift is the largest that keeps
# `X * M < 2^(W-2)`: with `X < 2^L`, `L <= W-2` and `2^lg <= d`,
# `M = cld(2^s, d) < 2^(s-lg) + 1` is an integer, so `M <= 2^(s-lg)` and
# `X * M < 2^(L+s-lg) = 2^(W-2)`. `lg < L` because `d <= X`, so `s <= W-3`,
# and `s >= 1` because `lg >= 1`. Exactness is the same criterion as
# `_barrett_ok`, `e * X < 2^s` for `e = M*d - 2^s`, tested by division so the
# product is never formed. Acceptance is NOT monotonic in `d`: a modulus whose
# `e` happens to be small passes where a smaller one fails, so never replace
# the test with a bound on `d`. Two divisions, once per preparation.
@inline function _clifford_barrett(::Type{T}, S::Int, d::Int) where {T<:Signed}
    W = 8 * sizeof(T)
    (S > 0 && 2 <= d <= typemax(T) && clifford_fast_dots(4S, d, T)) ||
        return (zero(T), 0)
    dt = T(d)
    X = T(2S) * dt * (dt - one(T))
    L = W - leading_zeros(X)
    L <= W - 2 || return (zero(T), 0)
    lg = W - 1 - leading_zeros(dt)
    s = W - 2 - L + lg
    p2 = one(T) << s
    M = cld(p2, dt)
    e = M * dt - p2
    (e == 0 || X <= (p2 - one(T)) ÷ e) || return (zero(T), 0)
    return (M, s)
end

# The narrowest lane for the qudit-major gate kernel, as its `sizeof`: 2, 4 or
# 8 for the first of `Int16`, `Int32`, `Int64` that has a constant from
# `_clifford_barrett(T, S, d)`, or 0 if none has one. Every unreduced value the
# kernel forms is at most `X = 2S·d·(d-1) < 2^(W-2)` (see `_clifford_barrett`),
# so it holds in that lane, and so does every entry, since `d <= X`. The lanes
# are named explicitly, never `Int`, so the choice is the same on 32- and 64-bit
# hosts. Acceptance is not monotonic in `d` (see `_clifford_barrett`), and
# neither is the lane.
@inline function _clifford_lane_bytes(S::Int, d::Int)
    _clifford_barrett(Int16, S, d)[1] != 0 && return 2
    _clifford_barrett(Int32, S, d)[1] != 0 && return 4
    _clifford_barrett(Int64, S, d)[1] != 0 && return 8
    return 0
end

# `x mod d` for `0 <= x <= X` under a constant from `_clifford_barrett(T, S, d)`,
# evaluated in `T`: `x * M < 2^(W-2)`, so nothing overflows. `1 <= s <= W-3`,
# so masking the shift to `W-1` changes nothing; it lets the shift compile
# without the range test of `>>`, which keeps the SIMD loops lean.
@inline _barrett_reduce(x::T, d::T, M::T, s::Int) where {T<:Signed} =
    x - ((x * M) >> ((s % UInt) & UInt(8 * sizeof(T) - 1))) * d
