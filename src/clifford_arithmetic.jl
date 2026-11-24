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

# Barrett constant `(M, s)` of the unsigned type `U ∈ (UInt16, UInt32)` for the
# lane `T = signed(U)`, or `(0, 0)` when there is none: `div(x, d) ==
# (x * M) >> s` for every `0 <= x <= X = 2S·d·(d-1)`, with `M <= typemax(U)`
# and the product formed in `widen(U)`. `X <= typemax(T)` is required
# (`_lane_holds`), so with `W = 8·sizeof(U)`, `X < 2^L` for an `L <= W-1`, and
# `2^lg <= d < 2^(lg+1)` with `lg <= W-2`. The shift is `s = L + lg + 1`:
# `M = cld(2^s, d)` leaves `e = M·d - 2^s < d < 2^(lg+1)`, so
# `e·X < 2^(lg+1+L) = 2^s`, which is the exactness criterion of
# `_clifford_barrett`; and `M <= 2^(s-lg) = 2^(L+1) <= 2^W`, with equality only
# when `d` is a power of two. So every odd `d` whose `X` the lane holds has a
# constant, unlike the signed constants, and `x * M < 2^(2W-1)` fits
# `widen(U)`. Both conditions are still tested (exactness by division, as in
# `_clifford_barrett`), so a power of two is rejected or exact. `s <= 2W-2`
# fits the `UInt64` the derivation runs in on every host. One division pair,
# once per gate.
@inline function _clifford_barrett_unsigned(::Type{U}, S::Int,
                                            d::Int) where {U<:Union{UInt16,UInt32}}
    _lane_holds(signed(U), S, d) || return (zero(U), 0)
    W = 8 * sizeof(U)
    du = UInt64(d)
    X = UInt64(2S) * du * (du - one(UInt64))
    L = 8 * sizeof(UInt64) - leading_zeros(X)
    lg = 8 * sizeof(UInt64) - 1 - leading_zeros(du)
    s = L + lg + 1
    p2 = one(UInt64) << s
    M = cld(p2, du)
    M <= typemax(U) || return (zero(U), 0)
    e = M * du - p2
    (e == 0 || X <= (p2 - one(UInt64)) ÷ e) || return (zero(U), 0)
    return (M % U, s)
end

# The lane of the qudit-major gate kernel for a support of `S` coordinates at
# dimension `d`, as `(bytes, M, s, wide)`: `bytes` is the `sizeof` of the lane
# type `T`, or 0 when there is no lane; `(M, s)` is its Barrett constant as an
# `Int64`: the signed constant `_clifford_barrett(T, S, d)` when `wide` is
# false, and the unsigned constant `_clifford_barrett_unsigned(unsigned(T), S,
# d)`, whose product is formed in `widen(unsigned(T))`, when it is true. Every
# unreduced value the kernel forms is at most `X = 2S·d·(d-1)` (each site
# states its bound), and every entry is below `d <= X`.
#
# `Int16` and `Int32` lanes are chosen by that value bound: `T` is the narrowest
# with `X <= typemax(T)`. Its own signed constant is preferred; it exists only
# for small `d` (acceptance is not monotonic in `d`, so it is tested, never
# bounded), and reduces without leaving `T`. Otherwise the reduction multiplies
# two zero-extended `unsigned(T)` values in `widen(unsigned(T))`, a product
# NEON and AVX2 have a vector instruction for (neither has one for a full
# 64-bit product). That constant
# exists for every odd `d` the lane holds, so no odd `d` falls through to a
# wider lane. An `Int64` lane reduces in `Int64` itself, under
# `_clifford_barrett(Int64, S, d)`. The lanes are named explicitly, never
# `Int`, so a 32-bit host chooses the same lane as a 64-bit one.
#
# Measured per gate on one to three qudits (Stab, n = 2048, M3 Pro): where a
# lane has both constants (d = 3, 5, 61, 71, 101), the signed one takes
# 0.75-0.81 of the unsigned one's time, so it comes first. With the unsigned
# constant a two-qudit gate takes about 1650 ns in the Int16 lane (d = 31)
# and 2800 ns in the Int32 lane (d = 127, 251), against 2290 ns in the Int32
# lane with its own signed constant (d = 71, 101), so the narrower lane comes
# before the wider one's signed constant.
@inline function _clifford_lane(S::Int, d::Int)
    if _lane_holds(Int16, S, d)
        M, s = _clifford_barrett(Int16, S, d)
        M != 0 && return (2, Int64(M), s, false)
        Mu, su = _clifford_barrett_unsigned(UInt16, S, d)
        Mu != 0 && return (2, Int64(Mu), su, true)
    end
    if _lane_holds(Int32, S, d)
        M, s = _clifford_barrett(Int32, S, d)
        M != 0 && return (4, Int64(M), s, false)
        Mu, su = _clifford_barrett_unsigned(UInt32, S, d)
        Mu != 0 && return (4, Int64(Mu), su, true)
    end
    M, s = _clifford_barrett(Int64, S, d)
    M != 0 && return (8, M, s, false)
    return (0, Int64(0), 0, false)
end

@inline _clifford_lane_bytes(S::Int, d::Int) = _clifford_lane(S, d)[1]

# `X = 2S·d·(d-1) <= typemax(T)` for `T ∈ (Int16, Int32)`. With `S` and `d`
# at most `typemax(T) < 2^31`, `X < 2^94` is formed in `Int128` without
# overflow on any host.
@inline function _lane_holds(::Type{T}, S::Int, d::Int) where {T<:Union{Int16,Int32}}
    (1 <= S <= typemax(T) && 2 <= d <= typemax(T)) || return false
    return Int128(2S) * d * (d - 1) <= typemax(T)
end

# `x mod d` for `0 <= x <= X` under a constant `(M, s)` from
# `_clifford_barrett(W, S, d)`, with `x` and `d` in a lane type `T` no wider
# than `W`: `x * M < 2^(W-2)` is formed in `W` and the quotient is exact, so
# it and the result fit `T`. With `W = T = Int` this is the reduction of the
# tuple kernels. `1 <= s <= W-3`, so masking the shift to `W-1` changes
# nothing; it lets the shift compile without the range test of `>>`, which
# keeps the SIMD loops lean.
@inline function _barrett_reduce(x::T, d::T, M::W, s::Int) where {T<:Signed,W<:Signed}
    q = ((x % W) * M) >> ((s % UInt) & UInt(8 * sizeof(W) - 1))
    return x - (q % T) * d
end

# `x mod d` for `0 <= x <= X` under a constant `(M, s)` from
# `_clifford_barrett_unsigned(U, S, d)`, with `x` and `d` in the lane type
# `T = signed(U)` of `W` bits: `0 <= x <= X <= typemax(T)`, so `x % U` is `x`,
# and `x * M < 2^(W-1) · 2^W` is formed in `widen(U)` from two zero-extended
# `U` values. The quotient is exact and at most `x ÷ d`, so it and the result
# fit `T`. `1 <= s <= 2W-2`, so masking the shift to `2W-1` changes nothing.
@inline function _barrett_reduce(x::T, d::T, M::U, s::Int) where {T<:Signed,U<:Unsigned}
    V = widen(U)
    q = ((x % U % V) * (M % V)) >> ((s % UInt) & UInt(8 * sizeof(V) - 1))
    return x - (q % T) * d
end
