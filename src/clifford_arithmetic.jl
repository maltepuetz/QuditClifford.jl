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
@inline function clifford_fast_dots(S::Int, d::Int)
    S == 0 && return true
    B = max(d - 1, phase_modulus(d) - 1)
    return B <= isqrt(typemax(Int) ÷ S)
end

# Barrett constant `(M, s)` for the tuple kernels, or `(0, 0)` when there is
# none: `div(x, d) == (x * M) >> s` and `x * M` fits in `Int` for every
# `0 <= x <= X`, with `X = 2S·d·(d-1)` the largest unreduced value those
# kernels form for a support of `S = 2k` coordinates (each call site states its
# own bound). This is the constant of `src/modular.jl`'s Barrett tier with the
# bound raised from `(d-1)^2` to `X`, and the shift chosen per modulus.
#
# `clifford_fast_dots(4S, d)` bounds `4S(d-1)^2 >= X`, so `X` itself is formed
# without overflow. The shift is the largest that keeps `X * M < 2^63`: with
# `X < 2^L`, `L <= 62` and `2^lg <= d`, `M <= 2^(s-lg) + 1` gives
# `X * M < 2^(L+s-lg) + 2^L <= 2^63`. Exactness is the same criterion
# as `_barrett_ok`, `e * X < 2^s` for `e = M*d - 2^s`, tested by division so
# the product is never formed. Acceptance is NOT monotonic in `d`: a modulus
# whose `e` happens to be small passes where a smaller one fails, so never
# replace the test with a bound on `d`. Two divisions, once per preparation.
@inline function clifford_barrett(S::Int, d::Int)
    (S > 0 && d >= 2 && clifford_fast_dots(4S, d)) || return (0, 0)
    X = 2S * d * (d - 1)
    L = 64 - leading_zeros(X)
    L <= 62 || return (0, 0)
    lg = 63 - leading_zeros(d)
    s = 62 - L + lg
    M = cld(1 << s, d)
    e = M * d - (1 << s)
    (e == 0 || X <= ((1 << s) - 1) ÷ e) || return (0, 0)
    return (M, s)
end

# `x mod d` for `0 <= x <= X` under a constant from `clifford_barrett`.
@inline _barrett_reduce(x::Int, d::Int, M::Int, s::Int) = x - ((x * M) >> s) * d
