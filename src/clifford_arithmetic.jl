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
