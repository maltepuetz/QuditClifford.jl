##### some helper functions used in multiple files #####


#################################
# Phase-aware generator algebra #
#################################

# phase modulus:
# - odd prime d: ω^k with k mod d
# - qubits d=2: i^k with k mod 4   (needed for Clifford closure)
@inline phase_modulus(d::Int) = (d == 2 ? 4 : d)

# Helper: dot product x_i ⋅ z_j for a given column i and j, mod d.
# useful when updating phases while multiplying stabilizer generators
@inline function dot_xz_col(tab::AbstractMatrix{Int64}, n::Int, i::Int, j::Int, d::Int)
    s = 0
    @turbo for q in 1:n
        s += tab[q, i] * tab[n+q, j]
    end
    return mod(s, d)
end
@inline dot_xz_col(tab::AbstractMatrix{Int64}, n::Int, j::Int, d::Int) = dot_xz_col(tab, n, j, j, d)

# dot(x_src, z_tgt) where x_src from generator_workspace and z_tgt from tableau column
@inline function dot_xz_ws_vs_col(genws::Vector{Int64}, tab::AbstractMatrix{Int64}, n::Int, tgt::Int, d::Int)
    s = 0
    @turbo for q in 1:n
        s += genws[q] * tab[n+q, tgt]
    end
    return mod(s, d)
end

# dot(x_ws, z_ws) for a generator stored in workspace
@inline function dot_xz_ws(genws::Vector{Int64}, n::Int, d::Int)
    s = 0
    @turbo for q in 1:n
        s += genws[q] * genws[n+q]
    end
    return mod(s, d)
end

# dot(x_col(j), zacc) mod d
@inline function dot_xz_col_vs_zacc(tab::AbstractMatrix{Int64}, n::Int, j::Int, zacc::Vector{Int64}, d::Int)
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
