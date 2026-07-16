#############################################################
# Span decomposition helpers shared by expectation + measure #
#############################################################

"""
    coeffs_from_generators!(tab, op)

Fill `tab.c_workspace` with coefficients of `op` in the active stabilizer-generator basis.

- For `StabilizerTableau`, coefficients are read from pivot rows of the canonical tableau.
- For `DestabilizerTableau`, coefficients are computed via dual symplectic pairings.
"""
function coeffs_from_generators! end

@inline function coeffs_from_generators!(tab::StabilizerTableau, op)
    n = tab.n
    d = tab.d

    tab.iscanonical || canonicalize!(tab)

    cvec = tab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end

    # Read coefficients from pivot rows in canonical form.
    opvec = tab.res_workspace
    set_operator!(opvec, tab, op)
    piv = tab.pivcol_of_row
    @inbounds for r in 1:(2n)
        pc = piv[r]
        pc == 0 && continue
        cvec[pc] = mod(opvec[r], d)
    end

    return cvec
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op)
    n = tab.n
    m = tab.m
    d = tab.d
    destab = tab.destab

    cvec = tab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end

    # Use res_workspace as temporary xz(op) storage.
    opvec = tab.res_workspace
    set_operator!(opvec, tab, op)

    @inbounds for j in 1:m
        coeff = 0
        @turbo for q in 1:n
            coeff += destab[q, j] * opvec[n + q]
        end
        @turbo for q in 1:n
            coeff -= destab[n + q, j] * opvec[q]
        end
        cvec[j] = mod(coeff, d)
    end

    return cvec
end

@inline function _coeffs_from_destab_sparse!(
    tab::DestabilizerTableau,
    qudits::NTuple{D,Int},
    xs::NTuple{D,Int},
    zs::NTuple{D,Int},
) where D
    m = tab.m
    n = tab.n
    d = tab.d
    destab = tab.destab

    cvec = tab.c_workspace
    @turbo for j in eachindex(cvec)
        cvec[j] = 0
    end

    @inbounds for j in 1:m
        coeff = 0
        @simd for i in 1:D
            q = qudits[i]
            coeff += destab[q, j] * zs[i]
            coeff -= destab[n + q, j] * xs[i]
        end
        cvec[j] = mod(coeff, d)
    end
    return cvec
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op::SinglePauli)
    d = tab.d
    return _coeffs_from_destab_sparse!(
        tab,
        (op.qudit,),
        (mod(op.x, d),),
        (mod(op.z, d),),
    )
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op::DoublePauli)
    d = tab.d
    return _coeffs_from_destab_sparse!(
        tab,
        (op.qudit1, op.qudit2),
        (mod(op.x1, d), mod(op.x2, d)),
        (mod(op.z1, d), mod(op.z2, d)),
    )
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op::TriplePauli)
    d = tab.d
    return _coeffs_from_destab_sparse!(
        tab,
        (op.qudit1, op.qudit2, op.qudit3),
        (mod(op.x1, d), mod(op.x2, d), mod(op.x3, d)),
        (mod(op.z1, d), mod(op.z2, d), mod(op.z3, d)),
    )
end

@inline function coeffs_from_generators!(tab::DestabilizerTableau, op::NPauli{D}) where D
    d = tab.d
    xs = ntuple(i -> mod(op.xs[i], d), D)
    zs = ntuple(i -> mod(op.zs[i], d), D)
    return _coeffs_from_destab_sparse!(tab, op.qudits, xs, zs)
end

"""
    residual_from_coeffs!(tab, op)

Compute `op - Σ_j c_j g_j` in XZ space and store it in `tab.res_workspace`,
where `c_j` are taken from `tab.c_workspace` and `g_j` are active stabilizer generators.
"""
@inline function residual_from_coeffs!(tab::AbstractTableau, op)
    res = tab.res_workspace
    set_operator!(res, tab, op)

    d = tab.d
    m = tab.m
    n = tab.n
    cvec = tab.c_workspace
    stab = tab.stab

    @inbounds for j in 1:m
        aj = cvec[j]
        aj == 0 && continue
        @inbounds @simd for i in 1:(2n)
            res[i] = mod(res[i] - mod(aj * stab[i, j], d), d)
        end
    end
    return res
end

"""
    in_span_and_coeffs!(tab, op)

Fill coefficients in `tab.c_workspace`, residual in `tab.res_workspace`,
and return whether `op` is in the span of active stabilizer generators.
"""
@inline function in_span_and_coeffs!(tab::AbstractTableau, op)
    coeffs_from_generators!(tab, op)
    residual_from_coeffs!(tab, op)

    n = tab.n
    res = tab.res_workspace
    @inbounds for i in 1:(2n)
        res[i] != 0 && return false
    end
    return true
end

"""
    phase_exponent_from_coeffs!(tab::AbstractTableau)

Compute the phase exponent of `Q = ∏_j g_j^{c[j]}` from `tab.c_workspace`,
where `g_j` are active stabilizer generators.
"""
@inline function phase_exponent_from_coeffs!(tab::AbstractTableau)
    n = tab.n
    m = tab.m
    d = tab.d
    stab = tab.stab
    tab.storephase || return 0

    prow = 2n + 1
    d_phase = phase_modulus(d)

    cvec = tab.c_workspace
    zacc = tab.zacc_workspace
    @turbo for q in 1:n
        zacc[q] = 0
    end
    kacc = 0

    if d == 2
        @inbounds for j in 1:m
            aj = cvec[j]
            aj == 0 && continue

            cross = 0
            @turbo for q in 1:n
                cross += stab[q, j] * zacc[q]
            end
            cross = cross & 1

            kacc = mod(kacc + stab[prow, j] + 2 * cross, d_phase)

            @inbounds @simd for q in 1:n
                zacc[q] = (zacc[q] + stab[n+q, j]) & 1
            end
        end
        return kacc
    else
        inv2 = tab.inversemod(2, d)
        xdotz_cache = tab.xdotz_cache

        @inbounds for j in 1:m
            aj = mod(cvec[j], d)
            aj == 0 && continue

            xdotz_j = xdotz_cache[j]
            kpow = mod(aj * stab[prow, j] + binom2_mod_oddprime(aj, d, inv2) * xdotz_j, d_phase)

            cross_base = dot_xz_col_vs_zacc(stab, n, j, zacc, d)
            cross = mod(aj * cross_base, d)

            kacc = mod(kacc + kpow + cross, d_phase)

            @inbounds @simd for q in 1:n
                zacc[q] = mod(zacc[q] + mod(aj * stab[n+q, j], d), d)
            end
        end
        return kacc
    end
end
