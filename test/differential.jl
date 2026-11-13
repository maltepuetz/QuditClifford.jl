using QuditClifford
using Test
using LinearAlgebra
using Random

# Differential tests against an independent dense reference. The density
# matrix is built straight from the generator columns,
# ρ = d⁻ⁿ ∏ⱼ Σₖ gⱼᵏ, and evolved alongside the tableau with explicit gate,
# projector and channel matrices, so none of the package's own dense code is
# involved.

dft_omega(d) = cis(2π / d)
dft_zeta(d) = d == 2 ? complex(0.0, 1.0) : dft_omega(d)
dft_pmod(d) = d == 2 ? 4 : d

function dft_X(d)
    M = zeros(ComplexF64, d, d)
    for j in 0:d-1
        M[mod(j + 1, d) + 1, j + 1] = 1
    end
    return M
end
dft_Z(d) = Matrix(Diagonal([dft_omega(d)^j for j in 0:d-1]))
dft_eye(d) = Matrix{ComplexF64}(I, d, d)

# ζ^a ⊗_q X^x_q Z^z_q with qudit 1 the most significant tensor factor.
function dft_pauli(d, n, xz::AbstractVector, a::Integer)
    X, Z = dft_X(d), dft_Z(d)
    M = ones(ComplexF64, 1, 1)
    for q in 1:n
        M = kron(M, X^mod(xz[q], d) * Z^mod(xz[n+q], d))
    end
    return dft_zeta(d)^mod(a, dft_pmod(d)) * M
end

function dft_embed(d, n, q, M)
    out = ones(ComplexF64, 1, 1)
    for i in 1:n
        out = kron(out, i == q ? M : dft_eye(d))
    end
    return out
end

dft_digits(idx, d, n) = [(idx ÷ d^(n - q)) % d for q in 1:n]
dft_index(c, d) = foldl((acc, x) -> acc * d + x, c; init = 0)

# Unitary sending |c⟩ to ph(c)|f(c)⟩ for a permutation f of basis digits.
function dft_perm_unitary(d, n, f)
    D = d^n
    U = zeros(ComplexF64, D, D)
    for idx in 0:D-1
        c2, ph = f(dft_digits(idx, d, n))
        U[dft_index(c2, d) + 1, idx + 1] = ph
    end
    return U
end

function dft_gate(d, n, g)
    w = dft_omega(d)
    if g isa Fourier
        F = [w^(j * k) for k in 0:d-1, j in 0:d-1] / sqrt(d)
        return dft_embed(d, n, g.qudit, F)
    elseif g isa Phase
        M = d == 2 ? Matrix(Diagonal(ComplexF64[1, im])) :
            Matrix(Diagonal([w^(j * (j - 1) ÷ 2) for j in 0:d-1]))
        return dft_embed(d, n, g.qudit, M)
    elseif g isa Multiplier
        return dft_perm_unitary(d, n, c -> begin
            c2 = copy(c); c2[g.qudit] = mod(g.a * c[g.qudit], d); (c2, 1.0 + 0im)
        end)
    elseif g isa PauliGate
        return dft_embed(d, n, g.qudit, dft_X(d)^mod(g.x, d) * dft_Z(d)^mod(g.z, d))
    elseif g isa SUM
        return dft_perm_unitary(d, n, c -> begin
            c2 = copy(c); c2[g.target] = mod(c[g.target] + g.a * c[g.control], d)
            (c2, 1.0 + 0im)
        end)
    elseif g isa CPhase
        return dft_perm_unitary(d, n, c -> (copy(c), w^mod(g.a * c[g.qudit1] * c[g.qudit2], d)))
    elseif g isa SWAP
        return dft_perm_unitary(d, n, c -> begin
            c2 = copy(c); c2[g.qudit1] = c[g.qudit2]; c2[g.qudit2] = c[g.qudit1]
            (c2, 1.0 + 0im)
        end)
    end
    error("unknown gate $g")
end

function dft_rand_gate(rng, d, n)
    q = rand(rng, 1:n)
    c = rand(rng, n >= 2 ? (1:7) : (1:4))
    q2 = n >= 2 ? rand(rng, setdiff(1:n, q)) : 0
    c == 1 && return Fourier(q)
    c == 2 && return Phase(q)
    c == 3 && return Multiplier(q, rand(rng, 1:d-1) + d * rand(rng, -2:2))
    c == 4 && return PauliGate(q, rand(rng, -d:2d), rand(rng, -d:2d))
    c == 5 && return SUM(q, q2, rand(rng, -d:2d))
    c == 6 && return CPhase(q, q2, rand(rng, -d:2d))
    return SWAP(q, q2)
end

# A random Pauli on distinct in-range qudits, Hermitian for d = 2, with
# exponents and phase deliberately left unreduced (possibly negative).
# Returns the operator with its full-length exponent vector and phase.
function dft_rand_pauli(rng, d, n; kind = rand(rng, 1:5))
    k = kind == 5 ? n : min(n, kind == 4 ? rand(rng, 1:n) : kind)
    qs = randperm(rng, n)[1:k]
    xs = [rand(rng, 0:d-1) for _ in 1:k]
    zs = [rand(rng, 0:d-1) for _ in 1:k]
    ph = d == 2 ? mod(sum(xs .* zs), 2) + 2 * rand(rng, 0:1) : rand(rng, 0:d-1)
    xs2 = xs .+ d .* rand(rng, -2:2, k)
    zs2 = zs .+ d .* rand(rng, -2:2, k)
    ph2 = ph + dft_pmod(d) * rand(rng, -2:2)
    v = zeros(Int, 2n)
    for (q, x, z) in zip(qs, xs2, zs2)
        v[q] = x; v[n+q] = z
    end
    op = if kind == 5 || (k > 3 && rand(rng, Bool))
        GeneralPauli(copy(v), ph2)
    elseif k == 1
        SinglePauli(qs[1], xs2[1], zs2[1], ph2)
    elseif k == 2
        DoublePauli(qs[1], xs2[1], zs2[1], qs[2], xs2[2], zs2[2], ph2)
    elseif k == 3
        TriplePauli(qs[1], xs2[1], zs2[1], qs[2], xs2[2], zs2[2], qs[3], xs2[3], zs2[3], ph2)
    else
        NPauli(Tuple(qs), Tuple(xs2), Tuple(zs2), ph2)
    end
    return op, v, ph2
end

# Projector onto the eigenvalue ω^t of the Pauli matrix O (O^d = I).
dft_projector(d, O, t) = sum((dft_omega(d)^(-t) * O)^k for k in 0:d-1) / d

function dft_rho(tab)
    d, n = tab.d, tab.n
    D = d^n
    rho = dft_eye(D) / D
    for j in 1:tab.m
        g = dft_pauli(d, n, tab.stab[1:2n, j], tab.storephase ? tab.stab[2n+1, j] : 0)
        rho = rho * sum(g^k for k in 0:d-1)
    end
    return rho
end

# Fully depolarize site q.
function dft_depolarize(d, n, rho, q)
    for O in (dft_embed(d, n, q, dft_X(d)), dft_embed(d, n, q, dft_Z(d)))
        rho = sum(O^k * rho * O'^k for k in 0:d-1) / d
    end
    return rho
end

# Symplectic form with 128-bit accumulation, so it stays exact at d near 2^31.
function dft_symp(u, v, n, d)
    s = Int128(0)
    for q in 1:n
        s += Int128(u[q]) * v[n+q] - Int128(u[n+q]) * v[q]
    end
    return mod(s, d)
end

function dft_check_invariants(tab)
    d, n, m = tab.d, tab.n, tab.m
    @test 0 <= m <= n
    @test all(iszero, tab.stab[:, m+1:n])
    @test all(x -> 0 <= x < d, tab.stab[1:2n, :])
    tab.storephase && @test all(x -> 0 <= x < dft_pmod(d), tab.stab[2n+1, :])
    @test all(dft_symp(tab.stab[:, i], tab.stab[:, j], n, d) == 0 for i in 1:m, j in 1:m)
    if tab isa DestabilizerTableau
        @test all(iszero, tab.destab[:, m+1:n])
        @test all(x -> 0 <= x < d, tab.destab)
        @test all(dft_symp(tab.destab[:, i], tab.stab[:, j], n, d) == (i == j)
                  for i in 1:m, j in 1:m)
        @test all(tab.xdotz_cache[j] ==
                  mod(sum(Int128(tab.stab[q, j]) * tab.stab[n+q, j] for q in 1:n; init = Int128(0)), d)
                  for j in 1:m)
    end
    if tab.iscanonical
        c = copy(tab)
        c.iscanonical = false
        canonicalize!(c)
        @test c.stab == tab.stab
    end
    return nothing
end

# ---- entropies of the dense state ----

function dft_ptrace(rho, d, n, keep)
    k = length(keep)
    rest = setdiff(1:n, keep)
    out = zeros(ComplexF64, d^k, d^k)
    for i in 0:d^n-1, j in 0:d^n-1
        ci = dft_digits(i, d, n); cj = dft_digits(j, d, n)
        ci[rest] == cj[rest] || continue
        out[dft_index(ci[keep], d) + 1, dft_index(cj[keep], d) + 1] += rho[i+1, j+1]
    end
    return out
end

function dft_entropy(rho, d, n, A)
    isempty(A) && return 0.0
    ev = eigvals(Hermitian(dft_ptrace(rho, d, n, sort(A))))
    return -sum(x > 1e-12 ? x * log(x) : 0.0 for x in ev) / log(d)
end

function dft_negativity(rho, d, n, A, B)
    (isempty(A) || isempty(B)) && return 0.0
    keep = sort(vcat(A, B))
    k = length(keep)
    r = dft_ptrace(rho, d, n, keep)
    posB = [findfirst(==(b), keep) for b in B]
    rt = similar(r)
    for i in 0:d^k-1, j in 0:d^k-1
        ci = dft_digits(i, d, k); cj = dft_digits(j, d, k)
        ci2 = copy(ci); cj2 = copy(cj)
        ci2[posB] = cj[posB]; cj2[posB] = ci[posB]
        rt[dft_index(ci2, d) + 1, dft_index(cj2, d) + 1] = r[i+1, j+1]
    end
    return log(sum(abs, eigvals(Hermitian(rt)))) / log(d)
end

# Three disjoint, possibly empty, subsystems in shuffled order.
function dft_rand_parts(rng, n)
    lab = rand(rng, 0:3, n)
    return [shuffle(rng, findall(==(l), lab)) for l in 1:3]
end

function dft_check_entropies(rng, tab, rho)
    d, n = tab.d, tab.n
    A, B, C = dft_rand_parts(rng, n)
    S(X) = dft_entropy(rho, d, n, X)
    @test entanglement_entropy(tab, A) ≈ S(A) atol = 1e-8
    @test entanglement_profile(tab) ≈ [S(collect(1:l)) for l in 0:n] atol = 1e-8
    ord = randperm(rng, n)
    @test entanglement_profile(tab, ord) ≈ [S(ord[1:l]) for l in 0:n] atol = 1e-8
    @test mutual_information(tab, A, B) ≈ S(A) + S(B) - S(vcat(A, B)) atol = 1e-8
    @test conditional_mutual_information(tab, A, B, C) ≈
          S(vcat(A, C)) + S(vcat(B, C)) - S(C) - S(vcat(A, B, C)) atol = 1e-8
    @test tripartite_information(tab, A, B, C) ≈
          S(A) + S(B) + S(C) - S(vcat(A, B)) - S(vcat(A, C)) - S(vcat(B, C)) +
          S(vcat(A, B, C)) atol = 1e-8
    @test entanglement_negativity(tab, A, B) ≈ dft_negativity(rho, d, n, A, B) atol = 1e-7
    @test entanglement_negativity(tab, A) ≈
          dft_negativity(rho, d, n, A, setdiff(1:n, A)) atol = 1e-7
    @test is_pure(tab) == (abs(real(tr(rho * rho)) - 1) < 1e-8)
    return nothing
end

# ---- measurement against dense projectors ----

# Measures a copy of `tab` with a random and with every forced outcome and
# compares with the projected dense state. Returns the branch taken:
# :inspan (deterministic), :replace (anticommuting) or :append.
function dft_check_measure(rng, tab, rho, op, v, ph)
    d, n = tab.d, tab.n
    O = dft_pauli(d, n, v, ph)
    projs = [dft_projector(d, O, t) for t in 0:d-1]
    probs = [real(tr(P * rho)) for P in projs]
    @test sum(probs) ≈ 1 atol = 1e-8
    ex = tr(rho * O)
    @test expect!(copy(tab), op) ≈ ex atol = 1e-8
    @test expect!(copy(tab), vcat(v, ph)) ≈ ex atol = 1e-8
    ei = expect_int!(copy(tab), op)
    @test (ei == -1) == (abs(ex) < 1e-8)
    ei >= 0 && @test dft_zeta(d)^ei ≈ ex atol = 1e-8
    seed = rand(rng, UInt)
    c = copy(tab)
    r = measure!(Xoshiro(seed), c, op)
    dft_check_invariants(c)
    tmax = argmax(probs) - 1
    if probs[tmax+1] > 1 - 1e-8
        @test r == tmax
        @test c.m == tab.m
        @test dft_rho(c) ≈ rho atol = 1e-8
        return :inspan
    end
    @test all(p -> isapprox(p, 1 / d; atol = 1e-8), probs)
    @test 0 <= r < d
    @test dft_rho(c) ≈ projs[r+1] * rho * projs[r+1] / probs[r+1] atol = 1e-8
    c2 = copy(tab)
    @test measure!(Xoshiro(seed), c2, op) == r
    @test c2 == c
    for t in 0:d-1
        ct = copy(tab)
        @test measure!(ct, op; outcome = t) == t
        dft_check_invariants(ct)
        @test dft_rho(ct) ≈ projs[t+1] * rho * projs[t+1] / probs[t+1] atol = 1e-8
    end
    @test c.m in (tab.m, tab.m + 1)
    return c.m == tab.m ? :replace : :append
end

function dft_rand_basis(rng, d, k)
    return [rand(rng, d == 2 ? (:X, :Y, :Z) : (:X, :Z)) for _ in 1:k]
end

function dft_basis_generator(d, b)
    b === :X && return dft_X(d)
    b === :Z && return dft_Z(d)
    return im * dft_X(d) * dft_Z(d)
end

# ---- stored Cliffords ----

# The identity Clifford on ordered targets t.
function dft_identity(d, t)
    S = 2length(t)
    return CliffordOperator(d, t, Matrix{Int}(I, S, S), zeros(Int, S))
end

# Dense matrix of U P U† from U's generator images, with P = ζ^ph ∏X ∏Z.
function dft_conjugated_dense(U, d, n, v, ph)
    t = U.targets
    k = length(t)
    img(i) = begin
        w = zeros(Int, 2n)
        for q in 1:k
            w[t[q]] = U.F[q, i]; w[n+t[q]] = U.F[k+q, i]
        end
        dft_pauli(d, n, w, U.a[i])
    end
    rest = copy(v)
    rest[t] .= 0; rest[n .+ t] .= 0
    M = dft_zeta(d)^mod(ph, dft_pmod(d)) * dft_eye(d^n)
    for q in 1:k
        M *= img(q)^mod(v[t[q]], d)
    end
    M *= dft_pauli(d, n, vcat(rest[1:n], zeros(Int, n)), 0)
    for q in 1:k
        M *= img(k + q)^mod(v[n+t[q]], d)
    end
    return M * dft_pauli(d, n, vcat(zeros(Int, n), rest[n+1:2n]), 0)
end

# U ρ U† from the expansion ρ = d⁻ⁿ Σ_P tr(P†ρ) P over the phase-free Paulis.
function dft_apply_stored(U, d, n, rho)
    out = zeros(ComplexF64, size(rho))
    for idx in 0:d^(2n)-1
        v = dft_digits(idx, d, 2n)
        c = tr(dft_pauli(d, n, v, 0)' * rho)
        abs(c) < 1e-12 && continue
        out += c * dft_conjugated_dense(U, d, n, v, 0)
    end
    return out / d^n
end

# ---- the fuzz loop ----

function dft_fuzz(d, n, TT, nsteps, seed; storephase = true)
    rng = Xoshiro(seed)
    tab = TT(d, n; storephase)
    st = rand(rng, 1:4)
    st == 2 && reset!(tab; state = :product, basis = dft_rand_basis(rng, d, n))
    st == 3 && reset!(tab; state = :ghz)
    st == 4 && random_state!(rng, tab; m = rand(rng, 0:n))
    dft_check_invariants(tab)
    rho = storephase ? dft_rho(tab) : nothing
    for step in 1:nsteps
        act = rand(rng, 1:7)
        if act <= 2
            g = dft_rand_gate(rng, d, n)
            apply!(tab, rand(rng, Bool) ? g : CliffordOperator(g, d))
            if storephase
                G = dft_gate(d, n, g)
                rho = G * rho * G'
                P, v, ph = dft_rand_pauli(rng, d, n)
                got = conjugate(g, P, n; d)
                @test dft_pauli(d, n, got.xz, got.phase) ≈
                      G * dft_pauli(d, n, v, ph) * G' atol = 1e-8
            end
        elseif act == 3
            t = randperm(rng, n)[1:rand(rng, 1:n)]
            U = random_clifford(rng, d, t)
            apply!(tab, U)
            storephase && (rho = dft_apply_stored(U, d, n, rho))
        elseif act == 4
            op, v, ph = dft_rand_pauli(rng, d, n)
            if storephase
                dft_check_measure(rng, tab, rho, op, v, ph)
                t = rand(rng, 0:d-1)
                P = dft_projector(d, dft_pauli(d, n, v, ph), t)
                p = real(tr(P * rho))
                measure!(rng, tab, op; outcome = t)
                1e-8 < p < 1 - 1e-8 && (rho = P * rho * P / p)
            else
                measure!(rng, tab, op)
            end
        elseif act == 5
            op, v, ph = dft_rand_pauli(rng, d, n)
            dephase!(tab, op)
            if storephase
                O = dft_pauli(d, n, v, ph)
                rho = sum(P * rho * P for P in (dft_projector(d, O, t) for t in 0:d-1))
            end
        elseif act == 6
            sites = randperm(rng, n)[1:rand(rng, 1:n)]
            if rand(rng, Bool)
                traceout!(tab, sites)
                if storephase
                    for q in sites
                        rho = dft_depolarize(d, n, rho, q)
                    end
                end
            else
                bas = dft_rand_basis(rng, d, length(sites))
                reset!(tab, sites; basis = bas)
                if storephase
                    for (q, b) in zip(sites, bas)
                        rho = dft_depolarize(d, n, rho, q)
                        G = dft_embed(d, n, q, dft_basis_generator(d, b))
                        rho = rho * sum(G^k for k in 0:d-1)
                    end
                end
            end
        else
            before = copy(tab)
            canonicalize!(tab)
            @test tab == before
            @test hash(tab) == hash(before)
            @test tab.iscanonical
        end
        dft_check_invariants(tab)
        if storephase
            @test dft_rho(tab) ≈ rho atol = 1e-8
            step % 5 == 0 && dft_check_entropies(rng, tab, rho)
        end
    end
    return nothing
end

const DFT_TYPES = (StabilizerTableau, DestabilizerTableau)

@testset "Dense differential" begin
    @testset "fuzz d=$d n=$n $(nameof(TT)) seed=$s" for (d, ns) in ((2, 1:3), (3, 1:3), (5, 1:2)),
                                                        n in ns, TT in DFT_TYPES, s in 1:2
        dft_fuzz(d, n, TT, 25, 1000s + 10n + d)
    end

    @testset "fuzz storephase=false d=$d n=$n $(nameof(TT))" for (d, n) in ((2, 3), (3, 3)),
                                                                 TT in DFT_TYPES
        dft_fuzz(d, n, TT, 25, 7 + d; storephase = false)
    end

    @testset "measure! branches d=$d n=$n $(nameof(TT))" for (d, ns) in ((2, 2:3), (3, 2:3), (5, 2:2)),
                                                             n in ns, TT in DFT_TYPES
        rng = Xoshiro(100d + n)
        seen = Set{Symbol}()
        for _ in 1:12
            tab = TT(d, n)
            random_state!(rng, tab; m = rand(rng, 0:n))
            rho = dft_rho(tab)
            for _ in 1:4
                push!(seen, dft_check_measure(rng, tab, rho, dft_rand_pauli(rng, d, n)...))
            end
        end
        @test seen == Set([:inspan, :replace, :append])
    end
end

# ---- group laws on random stored Cliffords ----

@testset "Group laws on random CliffordOperators d=$d k=$k" for d in (2, 3, 5), k in (2, 3)
    rng = Xoshiro(31d + k)
    n = 3
    for _ in 1:6
        t = randperm(rng, n)[1:k]
        U, V, W = (random_clifford(rng, d, t) for _ in 1:3)
        Id = dft_identity(d, t)
        @test (U ∘ V) ∘ W == U ∘ (V ∘ W)
        @test U ∘ Id == U == Id ∘ U
        @test inv(U ∘ V) == inv(V) ∘ inv(U)
        @test inv(inv(U)) == U
        @test U ∘ inv(U) == Id == inv(U) ∘ U
        P, v, ph = dft_rand_pauli(rng, d, n)
        UV = conjugate(U ∘ V, P, n)
        nested = conjugate(U, conjugate(V, P, n))
        @test UV.xz == nested.xz && UV.phase == nested.phase
        @test dft_pauli(d, n, UV.xz, UV.phase) ≈ dft_conjugated_dense(U ∘ V, d, n, v, ph) atol = 1e-8
        Vv = conjugate(V, P, n)
        @test dft_pauli(d, n, UV.xz, UV.phase) ≈
              dft_conjugated_dense(U, d, n, mod.(Vv.xz, d), Vv.phase) atol = 1e-8
        for TT in DFT_TYPES
            tab = TT(d, n)
            random_state!(rng, tab; m = rand(rng, 1:n))
            a = copy(tab); b = copy(tab)
            apply!(a, V); apply!(a, U)
            apply!(b, U ∘ V)
            @test a.stab == b.stab
            TT === DestabilizerTableau && @test a.destab == b.destab
            dft_check_invariants(b)
            apply!(b, inv(U ∘ V))
            @test b == tab
        end
    end
end

# ---- exact cross-check of the two tableau types at large primes ----

# The same seeded circuit with the same forced outcomes on both tableau types.
# `inversemod` is passed through so the largest prime need not build a
# d-entry lookup table.
function dft_cross_check(d, n, nsteps, seed; inversemod = QuditClifford.PrecomputedInvMod(d))
    rng = Xoshiro(seed)
    s = StabilizerTableau(d, n; inversemod)
    t = DestabilizerTableau(d, n; inversemod)
    m0 = rand(rng, 1:n)
    random_state!(Xoshiro(seed), s; m = m0)
    random_state!(Xoshiro(seed), t; m = m0)
    @test s == t
    for _ in 1:nsteps
        act = rand(rng, 1:6)
        if act == 1
            g = dft_rand_gate(rng, d, n)
            apply!(s, g); apply!(t, g)
        elseif act == 2
            U = random_clifford(rng, d, randperm(rng, n)[1:rand(rng, 1:n)])
            apply!(s, U); apply!(t, U)
        elseif act == 3
            op, _, _ = dft_rand_pauli(rng, d, n)
            o = rand(rng, 0:d-1)
            @test measure!(s, op; outcome = o) == measure!(t, op; outcome = o)
        elseif act == 4
            op, _, _ = dft_rand_pauli(rng, d, n)
            dephase!(s, op); dephase!(t, op)
        elseif act == 5
            q = rand(rng, 1:n)
            if rand(rng, Bool)
                traceout!(s, q); traceout!(t, q)
            else
                b = rand(rng, (:X, :Z))
                reset!(s, q; basis = b); reset!(t, q; basis = b)
            end
        else
            canonicalize!(s); canonicalize!(t)
        end
        @test s == t
        dft_check_invariants(s)
        dft_check_invariants(t)
        op, _, _ = dft_rand_pauli(rng, d, n)
        @test expect_int!(s, op) == expect_int!(t, op)
    end
    return nothing
end

@testset "Exact cross-check of tableau types" begin
    dims = Sys.WORD_SIZE == 64 ? (127, 131, 137, 1021, 1031, 65537) : (127, 131, 137, 1021, 1031)
    @testset "d=$d" for d in dims
        dft_cross_check(d, 3, 60, d)
    end
    dmax = Sys.WORD_SIZE == 64 ? 2147483647 : 32749
    @testset "d=$dmax n=2" begin
        dft_cross_check(dmax, 2, 60, 5; inversemod = QuditClifford.JustInTimeInvMod())
    end
end
