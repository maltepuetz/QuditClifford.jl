using QuditClifford
using Test
using LinearAlgebra
using Random

const QC = QuditClifford

const DENSE_TABLEAU_TYPES = [("StabilizerTableau", StabilizerTableau),
                             ("DestabilizerTableau", DestabilizerTableau)]

# Independent reference implementation, `ref_*`. Built only from the documented
# basis actions -- it never calls a production root, index or group-product
# helper -- so agreement with it cannot come from sharing a convention mistake.
#
# Every top-level name in this file is prefixed `ref_` or `dense_`: runtests.jl
# includes all test files into one module, and test/unitaries.jl already owns
# `oracle_pauli`, `oracle_projector`, `oracle_embed` and a 0-based
# `oracle_index`.

ref_p(d) = d == 2 ? 4 : d
ref_zeta(e, d) = cis(2π * e / ref_p(d))      # i^e for qubits, ω^e otherwise
ref_omega(e, d) = cis(2π * e / d)

function ref_shift(d)                           # X|j⟩ = |j+1 mod d⟩
    X = zeros(ComplexF64, d, d)
    for j in 0:d-1
        X[mod(j + 1, d)+1, j+1] = 1
    end
    return X
end
ref_clock(d) = Matrix{ComplexF64}(Diagonal([ref_omega(j, d) for j in 0:d-1]))

# ζ^a ⊗_q X_q^x_q Z_q^z_q, qudit 1 as the most significant kron factor.
function ref_pauli(d, x, z, a)
    X, Z = ref_shift(d), ref_clock(d)
    M = ones(ComplexF64, 1, 1)
    for q in eachindex(x)
        M = kron(M, X^x[q] * Z^z[q])
    end
    return ref_zeta(a, d) * M
end

ref_index(c, d) = 1 + sum((c[q] * d^(length(c) - q) for q in eachindex(c)); init=0)
ref_digits(i, d, n) = [div(i - 1, d^(n - q)) % d for q in 1:n]
function ref_basis(c, d)
    v = zeros(ComplexF64, d^length(c))
    v[ref_index(c, d)] = 1
    return v
end

# The code-space projector of the ORIGINAL generators, Π = ∏_j (Σ_t g_j^t)/d.
function ref_projector(tab)
    d, n = tab.d, tab.n
    Π = Matrix{ComplexF64}(I, d^n, d^n)
    for j in 1:tab.m
        g = ref_pauli(d, tab.stab[1:n, j], tab.stab[(n+1):(2n), j], tab.stab[2n+1, j])
        Π = Π * sum(g^t for t in 0:d-1) / d
    end
    return Π
end

# A preset driven through a seeded mix of the seven named gates and measure!,
# as test/unitaries.jl prepares states. `mixed=true` starts maximally mixed and
# keeps m < n; phase_policy=1 keeps every qubit generator Hermitian.
function dense_random_tableau(TT, d, n, rng; mixed::Bool=false, steps::Int=12)
    tab = TT(d, n; state=mixed ? :mixed : :product, basis=:Z)
    mixed && n >= 2 && measure!(tab, SinglePauli(1, 0, 1); outcome=rand(rng, 0:d-1))
    for _ in 1:steps
        q = rand(rng, 1:n)
        choice = rand(rng, 1:(n >= 2 ? 8 : 5))
        if choice == 1
            apply!(tab, Fourier(q))
        elseif choice == 2
            apply!(tab, Phase(q))
        elseif choice == 3
            apply!(tab, PauliGate(q, rand(rng, 0:d-1), rand(rng, 0:d-1)))
        elseif choice == 4
            d > 2 && apply!(tab, Multiplier(q, rand(rng, 1:d-1)))
        elseif choice == 5
            x, z = rand(rng, 0:d-1), rand(rng, 0:d-1)
            (x, z) == (0, 0) && (z = 1)
            if !mixed || tab.m < n - 1
                measure!(tab, SinglePauli(q, x, z); outcome=rand(rng, 0:d-1), phase_policy=1)
            end
        else
            q2 = rand(rng, setdiff(1:n, q))
            choice == 6 && apply!(tab, SUM(q, q2, rand(rng, 1:d-1)))
            choice == 7 && apply!(tab, CPhase(q, q2, rand(rng, 1:d-1)))
            choice == 8 && apply!(tab, SWAP(q, q2))
        end
    end
    return tab
end

# The exception `f()` throws, or `nothing` when it returns.
function dense_thrown(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

# Every field of a tableau, deep-copied: stabilizers, destabilizers, inactive
# capacity, flags, caches and workspaces alike.
dense_snapshot(tab) = [deepcopy(getfield(tab, f)) for f in fieldnames(typeof(tab))]
dense_same_field(a, b) = a == b
dense_same_field(a::QC.PrecomputedInvMod, b::QC.PrecomputedInvMod) = a.lookuptable == b.lookuptable
dense_same_field(::QC.JustInTimeInvMod, ::QC.JustInTimeInvMod) = true
dense_unchanged(tab, snap) =
    all(dense_same_field(getfield(tab, f), s) for (f, s) in zip(fieldnames(typeof(tab)), snap))

# The contract each conversion enforces before touching the tableau: stored
# phases (refused even maximally mixed), purity where it applies, Hermitian
# qubit generators reported by original column, and a positive budget. The
# non-Hermitian fixture is Z₂ then XZ on qubit 1 with phase 0: commuting and
# independent, so the constructors accept it, and the conversions must not.
function dense_check_contract(f, TT; needs_pure::Bool)
    for state in (:ghz, :mixed)
        @test_throws ArgumentError f(TT(2, 2; state, storephase=false))
    end
    needs_pure && @test_throws ArgumentError f(TT(3, 2; state=:mixed))
    bad = TT(2, [0 1; 0 0; 0 1; 1 0; 0 0]; m=2)
    err = dense_thrown(() -> f(bad))
    @test err isa ArgumentError && occursin("column 2", err.msg)
    for budget in (0, -1)
        @test_throws ArgumentError f(TT(3, 2; state=:ghz); maxentries=budget)
    end
end

##########################
# State expansion kernels #
##########################

@testset "Overflow-safe powers" begin
    @test QC._bounded_pow(3, 0, 1) == 1
    @test QC._bounded_pow(3, 4, 81) == 81
    @test QC._bounded_pow(3, 4, 80) == -1
    @test QC._bounded_pow(3, 100, 1 << 24) == -1        # 3^100 is never formed
    @test QC._bounded_pow(2, 30, typemax(Int)) == 2^30
    @test QC._bounded_pow(2, Sys.WORD_SIZE - 1, typemax(Int)) == -1
end

@testset "Group walk visits each element once, in odometer order" begin
    for (label, TT) in DENSE_TABLEAU_TYPES, d in (2, 3)
        tab = dense_random_tableau(TT, d, 3, Xoshiro(10 + d))
        n, m = tab.n, tab.m
        G = tab.stab
        gens = [ref_pauli(d, G[1:n, j], G[(n+1):(2n), j], G[2n+1, j]) for j in 1:m]
        seen = Matrix{ComplexF64}[]
        QC._walk_group(G, n, m, d, zeros(Int, 2n), zeros(Int, m)) do xz, a
            push!(seen, ref_pauli(d, xz[1:n], xz[(n+1):(2n)], a))
        end
        # exponent vectors in lexicographic order, last one fastest; d = 3,
        # m = 3 carries across one digit at step 3 and two at step 9
        expected = [prod(gens[j]^u[j] for j in 1:m)
                    for u in (ref_digits(w, d, m) for w in 1:d^m)]
        @test length(seen) == d^m
        @test all(isapprox.(seen, expected; atol=1e-10))
    end
    # no generators: the identity, once
    visits = Ref(0)
    QC._walk_group(zeros(Int, 5, 2), 2, 0, 3, zeros(Int, 4), Int[]) do xz, a
        visits[] += 1
        @test all(iszero, xz) && a == 0
    end
    @test visits[] == 1
end

@testset "Support preparation ($label, d=$d)" for (label, TT) in DENSE_TABLEAU_TYPES, d in (2, 3)
    for trial in 1:6
        tab = dense_random_tableau(TT, d, 3, Xoshiro(100d + trial))
        before = dense_snapshot(tab)
        G, k, cstar = QC._prepare_support(tab)
        n = tab.n
        @test dense_unchanged(tab, before)
        @test G !== tab.stab
        @test all(any(!iszero, G[1:n, j]) for j in 1:k)
        @test all(all(iszero, G[1:n, j]) for j in (k+1):n)
        # the support is exactly d^k labels, and cstar is the smallest
        Π = ref_projector(tab)
        support = [ref_digits(i, d, n) for i in 1:d^n if real(Π[i, i]) > 1e-9]
        @test length(support) == d^k
        @test cstar == first(support)
    end
end

##########################
# StabilizerKet and ket  #
##########################

@testset "ket is the state the generators fix ($label, d=$d)" for (label, TT) in DENSE_TABLEAU_TYPES, d in (2, 3)
    for n in 1:3, trial in 1:3
        tab = dense_random_tableau(TT, d, n, Xoshiro(1000d + 10n + trial))
        k = ket(tab)
        T = length(k.phases)
        @test k.d == d && k.n == n
        @test size(k.labels) == (n, T)
        @test all(0 .<= k.labels .< d)
        @test all(0 .<= k.phases .< ref_p(d))
        # canonical form: strictly ascending labels, first phase zero
        cols = collect(eachcol(k.labels))
        @test issorted(cols) && allunique(cols)
        @test k.phases[1] == 0
        # the vector it denotes is fixed by the rank-one projector
        ψ = sum(ref_zeta(k.phases[t], d) * ref_basis(k.labels[:, t], d) for t in 1:T) / sqrt(T)
        @test ref_projector(tab) * ψ ≈ ψ atol = 1e-10
    end
end

@testset "ket: known states ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    k = ket(TT(2, 4; state=:product, basis=:Z))
    @test k.labels == zeros(Int, 4, 1) && k.phases == [0]

    for d in (2, 3)
        full = ket(TT(d, 2; state=:product, basis=:X))
        @test full.labels == reduce(hcat, [ref_digits(i, d, 2) for i in 1:d^2])
        @test full.phases == zeros(Int, d^2)

        ghz = ket(TT(d, 3; state=:ghz))
        @test ghz.labels == repeat((0:d-1)', 3)
        @test ghz.phases == zeros(Int, d)
    end

    # the four qubit Bell sign/phase patterns
    for (gates, phase) in ((AbstractClifford[], 0), ([PauliGate(1, 0, 1)], 2),
                           ([Phase(1)], 1), ([Phase(1), PauliGate(1, 0, 1)], 3))
        tab = TT(2, 2; state=:ghz)
        foreach(g -> apply!(tab, g), gates)
        @test ket(tab).labels == [0 1; 0 1]
        @test ket(tab).phases == [0, phase]
    end

    # a nonzero affine offset: (|01⟩ + |10⟩)/√2 starts at cstar = (0, 1)
    tab = TT(2, 2; state=:ghz)
    apply!(tab, PauliGate(2, 1, 0))
    @test ket(tab).labels == [0 1; 1 0]

    # zero qudits: one empty label with phase 0
    z = ket(TT(2, 0))
    @test size(z.labels) == (0, 1) && z.phases == [0]
end

@testset "ket: sign fixtures ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    # Y = iXZ fixes (|0⟩ + i|1⟩)/√2 and -Y = i³XZ fixes (|0⟩ - i|1⟩)/√2
    @test ket(TT(2, 1; state=:product, basis=:Y)).phases == [0, 1]
    @test ket(TT(2, reshape([1, 1, 1], 3, 1))).phases == [0, 1]
    @test ket(TT(2, reshape([1, 1, 3], 3, 1))).phases == [0, 3]
    # -Z = i²Z: a Z-only qubit column whose even phase halves exactly
    @test ket(TT(2, reshape([0, 1, 2], 3, 1))).labels == reshape([1], 1, 1)
    # ωX at d = 3 fixes the X eigenvector of eigenvalue ω²
    k = ket(TT(3, reshape([1, 0, 1], 3, 1)))
    @test k.labels == [0 1 2] && k.phases == [0, 1, 2]
    # ωZ fixes only |2⟩
    @test ket(TT(3, reshape([0, 1, 1], 3, 1))).labels == reshape([2], 1, 1)
end

@testset "ket: exact equality across generator bases ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    for d in (2, 3), trial in 1:4
        rng = Xoshiro(500d + trial)
        tab = dense_random_tableau(TT, d, 3, rng)
        before = ket(tab)
        again = ket(deepcopy(tab))
        @test again == before && isequal(again, before) && hash(again) == hash(before)
        permuted = TT(d, tab.stab[:, randperm(rng, 3)]; m=3)
        @test ket(permuted) == before && hash(ket(permuted)) == hash(before)
        # canonicalize! returns nothing, so compare after the fact
        canonicalize!(tab)
        @test ket(tab) == before
    end

    # the same state by two preparations
    for d in (2, 3)
        viagates = TT(d, 3; state=:product, basis=:Z)
        apply!(viagates, Fourier(1))
        apply!(viagates, SUM(1, 2))
        apply!(viagates, SUM(1, 3))
        @test ket(viagates) == ket(TT(d, 3; state=:ghz))
    end
    flipped = TT(2, 1; state=:product, basis=:Z)
    apply!(flipped, PauliGate(1, 1, 0))
    @test ket(flipped) == ket(TT(2, reshape([0, 1, 2], 3, 1)))

    # {X₁Z₂, X₁X₂Z₁Z₂} and {X₁Z₂, -Z₁X₂} generate one group, hence one state;
    # the product's sign is what a phase-free tableau loses
    A = TT(2, [1 1; 0 1; 0 1; 1 1; 0 0]; m=2)
    B = TT(2, [1 0; 0 1; 0 1; 1 0; 0 2]; m=2)
    B_unsigned = TT(2, [1 0; 0 1; 0 1; 1 0; 0 0]; m=2)
    @test ket(A) == ket(B)
    @test ket(A) != ket(B_unsigned)
    @test Dict(ket(A) => :state)[ket(B)] == :state
end

@testset "ket owns its data" begin
    tab = StabilizerTableau(3, 2; state=:ghz)
    k = ket(tab)
    frozen = deepcopy(k)
    apply!(tab, Fourier(1))
    measure!(tab, SinglePauli(2, 1, 0); outcome=1)
    @test k == frozen
    # no public constructor from raw data
    @test_throws MethodError StabilizerKet(3, 2, zeros(Int, 2, 1), [0])
end

@testset "ket: contract ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    dense_check_contract(ket, TT; needs_pure=true)
end

@testset "ket: budgets are exact at the threshold ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    @test length(ket(TT(3, 2; state=:ghz); maxentries=9).phases) == 3   # (n + 1) T = 9
    @test_throws ArgumentError ket(TT(3, 2; state=:ghz); maxentries=8)
    # full support needs (n + 1) d^n = 12 here
    @test_throws ArgumentError ket(TT(2, 2; state=:product, basis=:X); maxentries=4)
    # 100 qutrits: three terms of 100 digits and a phase each
    big = TT(3, 100; state=:ghz)
    @test length(ket(big; maxentries=303).phases) == 3
    @test_throws ArgumentError ket(big; maxentries=302)
end

################
# state_vector #
################

@testset "state_vector ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    # |1,0⟩ pins the ordering; Bell and GHZ states are symmetric and cannot
    tab = TT(2, 2; state=:product, basis=:Z)
    apply!(tab, PauliGate(1, 1, 0))
    @test state_vector(tab) == ComplexF64[0, 0, 1, 0]
    @test state_vector(TT(2, 2; state=:product, basis=[:Z, :X])) ≈ kron([1, 0], [1, 1] / sqrt(2))
    @test state_vector(TT(3, 0)) == ComplexF64[1]
    @test state_vector(ket(TT(3, 0))) == ComplexF64[1]

    for d in (2, 3), n in 1:3, trial in 1:3
        tab = dense_random_tableau(TT, d, n, Xoshiro(2000d + 10n + trial))
        k = ket(tab)
        ψ = state_vector(tab)
        @test length(ψ) == d^n
        @test ψ ≈ state_vector(k)
        @test norm(ψ) ≈ 1
        i = findfirst(v -> abs(v) > 1e-12, ψ)
        @test abs(imag(ψ[i])) < 1e-12 && real(ψ[i]) > 0
        T = length(k.phases)
        @test all(ψ[ref_index(k.labels[:, t], d)] ≈ ref_zeta(k.phases[t], d) / sqrt(T) for t in 1:T)
        @test count(v -> abs(v) > 1e-12, ψ) == T
    end
end

@testset "state_vector: contract ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    dense_check_contract(state_vector, TT; needs_pure=true)
end

@testset "state_vector: budgets are exact at the threshold ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    @test length(state_vector(TT(3, 2; state=:ghz); maxentries=9)) == 9
    @test_throws ArgumentError state_vector(TT(3, 2; state=:ghz); maxentries=8)
    # A full-support vector fits in D = 4 although its exact ket, (n + 1) D,
    # does not: the tableau method must not route through ket.
    plus = TT(2, 2; state=:product, basis=:X)
    @test length(state_vector(plus; maxentries=4)) == 4
    @test_throws ArgumentError state_vector(ket(plus); maxentries=3)
    # d^n is never formed, so this is a clean refusal naming the power
    err = dense_thrown(() -> state_vector(TT(3, 100; state=:ghz)))
    @test err isa ArgumentError && occursin("3^100", err.msg)
end

##################
# density_matrix #
##################

ref_embed(d, n, q, u) = kron(Matrix{ComplexF64}(I, d^(q - 1), d^(q - 1)), u,
                                Matrix{ComplexF64}(I, d^(n - q), d^(n - q)))
ref_fourier(d) = [ref_omega(r * c, d) / sqrt(d) for r in 0:d-1, c in 0:d-1]
ref_phase_gate(d) = d == 2 ? ComplexF64[1 0; 0 im] :
    Matrix{ComplexF64}(Diagonal([ref_omega(j * (j - 1) ÷ 2, d) for j in 0:d-1]))
function ref_multiplier(d, a)
    M = zeros(ComplexF64, d, d)
    for j in 0:d-1
        M[mod(a * j, d)+1, j+1] = 1
    end
    return M
end

# A gate given by its action on basis labels, c ↦ amplitude · |c′⟩.
function ref_monomial(d, n, action)
    D = d^n
    U = zeros(ComplexF64, D, D)
    for col in 1:D
        c′, amp = action(ref_digits(col, d, n))
        U[ref_index(c′, d), col] += amp
    end
    return U
end
function ref_sum(d, n, ctl, tgt, a)
    return ref_monomial(d, n, c -> begin
        c′ = copy(c)
        c′[tgt] = mod(c[tgt] + a * c[ctl], d)
        (c′, 1.0)
    end)
end
ref_cphase(d, n, q1, q2, a) =
    ref_monomial(d, n, c -> (c, ref_omega(a * c[q1] * c[q2], d)))
function ref_swap(d, n, q1, q2)
    return ref_monomial(d, n, c -> begin
        c′ = copy(c)
        c′[q1], c′[q2] = c[q2], c[q1]
        (c′, 1.0)
    end)
end

# (gate, U) for all seven named gates on n >= 2 qudits: nontrivial
# coefficients, a reversed control/target, and nonadjacent targets at n = 3.
function ref_gate_cases(d, n)
    cases = Any[
        (Fourier(2), ref_embed(d, n, 2, ref_fourier(d))),
        (Phase(n), ref_embed(d, n, n, ref_phase_gate(d))),
        (PauliGate(1, 1, d - 1), ref_embed(d, n, 1, ref_shift(d) * ref_clock(d)^(d - 1))),
        (SUM(n, 1, d - 1), ref_sum(d, n, n, 1, d - 1)),
        (SUM(1, 2), ref_sum(d, n, 1, 2, 1)),
        (CPhase(1, n, d - 1), ref_cphase(d, n, 1, n, d - 1)),
        (SWAP(n, 1), ref_swap(d, n, n, 1)),
    ]
    d > 2 && push!(cases, (Multiplier(2, d - 1), ref_embed(d, n, 2, ref_multiplier(d, d - 1))))
    return cases
end

# Every Pauli word P(x, z, a) on n qudits with every central phase a, checking
# tr(ρ P) against expect! on a COPY: expect! may canonicalize, and running it on
# the original would hide a mutation made by the conversion. Returns the
# failing (x, z, a), so a failure names its word.
function dense_expect_mismatches(tab, rho)
    d, n = tab.d, tab.n
    bad = Tuple{Vector{Int},Vector{Int},Int}[]
    for w in 0:(d^(2n)-1)
        xz = [div(w, d^(i - 1)) % d for i in 1:(2n)]
        x, z = xz[1:n], xz[(n+1):(2n)]
        for a in 0:(ref_p(d)-1)
            lhs = tr(rho * ref_pauli(d, x, z, a))
            rhs = expect!(deepcopy(tab), GeneralPauli(xz, a))
            isapprox(lhs, rhs; atol=1e-10) || push!(bad, (x, z, a))
        end
    end
    return bad
end


@testset "Dense basis counter against recomputation" begin
    # (3, 2) is where a row digit wraps without a column carry; (5, 1) makes
    # the wrap distance d - 1 = 4
    for (d, n) in ((2, 3), (3, 2), (5, 1))
        D, p = d^n, ref_p(d)
        strides = QC._kron_strides(d, n)
        bad = Tuple{Vector{Int},Int}[]
        for w in 0:(d^(2n)-1), a in 0:(p-1)
            xz = [div(w, d^(i - 1)) % d for i in 1:(2n)]
            rho = zeros(ComplexF64, D, D)
            QC._add_pauli!(rho, xz, a, n, d, p, p ÷ d, strides, zeros(Int, n), zeros(Int, n), 1.0)
            isapprox(rho, ref_pauli(d, xz[1:n], xz[(n+1):(2n)], a); atol=1e-12) || push!(bad, (xz, a))
        end
        @test isempty(bad)
    end
end

@testset "density_matrix agrees with expect! and the projector ($label, d=$d)" for (label, TT) in DENSE_TABLEAU_TYPES, d in (2, 3)
    for n in 1:3, mixed in (false, true), trial in 1:2
        tab = dense_random_tableau(TT, d, n, Xoshiro(3000d + 100n + 10mixed + trial); mixed)
        rho = density_matrix(tab)
        @test isempty(dense_expect_mismatches(tab, rho))
        @test rho ≈ ref_projector(tab) / d^(n - tab.m) atol = 1e-10
    end
end

@testset "density_matrix structure ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    for d in (2, 3), n in 1:3, mixed in (false, true)
        tab = dense_random_tableau(TT, d, n, Xoshiro(4000d + 10n + mixed); mixed)
        rho = density_matrix(tab)
        R = d^(n - tab.m)
        @test rho ≈ rho' atol = 1e-12
        @test tr(rho) ≈ 1
        @test rho * rho ≈ rho / R atol = 1e-10
        vals = eigvals(Hermitian(rho))
        @test count(v -> v > 1e-9, vals) == R
        @test all(v -> abs(v) < 1e-9 || abs(v - 1 / R) < 1e-9, vals)
        if tab.m == n
            ψ = state_vector(tab)
            @test rho ≈ ψ * ψ' atol = 1e-10
        end
    end
    # Z₁ alone on two qubits: |0⟩⟨0| ⊗ I/2, a mixed tensor product
    @test density_matrix(TT(2, reshape([0, 0, 1, 0, 0], 5, 1))) ≈ kron([1 0; 0 0], Matrix(I, 2, 2) / 2)
    @test density_matrix(TT(3, 2; state=:mixed)) ≈ Matrix(I, 9, 9) / 9
    @test density_matrix(TT(2, 0)) == fill(1.0 + 0im, 1, 1)
end

@testset "apply! agrees with U ρ U† ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    for (d, n) in ((2, 3), (3, 3), (5, 2)), mixed in (false, true)
        rng = Xoshiro(5000d + 10n + mixed)
        for (g, U) in ref_gate_cases(d, n)
            tab = dense_random_tableau(TT, d, n, rng; mixed)
            before = density_matrix(tab)
            apply!(tab, g)
            @test density_matrix(tab) ≈ U * before * U' atol = 1e-10
        end
    end
end

@testset "density_matrix: contract ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    dense_check_contract(density_matrix, TT; needs_pure=false)
end

@testset "density_matrix: budgets are exact at the threshold ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    @test size(density_matrix(TT(3, 2; state=:ghz); maxentries=81)) == (9, 9)
    @test_throws ArgumentError density_matrix(TT(3, 2; state=:ghz); maxentries=80)
    err = dense_thrown(() -> density_matrix(TT(3, 100; state=:ghz)))
    @test err isa ArgumentError && occursin("3^100", err.msg)
end

###########
# Display #
###########

@testset "Display ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    bell = ket(TT(2, 2; state=:ghz))
    @test sprint(show, MIME"text/plain"(), bell) == "StabilizerKet (d = 2, n = 2, 2 terms):\n  (|00⟩ + |11⟩)/√2"
    @test repr(bell) == "(|00⟩ + |11⟩)/√2"

    basis = ket(TT(2, 4; state=:product, basis=:Z))
    @test repr(basis) == "|0000⟩"
    @test sprint(show, MIME"text/plain"(), basis) == "StabilizerKet (d = 2, n = 4, 1 term):\n  |0000⟩"
    @test repr(ket(TT(2, 0))) == "|⟩"

    for (gates, text) in (([PauliGate(1, 0, 1)], "(|00⟩ − |11⟩)/√2"),
                          ([Phase(1)], "(|00⟩ + i|11⟩)/√2"),
                          ([Phase(1), PauliGate(1, 0, 1)], "(|00⟩ − i|11⟩)/√2"))
        tab = TT(2, 2; state=:ghz)
        foreach(g -> apply!(tab, g), gates)
        @test repr(ket(tab)) == text
    end
    @test repr(ket(TT(3, reshape([1, 0, 1], 3, 1)))) == "(|0⟩ + ω₃|1⟩ + ω₃²|2⟩)/√3"
    @test repr(ket(TT(11, 2; state=:ghz))) ==
          "(|0,0⟩ + |1,1⟩ + |2,2⟩ + |3,3⟩ + |4,4⟩ + |5,5⟩ + |6,6⟩ + |7,7⟩ + |8,8⟩ + |9,9⟩ + |10,10⟩)/√11"

    # term cap: the full count stays in the divisor
    plus = ket(TT(2, 5; state=:product, basis=:X))
    s = repr(plus)
    @test count("⟩", s) == 16 && endswith(s, " + …)/√32")
    @test sprint(show, plus; context=:max_ket_terms => 3) == "(|00000⟩ + |00001⟩ + |00010⟩ + …)/√32"

    # label cap, counted in coordinates
    wide = ket(TT(3, 100; state=:ghz))
    @test repr(wide) == "(|" * "0"^16 * "…" * "0"^16 * "⟩ + |" * "1"^16 * "…" * "1"^16 *
                        "⟩ + |" * "2"^16 * "…" * "2"^16 * "⟩)/√3"
    @test sprint(show, wide; context=:max_ket_label => 4) == "(|00…00⟩ + |11…11⟩ + |22…22⟩)/√3"
    @test sprint(show, wide; context=:max_ket_label => 3) == "(|00…0⟩ + |11…1⟩ + |22…2⟩)/√3"
    @test sprint(show, ket(TT(11, 5; state=:ghz)); context=(:max_ket_label => 2, :max_ket_terms => 2)) ==
          "(|0,…,0⟩ + |1,…,1⟩ + …)/√11"

    # a limited IOContext respects the display width
    narrow = sprint(show, plus; context=(:limit => true, :displaysize => (24, 40)))
    @test narrow == "(|00000⟩ + |00001⟩ + |00010⟩ + …)/√32"
    @test textwidth(narrow) <= 40
    boxed = sprint(show, MIME"text/plain"(), plus; context=(:limit => true, :displaysize => (24, 40)))
    @test textwidth(last(split(boxed, '\n'))) <= 40
    # the last term needs no room for an ellipsis: at exactly the expression's
    # width of 16 it prints whole, and one column less elides it
    @test textwidth(repr(bell)) == 16
    @test sprint(show, bell; context=(:limit => true, :displaysize => (24, 16))) == "(|00⟩ + |11⟩)/√2"
    @test sprint(show, bell; context=(:limit => true, :displaysize => (24, 15))) == "(|00⟩ + …)/√2"

    for bad in (0, -2, "x", 1.5)
        @test_throws ArgumentError sprint(show, bell; context=:max_ket_terms => bad)
        @test_throws ArgumentError sprint(show, bell; context=:max_ket_label => bad)
    end
end

##################################################
# Non-mutation, arithmetic envelope, allocation #
##################################################

# Bytes a call allocates net of the buffers it cannot avoid. Those are measured
# the same way, so the allocator's rounding of large blocks -- whole 16 KiB
# pages on Apple silicon -- cancels instead of being mistaken for waste. Every
# thunk runs once first, so compilation is excluded.
function dense_net_allocated(f, unavoidable...)
    f()
    total = @allocated f()
    for g in unavoidable
        g()
        total -= @allocated g()
    end
    return total
end


@testset "Conversions leave the tableau untouched ($label)" for (label, TT) in DENSE_TABLEAU_TYPES
    for d in (2, 3), mixed in (false, true)
        tab = dense_random_tableau(TT, d, 3, Xoshiro(6000d + mixed); mixed)
        for phase in (:as_prepared, :canonical)
            phase === :canonical && canonicalize!(tab)
            snap = dense_snapshot(tab)
            density_matrix(tab)
            if !mixed
                ket(tab)
                state_vector(tab)
            end
            @test_throws ArgumentError ket(tab; maxentries=1)
            @test_throws ArgumentError state_vector(tab; maxentries=1)
            @test_throws ArgumentError density_matrix(tab; maxentries=1)
            @test dense_unchanged(tab, snap)
        end
    end
end

@testset "Arithmetic envelope" begin
    jit = QC.JustInTimeInvMod()
    inside, past = Sys.WORD_SIZE == 64 ? (2147483647, 2147483659) : (32749, 32771)
    @test inside <= QC.max_safe_dimension(2) < past

    # The largest prime inside the n = 2 envelope, with a one-term ket: nothing
    # may scale with d, so an all-roots table would show up here.
    big = @test_logs StabilizerTableau(inside, 2; state=:product, basis=:Z, inversemod=jit)
    k = ket(big)
    @test k.labels == zeros(Int, 2, 1) && k.phases == [0]
    @test repr(k) == "|0,0⟩"
    @test dense_net_allocated(() -> ket(big)) < 4096

    # The first prime past it is refused rather than computed inexactly.
    over = @test_logs (:warn, r"overflow") StabilizerTableau(past, 2; state=:product, basis=:Z, inversemod=jit)
    for f in (ket, state_vector, density_matrix)
        err = dense_thrown(() -> f(over))
        @test err isa ArgumentError && occursin(string(QC.max_safe_dimension(2)), err.msg)
    end
end

@testset "Byte counts must be representable" begin
    # Entry counts inside maxentries = typemax(Int) whose byte counts are not.
    # The message is asserted, not just the type: Base's own array constructor
    # also refuses such sizes with an ArgumentError, so the type alone would
    # pass with the package's guard deleted.
    W = Sys.WORD_SIZE
    n = 0
    while (n + 2) * big(2)^(n + 1) <= typemax(Int)
        n += 1
    end
    @test (n + 1) * big(2)^n * sizeof(Int) > typemax(Int)
    for (f, tab) in ((state_vector, StabilizerTableau(2, W - 4; state=:product)),
                     (density_matrix, StabilizerTableau(2, (W - 4) ÷ 2; state=:mixed)),
                     (ket, StabilizerTableau(2, n; state=:product, basis=:X)))
        err = dense_thrown(() -> f(tab; maxentries=typemax(Int)))
        @test err isa ArgumentError && occursin("byte count", err.msg)
    end
end

@testset "Allocation sanity" begin
    I8 = sizeof(Int)
    # O(n) scratch per call, plus the result where it is small
    small(n, extra=0) = I8 * (16n + extra) + 2048

    # a basis state and a 100-qutrit GHZ: polynomial, nothing proportional to d^n
    basis = StabilizerTableau(2, 64; state=:product, basis=:Z)
    ghz = StabilizerTableau(3, 100; state=:ghz)
    @test dense_net_allocated(() -> ket(basis), () -> copy(basis.stab)) < small(64, 65)
    @test dense_net_allocated(() -> ket(ghz), () -> copy(ghz.stab)) < small(100, 3 * 101)

    # full support: the dense vector alone. Routing through an exact ket would
    # add I8 (n + 1) 2^n bytes -- 90 KB here, far past the slack.
    plus = StabilizerTableau(2, 10; state=:product, basis=:X)
    @test dense_net_allocated(() -> state_vector(plus),
                        () -> copy(plus.stab), () -> zeros(ComplexF64, 2^10)) < small(10)

    # maximally mixed: the output and O(n + m) scratch
    mm = StabilizerTableau(2, 6; state=:mixed)
    @test dense_net_allocated(() -> density_matrix(mm), () -> zeros(ComplexF64, 64, 64)) < small(6)

    # display touches only the terms it prints, never all 2^16
    wide = ket(StabilizerTableau(2, 16; state=:product, basis=:X))
    @test dense_net_allocated(() -> repr(wide)) < 32_768
end
