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
