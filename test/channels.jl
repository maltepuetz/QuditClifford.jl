using QuditClifford
using Test
using Random
using LinearAlgebra

# Every top-level name in this file is prefixed `chan_`: runtests.jl includes all
# test files into one module.

const CHAN_TABLEAU_TYPES = [("StabilizerTableau", StabilizerTableau),
                            ("DestabilizerTableau", DestabilizerTableau)]

# ⊗_q X^x_q Z^z_q with qudit 1 as the most significant kron factor, no phase.
function chan_pauli_matrix(d::Int, n::Int, xz::Vector{Int})
    X = zeros(ComplexF64, d, d)
    for j in 0:d-1
        X[mod(j + 1, d)+1, j+1] = 1
    end
    Z = zeros(ComplexF64, d, d)
    for j in 0:d-1
        Z[j+1, j+1] = cis(2π * j / d)
    end
    M = ones(ComplexF64, 1, 1)
    for q in 1:n
        M = kron(M, X^mod(xz[q], d) * Z^mod(xz[n+q], d))
    end
    return M
end

# (1/d) Σ_j P^j ρ P^{-j}. Conjugation by P^d is trivial, so this holds for every
# phase convention.
function chan_dephase_oracle(rho::AbstractMatrix, d::Int, n::Int, xz::Vector{Int})
    P = chan_pauli_matrix(d, n, xz)
    out = zeros(ComplexF64, size(rho))
    Pj = Matrix{ComplexF64}(I, size(rho)...)
    for _ in 0:d-1
        out .+= Pj * rho * Pj'
        Pj = Pj * P
    end
    return out ./ d
end

function chan_assert_invariants(tab)
    n, d, m = tab.n, tab.d, tab.m
    @test 0 <= m <= n
    rows = size(tab.stab, 1)
    @test all(==(0), view(tab.stab, :, m+1:n))
    @test all(x -> 0 <= x < d, view(tab.stab, 1:2n, 1:m))
    if tab.storephase
        @test all(x -> 0 <= x < (d == 2 ? 4 : d), view(tab.stab, rows, 1:m))
    end
    if tab isa DestabilizerTableau
        @test all(==(0), view(tab.destab, :, m+1:n))
        @test all(x -> 0 <= x < d, view(tab.destab, :, 1:m))
        for i in 1:m, j in 1:m
            s = 0
            for q in 1:n
                s += tab.destab[q, i] * tab.stab[n+q, j] - tab.destab[n+q, i] * tab.stab[q, j]
            end
            @test mod(s, d) == (i == j ? 1 : 0)
        end
        for j in 1:m
            xz = 0
            for q in 1:n
                xz += tab.stab[q, j] * tab.stab[n+q, j]
            end
            @test tab.xdotz_cache[j] == mod(xz, d)
        end
    end
    return nothing
end

# Phase-free StabilizerTableau holding the active XZ columns of `tab`.
function chan_strip(tab::AbstractTableau)
    tab.m == 0 && return StabilizerTableau(tab.d, tab.n; state=:mixed, storephase=false)
    return StabilizerTableau(tab.d, tab.stab[1:2tab.n, 1:tab.m]; m=tab.m, storephase=false)
end

# A random nonzero-phase operator in each representation, with the exponents
# offset by random multiples of d and a random phase.
function chan_random_ops(rng, d, n)
    off() = d * rand(rng, -2:2)
    ops = Any[SinglePauli(rand(rng, 1:n), rand(rng, 0:d-1) + off(), rand(rng, 0:d-1) + off(), rand(rng, 1:5))]
    if n >= 2
        q = randperm(rng, n)
        push!(ops, DoublePauli(q[1], rand(rng, 0:d-1) + off(), rand(rng, 0:d-1) + off(),
                               q[2], rand(rng, 0:d-1) + off(), rand(rng, 0:d-1) + off(), rand(rng, 1:5)))
    end
    xz = [rand(rng, 0:d-1) + off() for _ in 1:2n]
    push!(ops, GeneralPauli(copy(xz), rand(rng, 1:5)))
    push!(ops, copy(xz))
    return ops
end

function chan_xz(op, d, n)
    if op isa AbstractVector
        return mod.(op[1:2n], d)
    elseif op isa GeneralPauli
        return mod.(op.xz, d)
    end
    qudits, xs, zs, _ = QuditClifford._few_qudit_pauli_data(op)
    xz = zeros(Int, 2n)
    for (q, x, z) in zip(qudits, xs, zs)
        xz[q] = mod(x, d)
        xz[n+q] = mod(z, d)
    end
    return xz
end

chan_allocs_dephase(t, op) = @allocated dephase!(t, op)

@testset "dephase!" begin
    for (label, TT) in CHAN_TABLEAU_TYPES
        @testset "$label" begin
            @testset "Bell pair" for d in (2, 3, 5)
                tab = TT(d, 2; state=:ghz)
                @test dephase!(tab, SinglePauli(1, 0, 1)) === tab
                @test tab.m == 1
                @test entanglement_entropy(tab, [1]) == 1
                @test mutual_information(tab, [1], [2]) == 1
                @test entanglement_negativity(tab, [1], [2]) == 0
                chan_assert_invariants(tab)
            end

            @testset "matches the dense channel" for d in (2, 3, 5)
                rng = Xoshiro(100 + d)
                for n in 1:3, m in 0:n, _ in 1:3
                    tab0 = TT(d, n)
                    random_state!(rng, tab0; m)
                    rho0 = density_matrix(tab0)
                    for op in chan_random_ops(rng, d, n)
                        tab = copy(tab0)
                        dephase!(tab, op)
                        @test isapprox(density_matrix(tab),
                                       chan_dephase_oracle(rho0, d, n, chan_xz(op, d, n)); atol=1e-10)
                        @test tab.m in (tab0.m, tab0.m - 1)
                        chan_assert_invariants(tab)
                    end
                end
            end

            @testset "storephase=false" for d in (2, 3, 5)
                rng = Xoshiro(200 + d)
                for n in 1:3, m in 0:n, _ in 1:3
                    phased = TT(d, n)
                    random_state!(rng, phased; m)
                    free = TT(d, n; storephase=false)
                    random_state!(rng, free; m)
                    for op in chan_random_ops(rng, d, n)
                        a = copy(free)
                        dephase!(a, op)
                        @test a.m in (m, m - 1)
                        chan_assert_invariants(a)
                    end
                    # The same generators with and without phases evolve identically.
                    stripped = chan_strip(phased)
                    for op in chan_random_ops(rng, d, n)
                        b = copy(phased)
                        dephase!(b, op)
                        a = copy(stripped)
                        dephase!(a, op)
                        @test chan_strip(b) == a
                    end
                end
            end

            @testset "commuting is a no-op" for d in (2, 3, 5)
                tab = TT(d, 2; state=:ghz)
                canonicalize!(tab)
                before = copy(tab)
                for op in (DoublePauli(1, 0, 1, 2, 0, d - 1), SinglePauli(1, 0, 0))
                    dephase!(tab, op)
                    @test tab.stab == before.stab
                    @test tab.m == before.m
                    @test tab.iscanonical
                end
                mixed = TT(d, 3; state=:mixed)
                dephase!(mixed, SinglePauli(1, 1, 0))
                @test mixed == TT(d, 3; state=:mixed)
                @test mixed.m == 0
            end

            @testset "ignores the phase and powers" begin
                rng = Xoshiro(7)
                for d in (2, 3), _ in 1:5
                    t = TT(d, 3)
                    random_state!(rng, t; m=2)
                    @test dephase!(copy(t), SinglePauli(1, 1, 1)) ==
                          dephase!(copy(t), SinglePauli(1, 1, 1, 1))
                    @test_logs dephase!(copy(t), SinglePauli(1, 1, 1))
                end
                t = TT(3, 3)
                random_state!(rng, t; m=2)
                @test dephase!(copy(t), SinglePauli(2, 0, 2)) == dephase!(copy(t), SinglePauli(2, 0, 1))
                @test dephase!(copy(t), SinglePauli(2, 2, 1)) == dephase!(copy(t), SinglePauli(2, 1, 2))
            end

            @testset "is idempotent" for d in (2, 3, 5)
                rng = Xoshiro(11 + d)
                for _ in 1:5
                    t = TT(d, 3)
                    random_state!(rng, t; m=rand(rng, 1:3))
                    for op in chan_random_ops(rng, d, 3)
                        once = dephase!(copy(t), op)
                        twice = dephase!(copy(once), op)
                        @test once == twice
                    end
                end
            end

            @testset "reduces huge exponents" for d in (2, 3)
                rng = Xoshiro(13)
                for _ in 1:5
                    t = TT(d, 2)
                    random_state!(rng, t; m=2)
                    @test dephase!(copy(t), SinglePauli(1, typemin(Int), 1)) ==
                          dephase!(copy(t), SinglePauli(1, mod(typemin(Int), d), 1))
                end
            end

            @testset "rejects Paulis outside the register" for d in (2, 3)
                tab = TT(d, 2; state=:ghz)
                before = copy(tab)
                for op in (SinglePauli(3, 1, 0), SinglePauli(0, 1, 0),
                           DoublePauli(1, 1, 0, 3, 0, 1),
                           GeneralPauli([1, 0, 0], 0), Int[1, 0, 0])
                    @test_throws ArgumentError dephase!(tab, op)
                    @test tab.stab == before.stab && tab.m == before.m
                end
            end

            @testset "allocates nothing" for d in (2, 3)
                rng = Xoshiro(17)
                t = TT(d, 4)
                random_state!(rng, t; m=3)
                for op in (SinglePauli(1, 1, 0), GeneralPauli([1, 0, 1, 1, 0, 1, 0, 0], 0))
                    chan_allocs_dephase(copy(t), op)
                    fresh = copy(t)
                    @test chan_allocs_dephase(fresh, op) == 0
                end
            end
        end
    end
end
