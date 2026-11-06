using QuditClifford
using Test
using Random
using LinearAlgebra

# Dense reference: trace out every site outside A ∪ B, transpose B, and take
# log_d of the trace norm. Sites are in kron order, qudit 1 most significant.
function neg_dense_negativity(tab, A, B)
    d, n = tab.d, tab.n
    rho = density_matrix(tab)
    keep = sort(vcat(A, B))
    k = length(keep)
    rest = setdiff(1:n, keep)
    digits_of(i, len) = reverse(digits(i, base=d, pad=len))
    index_of(ds) = foldl((a, b) -> a * d + b, ds; init=0) + 1
    rhoAB = zeros(ComplexF64, d^k, d^k)
    for i in 0:(d^n - 1), j in 0:(d^n - 1)
        di = digits_of(i, n)
        dj = digits_of(j, n)
        all(di[q] == dj[q] for q in rest) || continue
        rhoAB[index_of(di[keep]), index_of(dj[keep])] += rho[i + 1, j + 1]
    end
    local_B = [findfirst(==(b), keep) for b in B]
    rhoT = similar(rhoAB)
    for i in 0:(d^k - 1), j in 0:(d^k - 1)
        di = digits_of(i, k)
        dj = digits_of(j, k)
        for q in local_B
            di[q], dj[q] = dj[q], di[q]
        end
        rhoT[index_of(di), index_of(dj)] = rhoAB[i + 1, j + 1]
    end
    return log(d, sum(abs, eigvals(Hermitian(rhoT))))
end

# Disjoint random A and B, both nonempty, possibly leaving sites traced out.
function neg_random_split(rng, n)
    perm = randperm(rng, n)
    na = rand(rng, 1:(n - 1))
    nb = rand(rng, 1:(n - na))
    return perm[1:na], perm[(na + 1):(na + nb)]
end

neg_allocated(tab, A, B) = @allocated entanglement_negativity(tab, A, B)
neg_allocated(tab, A) = @allocated entanglement_negativity(tab, A)

@testset "Entanglement negativity" begin
    @testset "Subsystem validation" begin
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            for tab in (TT(d, 4; state=:ghz), TT(d, 4; state=:mixed))
                for bad in ([5], [0], [-1], [1, 1], [2, 1, 2])
                    @test_throws ArgumentError entanglement_negativity(tab, bad, [3])
                    @test_throws ArgumentError entanglement_negativity(tab, [3], bad)
                    @test_throws ArgumentError entanglement_negativity(tab, bad)
                end
                @test_throws ArgumentError entanglement_negativity(tab, [1.5], [2])
                @test_throws ArgumentError entanglement_negativity(tab, [1, 2], [2, 3])
                @test_throws ArgumentError entanglement_negativity(tab, [1], [1])
                @test entanglement_negativity(tab, Int[], [1, 2]) == 0
                @test entanglement_negativity(tab, [1, 2], Int[]) == 0
                @test entanglement_negativity(tab, Int[]) == 0
                @test entanglement_negativity(tab, 1:4) == 0
                # Subsystems stored in the scratch mask are copied, not clobbered.
                tab.c_workspace[1:3] .= (2, 1, 4)
                @test entanglement_negativity(tab, view(tab.c_workspace, 1:2),
                                              view(tab.c_workspace, 3:3)) ==
                      entanglement_negativity(tab, [1, 2], [4])
                tab.c_workspace[1:2] .= (2, 1)
                @test entanglement_negativity(tab, view(tab.c_workspace, 1:2)) ==
                      entanglement_negativity(tab, [1, 2])
            end
        end
    end

    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            for d in (2, 3)
                @testset "Named states (d=$d)" begin
                    bell = TT(d, 2; state=:ghz)
                    @test entanglement_negativity(bell, [1], [2]) == 1
                    @test entanglement_negativity(bell, [2], [1]) == 1
                    @test entanglement_negativity(bell, [1]) == 1

                    # Tracing out one GHZ qudit leaves classical correlations.
                    ghz = TT(d, 3; state=:ghz)
                    @test entanglement_negativity(ghz, [1], [2]) == 0
                    @test entanglement_negativity(ghz, [1], [2, 3]) == 1
                    @test entanglement_negativity(ghz, [3]) == 1
                    @test entanglement_entropy(ghz, [1, 2]) == 1

                    @test entanglement_negativity(TT(d, 4; state=:product), [1, 2], [3, 4]) == 0
                    @test entanglement_negativity(TT(d, 4; state=:mixed), [1, 2], [3, 4]) == 0

                    # Two Bell pairs across the cut {1,3} | {2,4}.
                    pairs = TT(d, 4; state=:product, basis=[:X, :Z, :X, :Z])
                    apply!(pairs, SUM(1, 2))
                    apply!(pairs, SUM(3, 4))
                    @test entanglement_negativity(pairs, [1, 3], [2, 4]) == 2
                    @test entanglement_negativity(pairs, [1], [2, 3, 4]) == 1
                    @test entanglement_negativity(pairs, [1], [4]) == 0
                end
            end

            @testset "Dense reference" begin
                rng = Xoshiro(20261105)
                nonzero = 0
                for (d, nmax) in ((2, 4), (3, 4), (5, 3), (7, 2)), trial in 1:40
                    n = rand(rng, 2:nmax)
                    tab = TT(d, n)
                    random_state!(rng, tab; m=rand(rng, 0:n))
                    A, B = neg_random_split(rng, n)
                    stab_before = copy(tab.stab)
                    E = entanglement_negativity(tab, A, B)
                    @test tab.stab == stab_before
                    @test E isa Int
                    @test E ≈ neg_dense_negativity(tab, A, B) atol = 1e-8
                    @test entanglement_negativity(tab, B, A) == E
                    nonzero += E > 0
                end
                # The reference must also have exercised entangled cases.
                @test nonzero >= 20
            end

            for d in (2, 3)
                @testset "Structural identities (d=$d)" begin
                    rng = Xoshiro(17 + d)
                    for trial in 1:30
                        n = rand(rng, 2:10)
                        A, B = neg_random_split(rng, n)
                        complement = setdiff(1:n, A)

                        pure = TT(d, n; storephase=false)
                        random_state!(rng, pure)
                        @test entanglement_negativity(pure, A) == entanglement_entropy(pure, A)
                        @test entanglement_negativity(pure, A) ==
                              entanglement_negativity(pure, A, complement)

                        mixed = TT(d, n)
                        random_state!(rng, mixed; m=rand(rng, 0:n))
                        E = entanglement_negativity(mixed, A, B)
                        @test 0 <= E <= min(length(A), length(B))
                        # Local Cliffords on A and on B leave E unchanged.
                        apply!(mixed, random_clifford(rng, d, A))
                        apply!(mixed, random_clifford(rng, d, B))
                        @test entanglement_negativity(mixed, A, B) == E
                        # Enlarging B by traced-out sites cannot raise E.
                        rest = setdiff(1:n, vcat(A, B))
                        @test entanglement_negativity(mixed, A, vcat(B, rest)) >= E
                    end
                end
            end

            @testset "No allocations" begin
                tab = TT(3, 8)
                random_state!(Xoshiro(5), tab; m=6)
                A = [1, 4, 6]
                B = [2, 3, 8]
                neg_allocated(tab, A, B)
                neg_allocated(tab, A)
                @test neg_allocated(tab, A, B) == 0
                @test neg_allocated(tab, A) == 0
            end
        end
    end
end
