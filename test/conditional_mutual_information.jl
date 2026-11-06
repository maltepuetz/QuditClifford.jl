using QuditClifford
using Test
using Random

# Disjoint random subsystems A, B and C, with A and B nonempty and C possibly
# empty, leaving the remaining sites traced out.
function cmi_random_parts(rng, n)
    perm = randperm(rng, n)
    na = rand(rng, 1:(n - 1))
    nb = rand(rng, 1:(n - na))
    nc = rand(rng, 0:(n - na - nb))
    return perm[1:na], perm[(na + 1):(na + nb)], perm[(na + nb + 1):(na + nb + nc)]
end

cmi_S(tab, X) = entanglement_entropy(tab, X)
cmi_reference(tab, A, B, C) =
    cmi_S(tab, vcat(A, C)) + cmi_S(tab, vcat(B, C)) - cmi_S(tab, C) - cmi_S(tab, vcat(A, B, C))

cmi_allocated(tab, A, B, C) = @allocated conditional_mutual_information(tab, A, B, C)

@testset "Conditional mutual information" begin
    @testset "Subsystem validation" begin
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            for tab in (TT(d, 4; state=:ghz), TT(d, 4; state=:mixed))
                for bad in ([5], [0], [-1], [1, 1])
                    @test_throws ArgumentError conditional_mutual_information(tab, bad, [3], [4])
                    @test_throws ArgumentError conditional_mutual_information(tab, [3], bad, [4])
                    @test_throws ArgumentError conditional_mutual_information(tab, [3], [4], bad)
                end
                @test_throws ArgumentError conditional_mutual_information(tab, [1.5], [2], [3])
                @test_throws ArgumentError conditional_mutual_information(tab, [1], [2], [1])
                @test_throws ArgumentError conditional_mutual_information(tab, [1], [2, 3], [3])
                @test_throws ArgumentError conditional_mutual_information(tab, [1, 2], [2], Int[])
                @test conditional_mutual_information(tab, Int[], [1], [2]) == 0
                @test conditional_mutual_information(tab, [1], Int[], [2]) == 0
                @test conditional_mutual_information(tab, Int[], Int[], 1:4) == 0
                # Subsystems stored in the scratch mask are copied, not clobbered.
                tab.c_workspace[1:3] .= (2, 1, 4)
                @test conditional_mutual_information(tab, view(tab.c_workspace, 1:1),
                                                     view(tab.c_workspace, 2:2),
                                                     view(tab.c_workspace, 3:3)) ==
                      conditional_mutual_information(tab, [2], [1], [4])
            end
        end
    end

    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            for d in (2, 3)
                @testset "Named states (d=$d)" begin
                    # GHZ with qudit 4 traced out: qudits 1 to 3 are classically
                    # correlated, and conditioning on qudit 3 removes all of it.
                    ghz = TT(d, 4; state=:ghz)
                    @test conditional_mutual_information(ghz, [1], [2], [3]) == 0
                    # When nothing is traced out the state is pure, and
                    # I(A:B|C) = S(A) + S(B) - S(C).
                    @test conditional_mutual_information(ghz, [1], [2], [3, 4]) == 1
                    @test conditional_mutual_information(ghz, [1], [2], [3, 4]) ==
                          cmi_S(ghz, [1]) + cmi_S(ghz, [2]) - cmi_S(ghz, [3, 4])
                    @test conditional_mutual_information(TT(d, 3; state=:ghz), [1], [2], [3]) == 1
                    @test conditional_mutual_information(ghz, [1], [2], Int[]) ==
                          mutual_information(ghz, [1], [2])

                    # A Bell pair on qudits 1 and 3, with qudit 2 independent.
                    bell = TT(d, 3; state=:product, basis=[:X, :Z, :Z])
                    apply!(bell, SUM(1, 3))
                    @test conditional_mutual_information(bell, [1], [3], [2]) == 2
                    @test conditional_mutual_information(bell, [1], [2], [3]) == 0

                    @test conditional_mutual_information(TT(d, 4; state=:product), [1], [2], [3]) == 0
                    @test conditional_mutual_information(TT(d, 4; state=:mixed), [1], [2], [3]) == 0
                end
            end

            @testset "Absolutely maximally entangled qutrits" begin
                # |i, j, i+j, i+2j⟩ summed over i and j: every pair of qutrits
                # carries two units of entropy.
                ame = TT(3, 4; state=:product)
                apply!(ame, Fourier(1))
                apply!(ame, Fourier(2))
                for g in (SUM(1, 3), SUM(2, 3), SUM(1, 4), SUM(2, 4, 2))
                    apply!(ame, g)
                end
                @test conditional_mutual_information(ame, [1], [2], [3]) == 2
                @test conditional_mutual_information(ame, [1], [2], Int[]) == 0
                @test conditional_mutual_information(ame, [1], [2], [3, 4]) == 0
            end

            @testset "Entropy reference" begin
                rng = Xoshiro(20261106)
                values = Set{Int}()
                for d in (2, 3, 5), trial in 1:60
                    n = rand(rng, 2:12)
                    tab = TT(d, n; storephase=isodd(trial))
                    random_state!(rng, tab; m=rand(rng, 0:n))
                    A, B, C = cmi_random_parts(rng, n)
                    stab_before = copy(tab.stab)

                    I = conditional_mutual_information(tab, A, B, C)
                    @test I isa Int
                    @test I == cmi_reference(tab, A, B, C)
                    @test I >= 0  # strong subadditivity
                    @test conditional_mutual_information(tab, B, A, C) == I
                    @test I <= 2 * min(length(A), length(B))
                    # Chain rule: I(A:BC) = I(A:C) + I(A:B|C).
                    @test mutual_information(tab, A, vcat(B, C)) ==
                          mutual_information(tab, A, C) + I
                    # I₃(A:B:C) = I(A:B) - I(A:B|C).
                    if !isempty(C)
                        @test tripartite_information(tab, A, B, C) ==
                              mutual_information(tab, A, B) - I
                    end
                    @test tab.stab == stab_before
                    push!(values, I)
                end
                @test length(values) >= 4
            end

            @testset "Pure states trace out the smaller side" begin
                # For a pure state the conditioning system and the traced-out
                # rest can be swapped; cover both C smaller and larger than it.
                rng = Xoshiro(7)
                for d in (2, 3), trial in 1:20
                    n = 12
                    tab = TT(d, n)
                    random_state!(rng, tab)
                    perm = randperm(rng, n)
                    A, B = perm[1:2], perm[3:4]
                    for C in (perm[5:6], perm[5:10], perm[5:12])
                        rest = setdiff(1:n, vcat(A, B, C))
                        I = conditional_mutual_information(tab, A, B, C)
                        @test I == cmi_reference(tab, A, B, C)
                        @test I == conditional_mutual_information(tab, A, B, rest)
                    end
                end
            end

            @testset "No allocations" begin
                tab = TT(3, 10)
                random_state!(Xoshiro(5), tab; m=8)
                A = [1, 4, 6]
                B = [2, 3]
                C = [8, 10]
                cmi_allocated(tab, A, B, C)
                @test cmi_allocated(tab, A, B, C) == 0
                pure = TT(3, 10)
                random_state!(Xoshiro(6), pure)
                cmi_allocated(pure, A, B, C)
                @test cmi_allocated(pure, A, B, C) == 0
            end
        end
    end
end
