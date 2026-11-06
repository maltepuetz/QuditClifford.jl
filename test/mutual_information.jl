using QuditClifford
using Test
using Random

# Disjoint random subsystems, the first `count` nonempty, possibly leaving
# sites traced out.
function mi_random_parts(rng, n, count)
    perm = randperm(rng, n)
    parts = Vector{Vector{Int}}()
    start = 1
    for p in 1:count
        remaining = n - start + 1 - (count - p)
        len = rand(rng, 1:remaining)
        push!(parts, perm[start:(start + len - 1)])
        start += len
    end
    return parts
end

mi_S(tab, X) = entanglement_entropy(tab, X)
mi_reference(tab, A, B) = mi_S(tab, A) + mi_S(tab, B) - mi_S(tab, vcat(A, B))
mi_tripartite_reference(tab, A, B, C) =
    mi_S(tab, A) + mi_S(tab, B) + mi_S(tab, C) -
    mi_S(tab, vcat(A, B)) - mi_S(tab, vcat(A, C)) - mi_S(tab, vcat(B, C)) +
    mi_S(tab, vcat(A, B, C))

# |i, j, i+j, i+2j⟩ summed over i, j: every pair of qutrits carries two units
# of entropy, so the state is absolutely maximally entangled.
function mi_ame_qutrits(TT)
    tab = TT(3, 4; state=:product)
    apply!(tab, Fourier(1))
    apply!(tab, Fourier(2))
    for g in (SUM(1, 3), SUM(2, 3), SUM(1, 4), SUM(2, 4, 2))
        apply!(tab, g)
    end
    return tab
end

mi_allocated(tab, A, B) = @allocated mutual_information(tab, A, B)
mi_allocated(tab, A) = @allocated mutual_information(tab, A)
mi_allocated(tab, A, B, C) = @allocated tripartite_information(tab, A, B, C)

@testset "Mutual information" begin
    @testset "Subsystem validation" begin
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            for tab in (TT(d, 4; state=:ghz), TT(d, 4; state=:mixed))
                for bad in ([5], [0], [-1], [1, 1])
                    @test_throws ArgumentError mutual_information(tab, bad, [3])
                    @test_throws ArgumentError mutual_information(tab, [3], bad)
                    @test_throws ArgumentError mutual_information(tab, bad)
                    @test_throws ArgumentError tripartite_information(tab, bad, [3], [4])
                    @test_throws ArgumentError tripartite_information(tab, [3], bad, [4])
                    @test_throws ArgumentError tripartite_information(tab, [3], [4], bad)
                end
                @test_throws ArgumentError mutual_information(tab, [1.5], [2])
                @test_throws ArgumentError mutual_information(tab, [1, 2], [2, 3])
                @test_throws ArgumentError tripartite_information(tab, [1], [2], [1])
                @test_throws ArgumentError tripartite_information(tab, [1], [2, 3], [3])
                @test mutual_information(tab, Int[], [1, 2]) == 0
                @test mutual_information(tab, [1, 2], Int[]) == 0
                @test mutual_information(tab, Int[]) == 0
                @test mutual_information(tab, 1:4) == 0
                @test tripartite_information(tab, Int[], [1], [2]) == 0
                @test tripartite_information(tab, [1], Int[], [2]) == 0
                @test tripartite_information(tab, [1], [2], Int[]) == 0
                # Subsystems stored in the scratch mask are copied, not clobbered.
                tab.c_workspace[1:3] .= (2, 1, 4)
                @test mutual_information(tab, view(tab.c_workspace, 1:2),
                                         view(tab.c_workspace, 3:3)) ==
                      mutual_information(tab, [1, 2], [4])
                tab.c_workspace[1:2] .= (2, 1)
                @test mutual_information(tab, view(tab.c_workspace, 1:2)) ==
                      mutual_information(tab, [1, 2])
                tab.c_workspace[1:3] .= (2, 1, 4)
                @test tripartite_information(tab, view(tab.c_workspace, 1:1),
                                             view(tab.c_workspace, 2:2),
                                             view(tab.c_workspace, 3:3)) ==
                      tripartite_information(tab, [2], [1], [4])
            end
        end
    end

    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            for d in (2, 3)
                @testset "Named states (d=$d)" begin
                    bell = TT(d, 2; state=:ghz)
                    @test mutual_information(bell, [1], [2]) == 2
                    @test mutual_information(bell, [1]) == 2

                    ghz = TT(d, 4; state=:ghz)
                    @test mutual_information(ghz, [1], [2]) == 1
                    @test mutual_information(ghz, [1, 2], [3, 4]) == 2
                    @test tripartite_information(ghz, [1], [2], [3]) == 1
                    @test tripartite_information(ghz, [1], [2], [3, 4]) == 0

                    @test mutual_information(TT(d, 4; state=:product), [1, 2], [3, 4]) == 0
                    @test mutual_information(TT(d, 4; state=:mixed), [1, 2], [3, 4]) == 0
                    @test tripartite_information(TT(d, 4; state=:mixed), [1], [2], [3]) == 0
                end
            end

            @testset "Absolutely maximally entangled qutrits" begin
                ame = mi_ame_qutrits(TT)
                for pair in ([1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4])
                    @test entanglement_entropy(ame, pair) == 2
                end
                @test mutual_information(ame, [1], [2]) == 0
                @test mutual_information(ame, [1], [2, 3]) == 2
                @test tripartite_information(ame, [1], [2], [3]) == -2
                # A pure state split into three parts with nothing traced out
                # has I₃ = 0, whatever its entanglement.
                @test tripartite_information(ame, [1], [2], [3, 4]) == 0
            end

            @testset "Entropy reference" begin
                rng = Xoshiro(20261106)
                values = Set{Int}()
                for d in (2, 3, 5), trial in 1:60
                    n = rand(rng, 3:12)
                    tab = TT(d, n; storephase=isodd(trial))
                    random_state!(rng, tab; m=rand(rng, 0:n))
                    A, B, C = mi_random_parts(rng, n, 3)
                    stab_before = copy(tab.stab)

                    I = mutual_information(tab, A, B)
                    @test I == mi_reference(tab, A, B)
                    @test I == mutual_information(tab, B, A)
                    @test mutual_information(tab, A) == mi_reference(tab, A, setdiff(1:n, A))

                    I3 = tripartite_information(tab, A, B, C)
                    @test I3 == mi_tripartite_reference(tab, A, B, C)
                    for (X, Y, Z) in ((B, C, A), (C, A, B), (B, A, C), (A, C, B))
                        @test tripartite_information(tab, X, Y, Z) == I3
                    end
                    @test I3 == mutual_information(tab, A, B) +
                                mutual_information(tab, A, C) -
                                mutual_information(tab, A, vcat(B, C))
                    @test tab.stab == stab_before
                    push!(values, I3)
                end
                # Random stabilizer states are scrambled, so I₃ is mostly
                # negative; the GHZ states above cover the positive sign.
                @test count(<(0), values) >= 1 && length(values) >= 3
            end

            @testset "No allocations" begin
                tab = TT(3, 10)
                random_state!(Xoshiro(5), tab; m=8)
                A = [1, 4, 6]
                B = [2, 3]
                C = [8, 10]
                mi_allocated(tab, A, B)
                mi_allocated(tab, A)
                mi_allocated(tab, A, B, C)
                @test mi_allocated(tab, A, B) == 0
                @test mi_allocated(tab, A) == 0
                @test mi_allocated(tab, A, B, C) == 0
            end
        end
    end
end
