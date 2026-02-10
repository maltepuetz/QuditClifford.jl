using QuditClifford
using Test

@testset "Entanglement entropy" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                @testset "Small systems" begin
                    # Product state |00⟩ with generators Z₁ and Z₂.
                    tab_prod = zeros(Int, 5, 2)
                    tab_prod[3, 1] = 1  # Z₁
                    tab_prod[4, 2] = 1  # Z₂
                    tab_prod = TT(2, tab_prod; m=2, storephase=true)
                    @test entanglement_entropy(tab_prod, [1]) == 0

                    # Bell state |Φ+⟩ with generators X₁X₂ and Z₁Z₂.
                    tab_bell = zeros(Int, 5, 2)
                    tab_bell[1, 1] = 1  # X₁
                    tab_bell[2, 1] = 1  # X₂
                    tab_bell[3, 2] = 1  # Z₁
                    tab_bell[4, 2] = 1  # Z₂
                    tab_bell = TT(2, tab_bell; m=2, storephase=true)
                    @test entanglement_entropy(tab_bell, [1]) == 1

                    # Mixed state (m < n).
                    tab_mixed = zeros(Int, 5, 2)
                    tab_mixed[3, 1] = 1  # Z₁ only
                    tab_mixed = TT(2, tab_mixed; m=1, storephase=true)
                    @test entanglement_entropy(tab_mixed, [1]) == 0
                    @test entanglement_entropy(tab_mixed, [2]) == 1
                    @test entanglement_entropy(tab_mixed, [1, 2]) == 1
                end

                @testset "Larger mixed systems" begin
                    # Qubits (d=2), start from maximally mixed and measure commuting operators.
                    tab = TT(2, 5; storephase=true) # m=0

                    # Maximally mixed: S(A) = |A|
                    @test entanglement_entropy(tab, [1]) == 1
                    @test entanglement_entropy(tab, [2, 4]) == 2
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 5

                    # Measure Z₁
                    measure!(tab, SinglePauli(1, 0, 1))
                    @test entanglement_entropy(tab, [1]) == 0
                    @test entanglement_entropy(tab, [2]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 1
                    @test entanglement_entropy(tab, [3, 4, 5]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 4

                    # Measure X₂
                    measure!(tab, SinglePauli(2, 1, 0))
                    @test entanglement_entropy(tab, [1]) == 0
                    @test entanglement_entropy(tab, [2]) == 0
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [2, 3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 3

                    # Measure X₄X₅
                    measure!(tab, DoublePauli(4, 1, 0, 5, 1, 0))
                    @test entanglement_entropy(tab, [4, 5]) == 1
                    @test entanglement_entropy(tab, [4]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 2

                    # Measure Z₄Z₅ (now qudits 4&5 become pure Bell pair)
                    measure!(tab, DoublePauli(4, 0, 1, 5, 0, 1))
                    @test entanglement_entropy(tab, [4, 5]) == 0
                    @test entanglement_entropy(tab, [4]) == 1
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [3, 4, 5]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 1

                    # test whether subsystem can be given in arbitrary order
                    @test entanglement_entropy(tab, [5, 4]) == 0
                    @test entanglement_entropy(tab, [5, 3, 4]) == 1
                    @test entanglement_entropy(tab, [4, 3, 2, 1, 5]) == 1
                end

                @testset "Larger pure systems" begin
                    # Qubits (d=2), three Bell pairs on 6 qubits.
                    tab = zeros(Int, 13, 6)
                    # Pair (1,2): X₁X₂ and Z₁Z₂
                    tab[1, 1] = 1
                    tab[2, 1] = 1
                    tab[7, 2] = 1
                    tab[8, 2] = 1
                    # Pair (3,4): X₃X₄ and Z₃Z₄
                    tab[3, 3] = 1
                    tab[4, 3] = 1
                    tab[9, 4] = 1
                    tab[10, 4] = 1
                    # Pair (5,6): X₅X₆ and Z₅Z₆
                    tab[5, 5] = 1
                    tab[6, 5] = 1
                    tab[11, 6] = 1
                    tab[12, 6] = 1

                    tab = TT(2, tab; m=6, storephase=true)

                    @test entanglement_entropy(tab, [1]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [1, 3]) == 2
                    @test entanglement_entropy(tab, [1, 3, 5]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4]) == 0
                    @test entanglement_entropy(tab, [2, 4, 6]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 1
                end
            end

            @testset "Qudit (d=3)" begin
                @testset "Small systems" begin
                    # Product state |00⟩ with generators Z₁ and Z₂.
                    tab_prod = zeros(Int, 5, 2)
                    tab_prod[3, 1] = 1  # Z₁
                    tab_prod[4, 2] = 1  # Z₂
                    tab_prod = TT(3, tab_prod; m=2, storephase=true)
                    @test entanglement_entropy(tab_prod, [1]) == 0

                    # Maximally entangled qutrit state with generators X₁X₂ and Z₁Z₂.
                    tab_bell = zeros(Int, 5, 2)
                    tab_bell[1, 1] = 1  # X₁
                    tab_bell[2, 1] = 1  # X₂
                    tab_bell[3, 2] = 1  # Z₁
                    tab_bell[4, 2] = 1  # Z₂
                    tab_bell = TT(3, tab_bell; m=2, storephase=true)
                    @test entanglement_entropy(tab_bell, [1]) == 1

                    # Mixed state (m < n).
                    tab_mixed = zeros(Int, 5, 2)
                    tab_mixed[3, 1] = 1  # Z₁ only
                    tab_mixed = TT(3, tab_mixed; m=1, storephase=true)
                    @test entanglement_entropy(tab_mixed, [1]) == 0
                    @test entanglement_entropy(tab_mixed, [2]) == 1
                    @test entanglement_entropy(tab_mixed, [1, 2]) == 1
                end

                @testset "Larger mixed systems" begin
                    # Qudits (d=3), start from maximally mixed and measure commuting operators.
                    tab = TT(3, 5; storephase=true) # m=0

                    # Maximally mixed: S(A) = |A|
                    @test entanglement_entropy(tab, [1]) == 1
                    @test entanglement_entropy(tab, [2, 4]) == 2
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 5

                    # Measure Z₁
                    measure!(tab, SinglePauli(1, 0, 1))
                    @test entanglement_entropy(tab, [1]) == 0
                    @test entanglement_entropy(tab, [2]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 1
                    @test entanglement_entropy(tab, [3, 4, 5]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 4

                    # Measure X₂
                    measure!(tab, SinglePauli(2, 1, 0))
                    @test entanglement_entropy(tab, [1]) == 0
                    @test entanglement_entropy(tab, [2]) == 0
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [2, 3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 3

                    # Measure X₄X₅
                    measure!(tab, DoublePauli(4, 1, 0, 5, 1, 0))
                    @test entanglement_entropy(tab, [4, 5]) == 1
                    @test entanglement_entropy(tab, [4]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 2

                    # Measure Z₄Z₅^{-1} (commutes with X₄X₅ for d=3)
                    measure!(tab, DoublePauli(4, 0, 1, 5, 0, 2))
                    @test entanglement_entropy(tab, [4, 5]) == 0
                    @test entanglement_entropy(tab, [4]) == 1
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [3, 4, 5]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 1

                    # test whether subsystem can be given in arbitrary order
                    @test entanglement_entropy(tab, [5, 4]) == 0
                    @test entanglement_entropy(tab, [5, 3, 4]) == 1
                    @test entanglement_entropy(tab, [4, 3, 2, 1, 5]) == 1
                end

                @testset "Larger pure systems" begin
                    # Qudits (d=3), three generalized Bell pairs on 6 qutrits.
                    tab = zeros(Int, 13, 6)
                    # Pair (1,2): X₁ X₂^{-1} and Z₁Z₂
                    tab[1, 1] = 1
                    tab[2, 1] = 2
                    tab[7, 2] = 1
                    tab[8, 2] = 1
                    # Pair (3,4): X₃ X₄^{-1} and Z₃Z₄
                    tab[3, 3] = 1
                    tab[4, 3] = 2
                    tab[9, 4] = 1
                    tab[10, 4] = 1
                    # Pair (5,6): X₅ X₆^{-1} and Z₅Z₆
                    tab[5, 5] = 1
                    tab[6, 5] = 2
                    tab[11, 6] = 1
                    tab[12, 6] = 1

                    tab = TT(3, tab; m=6, storephase=true)

                    @test entanglement_entropy(tab, [1]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [1, 3]) == 2
                    @test entanglement_entropy(tab, [1, 3, 5]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4]) == 0
                    @test entanglement_entropy(tab, [2, 4, 6]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 1
                end
            end
        end
    end
end
