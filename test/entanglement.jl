using QuditClifford
using Test

@testset "Entanglement entropy" begin
    @testset "Qubits (d=2)" begin
        # Product state |00⟩ with generators Z₁ and Z₂.
        tab_prod = zeros(Int, 5, 2)
        tab_prod[3, 1] = 1  # Z₁
        tab_prod[4, 2] = 1  # Z₂
        stab_prod = StabilizerTableau(2, 2, tab_prod; m=2, storephase=true)
        @test entanglement_entropy(stab_prod, [1]) == 0

        # Bell state |Φ+⟩ with generators X₁X₂ and Z₁Z₂.
        tab_bell = zeros(Int, 5, 2)
        tab_bell[1, 1] = 1  # X₁
        tab_bell[2, 1] = 1  # X₂
        tab_bell[3, 2] = 1  # Z₁
        tab_bell[4, 2] = 1  # Z₂
        stab_bell = StabilizerTableau(2, 2, tab_bell; m=2, storephase=true)
        @test entanglement_entropy(stab_bell, [1]) == 1

        # Mixed state (m < n).
        tab_mixed = zeros(Int, 5, 2)
        tab_mixed[3, 1] = 1  # Z₁ only
        stab_mixed = StabilizerTableau(2, 2, tab_mixed; m=1, storephase=true)
        @test entanglement_entropy(stab_mixed, [1]) == 0
        @test entanglement_entropy(stab_mixed, [2]) == 1
        @test entanglement_entropy(stab_mixed, [1, 2]) == 1
    end

    @testset "Qubits (d=2) --- larger mixed systems" begin
        # Qubits (d=2), start from maximally mixed and measure commuting operators.
        stab = StabilizerTableau(2, 5; storephase=true) # m=0

        # Maximally mixed: S(A) = |A|
        @test entanglement_entropy(stab, [1]) == 1
        @test entanglement_entropy(stab, [2, 4]) == 2
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 5

        # Measure Z₁
        measure!(stab, SinglePauli(1, 0, 1))
        @test entanglement_entropy(stab, [1]) == 0
        @test entanglement_entropy(stab, [2]) == 1
        @test entanglement_entropy(stab, [1, 2]) == 1
        @test entanglement_entropy(stab, [3, 4, 5]) == 3
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 4

        # Measure X₂
        measure!(stab, SinglePauli(2, 1, 0))
        @test entanglement_entropy(stab, [1]) == 0
        @test entanglement_entropy(stab, [2]) == 0
        @test entanglement_entropy(stab, [3]) == 1
        @test entanglement_entropy(stab, [1, 2]) == 0
        @test entanglement_entropy(stab, [2, 3]) == 1
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 3

        # Measure X₄X₅
        measure!(stab, DoublePauli(4, 1, 0, 5, 1, 0))
        @test entanglement_entropy(stab, [4, 5]) == 1
        @test entanglement_entropy(stab, [4]) == 1
        @test entanglement_entropy(stab, [1, 2]) == 0
        @test entanglement_entropy(stab, [3]) == 1
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 2

        # Measure Z₄Z₅ (now qudits 4&5 become pure Bell pair)
        measure!(stab, DoublePauli(4, 0, 1, 5, 0, 1))
        @test entanglement_entropy(stab, [4, 5]) == 0
        @test entanglement_entropy(stab, [4]) == 1
        @test entanglement_entropy(stab, [3]) == 1
        @test entanglement_entropy(stab, [3, 4, 5]) == 1
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 1

        # test whether subsystem can be given in arbitrary order
        @test entanglement_entropy(stab, [5, 4]) == 0
        @test entanglement_entropy(stab, [5, 3, 4]) == 1
        @test entanglement_entropy(stab, [4, 3, 2, 1, 5]) == 1
    end

    @testset "Qudits (d=3)" begin
        # Product state |00⟩ with generators Z₁ and Z₂.
        tab_prod = zeros(Int, 5, 2)
        tab_prod[3, 1] = 1  # Z₁
        tab_prod[4, 2] = 1  # Z₂
        stab_prod = StabilizerTableau(3, 2, tab_prod; m=2, storephase=true)
        @test entanglement_entropy(stab_prod, [1]) == 0

        # Maximally entangled qutrit state with generators X₁X₂ and Z₁Z₂.
        tab_bell = zeros(Int, 5, 2)
        tab_bell[1, 1] = 1  # X₁
        tab_bell[2, 1] = 1  # X₂
        tab_bell[3, 2] = 1  # Z₁
        tab_bell[4, 2] = 1  # Z₂
        stab_bell = StabilizerTableau(3, 2, tab_bell; m=2, storephase=true)
        @test entanglement_entropy(stab_bell, [1]) == 1

        # Mixed state (m < n).
        tab_mixed = zeros(Int, 5, 2)
        tab_mixed[3, 1] = 1  # Z₁ only
        stab_mixed = StabilizerTableau(3, 2, tab_mixed; m=1, storephase=true)
        @test entanglement_entropy(stab_mixed, [1]) == 0
        @test entanglement_entropy(stab_mixed, [2]) == 1
        @test entanglement_entropy(stab_mixed, [1, 2]) == 1
    end

    @testset "Qudits (d=3) --- larger mixed systems" begin
        # Qudits (d=3), start from maximally mixed and measure commuting operators.
        stab = StabilizerTableau(3, 5; storephase=true) # m=0

        # Maximally mixed: S(A) = |A|
        @test entanglement_entropy(stab, [1]) == 1
        @test entanglement_entropy(stab, [2, 4]) == 2
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 5

        # Measure Z₁
        measure!(stab, SinglePauli(1, 0, 1))
        @test entanglement_entropy(stab, [1]) == 0
        @test entanglement_entropy(stab, [2]) == 1
        @test entanglement_entropy(stab, [1, 2]) == 1
        @test entanglement_entropy(stab, [3, 4, 5]) == 3
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 4

        # Measure X₂
        measure!(stab, SinglePauli(2, 1, 0))
        @test entanglement_entropy(stab, [1]) == 0
        @test entanglement_entropy(stab, [2]) == 0
        @test entanglement_entropy(stab, [3]) == 1
        @test entanglement_entropy(stab, [1, 2]) == 0
        @test entanglement_entropy(stab, [2, 3]) == 1
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 3

        # Measure X₄X₅
        measure!(stab, DoublePauli(4, 1, 0, 5, 1, 0))
        @test entanglement_entropy(stab, [4, 5]) == 1
        @test entanglement_entropy(stab, [4]) == 1
        @test entanglement_entropy(stab, [1, 2]) == 0
        @test entanglement_entropy(stab, [3]) == 1
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 2

        # Measure Z₄Z₅^{-1} (commutes with X₄X₅ for d=3)
        measure!(stab, DoublePauli(4, 0, 1, 5, 0, 2))
        @test entanglement_entropy(stab, [4, 5]) == 0
        @test entanglement_entropy(stab, [4]) == 1
        @test entanglement_entropy(stab, [3]) == 1
        @test entanglement_entropy(stab, [3, 4, 5]) == 1
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 1

        # test whether subsystem can be given in arbitrary order
        @test entanglement_entropy(stab, [5, 4]) == 0
        @test entanglement_entropy(stab, [5, 3, 4]) == 1
        @test entanglement_entropy(stab, [4, 3, 2, 1, 5]) == 1
    end

    @testset "Qubits (d=2) --- larger pure systems" begin
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

        stab = StabilizerTableau(2, 6, tab; m=6, storephase=true)

        @test entanglement_entropy(stab, [1]) == 1
        @test entanglement_entropy(stab, [1, 2]) == 0
        @test entanglement_entropy(stab, [1, 3]) == 2
        @test entanglement_entropy(stab, [1, 3, 5]) == 3
        @test entanglement_entropy(stab, [1, 2, 3]) == 1
        @test entanglement_entropy(stab, [1, 2, 3, 4]) == 0
        @test entanglement_entropy(stab, [2, 4, 6]) == 3
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 1
        
    end
    @testset "Qudits (d=3) --- larger pure systems" begin
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

        stab = StabilizerTableau(3, 6, tab; m=6, storephase=true)

        @test entanglement_entropy(stab, [1]) == 1
        @test entanglement_entropy(stab, [1, 2]) == 0
        @test entanglement_entropy(stab, [1, 3]) == 2
        @test entanglement_entropy(stab, [1, 3, 5]) == 3
        @test entanglement_entropy(stab, [1, 2, 3]) == 1
        @test entanglement_entropy(stab, [1, 2, 3, 4]) == 0
        @test entanglement_entropy(stab, [2, 4, 6]) == 3
        @test entanglement_entropy(stab, [1, 2, 3, 4, 5]) == 1
    end
end
