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
    end
end
