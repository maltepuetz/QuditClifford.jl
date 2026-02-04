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

        # Mixed state (m < n) should throw for entanglement_entropy.
        tab_mixed = zeros(Int, 5, 2)
        tab_mixed[3, 1] = 1  # Z₁ only
        stab_mixed = StabilizerTableau(2, 2, tab_mixed; m=1, storephase=true)
        @test_throws ArgumentError entanglement_entropy(stab_mixed, [1])
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

        # Mixed state (m < n) should throw for entanglement_entropy.
        tab_mixed = zeros(Int, 5, 2)
        tab_mixed[3, 1] = 1  # Z₁ only
        stab_mixed = StabilizerTableau(3, 2, tab_mixed; m=1, storephase=true)
        @test_throws ArgumentError entanglement_entropy(stab_mixed, [1])
    end
end
