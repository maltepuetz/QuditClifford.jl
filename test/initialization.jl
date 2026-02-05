using Test
using QuditClifford

@testset "Initialization presets" begin
    # Mixed state
    stab_mixed = StabilizerTableau(2, 3; state=:mixed, storephase=true)
    @test stab_mixed.m == 0
    @test all(stab_mixed.tableau .== 0)

    # Accept transposed tableau inputs (m × (2n+1))
    tab_row = reshape(Int[0, 1, 0], 1, 3)
    stab_row = StabilizerTableau(2, tab_row; m=1, storephase=true)
    @test stab_row.tableau[2, 1] == 1

    # Product states in Z and X bases
    stab_z = StabilizerTableau(2, 2; state=:product, basis=:Z, storephase=true)
    @test stab_z.m == 2
    @test stab_z.tableau[3, 1] == 1
    @test stab_z.tableau[4, 2] == 1

    stab_x = StabilizerTableau(2, 2; state=:product, basis=:X, storephase=true)
    @test stab_x.tableau[1, 1] == 1
    @test stab_x.tableau[2, 2] == 1
    stab_x_alias = StabilizerTableau(2, 1; state=:X)
    @test stab_x_alias.tableau[1, 1] == 1

    # Y basis (qubits only) requires phase row
    stab_y = StabilizerTableau(2, 1; state=:product, basis=:Y, storephase=true)
    @test stab_y.tableau[1, 1] == 1
    @test stab_y.tableau[2, 1] == 1
    @test stab_y.tableau[3, 1] == 1
    @test_throws ArgumentError StabilizerTableau(2, 1; state=:product, basis=:Y, storephase=false)

    # GHZ state
    stab_ghz = StabilizerTableau(2, 3; state=:ghz, storephase=true)
    @test stab_ghz.m == 3
    @test stab_ghz.tableau[1, 1] == 1
    @test stab_ghz.tableau[2, 1] == 1
    @test stab_ghz.tableau[3, 1] == 1
    @test stab_ghz.tableau[4, 2] == 1
    @test stab_ghz.tableau[5, 2] == 1
    @test stab_ghz.tableau[5, 3] == 1
    @test stab_ghz.tableau[6, 3] == 1

    # reset! to presets
    stab_reset = StabilizerTableau(2, 2; state=:mixed, storephase=true)
    reset!(stab_reset; state=:product, basis=:Z)
    @test stab_reset.m == 2
    @test stab_reset.tableau[3, 1] == 1
    @test stab_reset.tableau[4, 2] == 1
    reset!(stab_reset, :ghz)
    @test stab_reset.tableau[1, 1] == 1
    @test stab_reset.tableau[2, 1] == 1
    @test stab_reset.tableau[3, 2] == 1

    stab_nophase = StabilizerTableau(2, 1; state=:mixed, storephase=false)
    @test_throws ArgumentError reset!(stab_nophase; state=:product, basis=:Y)

    # reset! should not allocate (after warm-up)
    reset!(stab_reset; state=:mixed)
    alloc_mixed = @allocated reset!(stab_reset; state=:mixed)
    @test alloc_mixed == 0

    basis_vec = [:Z, :X]
    reset!(stab_reset; state=:product, basis=basis_vec)
    alloc_prod = @allocated reset!(stab_reset; state=:product, basis=basis_vec)
    @test alloc_prod == 0

    reset!(stab_reset, :ghz)
    alloc_ghz = @allocated reset!(stab_reset, :ghz)
    @test alloc_ghz == 0
end
