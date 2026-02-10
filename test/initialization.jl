using Test
using QuditClifford

@testset "Initialization presets" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                # Mixed state
                tab_mixed = TT(2, 3; state=:mixed, storephase=true)
                @test tab_mixed.m == 0
                @test all(tab_mixed.stab .== 0)

                # Accept transposed tableau inputs (m × (2n+1))
                tab_row = reshape(Int[0, 1, 0], 1, 3)
                tab_row = TT(2, tab_row; m=1, storephase=true)
                @test tab_row.stab[2, 1] == 1

                # Product states in Z and X bases
                tab_z = TT(2, 2; state=:product, basis=:Z, storephase=true)
                @test tab_z.m == 2
                @test tab_z.stab[3, 1] == 1
                @test tab_z.stab[4, 2] == 1

                tab_x = TT(2, 2; state=:product, basis=:X, storephase=true)
                @test tab_x.stab[1, 1] == 1
                @test tab_x.stab[2, 2] == 1
                tab_x_alias = TT(2, 1; state=:X)
                @test tab_x_alias.stab[1, 1] == 1

                # Y basis (qubits only) requires phase row
                tab_y = TT(2, 1; state=:product, basis=:Y, storephase=true)
                @test tab_y.stab[1, 1] == 1
                @test tab_y.stab[2, 1] == 1
                @test tab_y.stab[3, 1] == 1

                # GHZ state
                tab_ghz = TT(2, 3; state=:ghz, storephase=true)
                @test tab_ghz.m == 3
                @test tab_ghz.stab[1, 1] == 1
                @test tab_ghz.stab[2, 1] == 1
                @test tab_ghz.stab[3, 1] == 1
                @test tab_ghz.stab[4, 2] == 1
                @test tab_ghz.stab[5, 2] == 1
                @test tab_ghz.stab[5, 3] == 1
                @test tab_ghz.stab[6, 3] == 1

                # reset! to presets
                tab_reset = TT(2, 2; state=:mixed, storephase=true)
                reset!(tab_reset; state=:product, basis=:Z)
                @test tab_reset.m == 2
                @test tab_reset.stab[3, 1] == 1
                @test tab_reset.stab[4, 2] == 1
                reset!(tab_reset, :ghz)
                @test tab_reset.stab[1, 1] == 1
                @test tab_reset.stab[2, 1] == 1
                @test tab_reset.stab[3, 2] == 1

                tab_nophase = TT(2, 1; state=:mixed, storephase=false)
                reset!(tab_nophase; state=:product, basis=:Y)

                # reset! should not allocate (after warm-up)
                reset!(tab_reset; state=:mixed)
                alloc_mixed = @allocated reset!(tab_reset; state=:mixed)
                @test alloc_mixed == 0

                basis_vec = [:Z, :X]
                reset!(tab_reset; state=:product, basis=basis_vec)
                alloc_prod = @allocated reset!(tab_reset; state=:product, basis=basis_vec)
                @test alloc_prod == 0

                reset!(tab_reset, :ghz)
                alloc_ghz = @allocated reset!(tab_reset, :ghz)
                @test alloc_ghz == 0
            end

            @testset "Qudit (d=3)" begin
                tab_mixed = TT(3, 2; state=:mixed, storephase=true)
                @test tab_mixed.m == 0
                @test all(tab_mixed.stab .== 0)

                tab_z = TT(3, 2; state=:product, basis=:Z, storephase=true)
                @test tab_z.m == 2
                @test tab_z.stab[3, 1] == 1
                @test tab_z.stab[4, 2] == 1

                tab_x = TT(3, 2; state=:product, basis=:X, storephase=true)
                @test tab_x.stab[1, 1] == 1
                @test tab_x.stab[2, 2] == 1
                tab_x_alias = TT(3, 1; state=:X)
                @test tab_x_alias.stab[1, 1] == 1

                tab_ghz = TT(3, 3; state=:ghz, storephase=true)
                @test tab_ghz.m == 3
                @test tab_ghz.stab[1, 1] == 1
                @test tab_ghz.stab[2, 1] == 1
                @test tab_ghz.stab[3, 1] == 1
                @test tab_ghz.stab[4, 2] == 1
                @test tab_ghz.stab[5, 2] == 2 # d-1 for qudits
                @test tab_ghz.stab[5, 3] == 1
                @test tab_ghz.stab[6, 3] == 2 # d-1 for qudits

                tab_reset = TT(3, 2; state=:mixed, storephase=true)
                reset!(tab_reset; state=:product, basis=:Z)
                @test tab_reset.m == 2
                reset!(tab_reset, :ghz)
                @test tab_reset.stab[1, 1] == 1
                @test tab_reset.stab[2, 1] == 1
                @test tab_reset.stab[3, 2] == 1
            end
        end
    end
end
