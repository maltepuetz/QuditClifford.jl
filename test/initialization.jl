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

@testset "Matrix construction and validation" begin
    for (label, TT) in [
        ("StabilizerTableau", StabilizerTableau),
        ("DestabilizerTableau", DestabilizerTableau),
    ]
        @testset "$label" begin
            # A compact active-generator matrix is padded to the inferred capacity.
            partial = reshape(Int[0, 0, 1, 0, 0], 5, 1) # Z₁ on a two-qudit system
            tab = TT(3, partial)
            @test tab.n == 2
            @test tab.m == 1
            @test size(tab.stab) == (5, 2)
            @test tab.stab[:, 1] == partial[:, 1]
            @test all(iszero, tab.stab[:, 2])

            full = zeros(Int, 5, 2)
            full[3, 1] = 1
            full[4, 2] = 1
            full_tab = TT(3, full)
            @test full_tab.n == full_tab.m == 2
            @test full_tab.stab == full

            # Positional state construction is an exact alias of the keyword API.
            positional = TT(2, 1, :X)
            keyword = TT(2, 1; state=:X)
            @test positional.stab == keyword.stab
            @test positional.m == keyword.m == 1

            @test_throws ArgumentError TT(2, zeros(Int, 4, 3))
            @test_throws ArgumentError TT(2, zeros(Int, 4, 2); storephase=true)
            @test_throws ArgumentError TT(2, zeros(Int, 5, 2); m=-1)
            @test_throws ArgumentError TT(2, zeros(Int, 5, 2); m=3)
            @test_throws ArgumentError TT(2, zeros(Int, 6, 2); m=1, storephase=false)
            @test_throws ArgumentError TT(4, 1; state=:mixed)
        end
    end
end

@testset "State aliases, heterogeneous bases, and validation" begin
    for (label, TT) in [
        ("StabilizerTableau", StabilizerTableau),
        ("DestabilizerTableau", DestabilizerTableau),
    ]
        @testset "$label" begin
            tab = TT(2, 3; state=:product, basis=(:x, :Y, :z))
            @test tab.stab[:, 1] == Int[1, 0, 0, 0, 0, 0, 0]
            @test tab.stab[:, 2] == Int[0, 1, 0, 0, 1, 0, 1]
            @test tab.stab[:, 3] == Int[0, 0, 0, 0, 0, 1, 0]

            mixed_alias = TT(2, 1; state=:maximally_mixed, basis=:z)
            @test mixed_alias.m == 0
            @test all(iszero, mixed_alias.stab)

            @test_throws ArgumentError TT(2, 1; state=:X, basis=:Y)
            @test_throws ArgumentError TT(2, 1; state=:X, basis=[:X])
            @test_throws ArgumentError TT(2, 1; state=:mixed, basis=[:Z])
            @test_throws ArgumentError TT(2, 1; state=:ghz, basis=:X)
            @test_throws ArgumentError TT(2, 1; state=:unknown)
            @test_throws ArgumentError TT(2, 1; state=:product, basis=:unknown)
            @test_throws ArgumentError TT(2, 2; state=:product, basis=(:X,))
            @test_throws ArgumentError TT(2, 2; state=:product, basis=(:X, 1))
            @test_throws ArgumentError TT(2, 1; state=:product, basis=1)
        end
    end

    # The preset builder has its own defensive contract in addition to the
    # public normalization layer.
    raw = zeros(Int, 3, 1)
    @test_throws ArgumentError QuditClifford._preset_tableau!(
        raw,
        2,
        1,
        :unknown,
        nothing,
        true,
    )
end

@testset "Modular inversion strategies" begin
    @test_throws ArgumentError QuditClifford.PrecomputedInvMod(4)

    precomputed = QuditClifford.PrecomputedInvMod(Int[1, 2])
    just_in_time = QuditClifford.JustInTimeInvMod()
    @test precomputed(1, 3) == 1
    @test precomputed(2, 3) == 2
    @test just_in_time(2, 3) == 2

    tab = StabilizerTableau(3, 2; state=:ghz, inversemod=just_in_time)
    canonicalize!(tab)
    @test tab.iscanonical
    @test is_pure(tab)
end

@testset "Tableau metadata and display" begin
    mixed = StabilizerTableau(2, 2; state=:mixed)
    @test QuditClifford.ngens(mixed) == 0
    @test QuditClifford.is_mixed(mixed)
    reset!(mixed; state=:product, basis=:Z)
    @test QuditClifford.ngens(mixed) == 2
    @test !QuditClifford.is_mixed(mixed)

    stab_mixed_text = sprint(show, StabilizerTableau(2, 2; state=:mixed))
    @test occursin("Stabilizer Tableau:", stab_mixed_text)
    @test occursin("no generators", stab_mixed_text)

    stab_one_text = sprint(show, StabilizerTableau(2, 1; state=:product, basis=:Z))
    @test occursin("    Tableau:", stab_one_text)
    @test occursin("X", stab_one_text)
    @test occursin("Z", stab_one_text)

    stab_two_text = sprint(show, StabilizerTableau(2, 2; state=:product, basis=:Z))
    @test occursin(" X     Z ", stab_two_text)

    partial = reshape(Int[10, 0, 0, 0, 0, 0, 0], 7, 1)
    stab_partial_text = sprint(show, StabilizerTableau(11, partial))
    @test occursin("Mixed tableau", stab_partial_text)
    @test occursin("10", stab_partial_text)

    stab_large_text = sprint(show, StabilizerTableau(2, 30; state=:mixed))
    @test occursin("too large to display", stab_large_text)

    destab_mixed_text = sprint(show, DestabilizerTableau(2, 2; state=:mixed))
    @test occursin("Destabilizer Tableau:", destab_mixed_text)
    @test occursin("no generators", destab_mixed_text)

    destab_one_text = sprint(show, DestabilizerTableau(2, 1; state=:product, basis=:Z))
    @test occursin("Stabilizers:", destab_one_text)
    @test occursin("Destabilizers:", destab_one_text)

    destab_two_text = sprint(show, DestabilizerTableau(2, 2; state=:product, basis=:Z))
    @test occursin(" X     Z ", destab_two_text)
    @test occursin("Destabilizers:", destab_two_text)

    destab_partial_text = sprint(show, DestabilizerTableau(11, partial))
    @test occursin("Mixed tableau", destab_partial_text)
    @test occursin("10", destab_partial_text)

    destab_large_text = sprint(show, DestabilizerTableau(2, 30; state=:mixed))
    @test occursin("too large to display", destab_large_text)
end

# Preset construction writes values that are already reduced, so it needs no
# separate mod pass over the (2n + storephase) × n matrix. Nothing else
# guarantees that: the GHZ generators are the natural place to write a literal
# -1, and only an assertion like this catches it once the pass is gone.
@testset "Preset tableaux are built already reduced" begin
    for TT in (StabilizerTableau, DestabilizerTableau),
        d in (2, 3, 5, 7), n in (1, 3, 4), storephase in (true, false)

        specs = Any[(:mixed, :Z), (:ghz, :Z),
                    (:product, :X), (:product, :Y), (:product, :Z)]
        n >= 3 && push!(specs, (:product, [:X, :Y, :Z][mod1.(1:n, 3)]))

        for (state, basis) in specs
            fresh = TT(d, n; state=state, basis=basis, storephase=storephase)
            after_reset = reset!(TT(d, n; state=:mixed, storephase=storephase);
                                 state=state, basis=basis)

            for tab in (fresh, after_reset)
                @test all(v -> 0 <= v < d, @view tab.stab[1:(2n), :])
                if storephase
                    dphase = QuditClifford.phase_modulus(d)
                    @test all(v -> 0 <= v < dphase, @view tab.stab[2n + 1, :])
                end
            end
        end
    end
end
