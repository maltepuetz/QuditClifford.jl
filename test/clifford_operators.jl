using QuditClifford
using Test
using Random

const QC = QuditClifford

# Build a dense prepared view from P1 tuple gate data, so the dense evaluators
# can be compared against the already-validated tuple ones before any public
# type exists. `F[row, col] = tuple_F[col][row]` -- the tuple form is columns.
function dense_prep_from_gate(g, d::Int; storephase::Bool = true,
                              force_safe::Bool = false)
    jit = QC.JustInTimeInvMod()
    targets, tF, ta = QC._clifford_data(g, d, jit)
    k = length(targets); S = 2k
    F = Matrix{Int}(undef, S, S)
    for col in 1:S, row in 1:S
        F[row, col] = tF[col][row]
    end
    a = collect(Int, ta)
    fast = !force_safe && QC.clifford_fast_dots(S, d)
    D = collect(Int, QC._image_xdotz(tF, k, d, fast))
    inv2 = d == 2 ? 0 : jit(2, d)
    return QC.PreparedDenseClifford(collect(Int, targets), F, a, D,
                                    zeros(Int, S), zeros(Int, S), zeros(Int, k),
                                    k, d, QC.phase_modulus(d), inv2, fast,
                                    storephase)
end

@testset "Dense evaluators agree with the tuple evaluators" begin
    jit = QC.JustInTimeInvMod()
    for d in (2, 3, 5), force_safe in (false, true)
        gates = Any[Fourier(1), Phase(1), PauliGate(1, 1, 1),
                    SUM(1, 2, 1), CPhase(1, 2, 1), SWAP(1, 2)]
        push!(gates, Multiplier(1, d == 2 ? 1 : 2))
        for g in gates
            targets, tF, ta = QC._clifford_data(g, d, jit)
            k = length(targets); S = 2k
            fast = !force_safe && QC.clifford_fast_dots(S, d)
            tprep = QC._prepare(g, d, jit, true; force_safe = force_safe)
            dprep = dense_prep_from_gate(g, d; force_safe = force_safe)
            # Exhaustive over local coordinate vectors for both k = 1 and k = 2.
            vectors = k == 1 ? [[x, z] for x in 0:(d - 1) for z in 0:(d - 1)] :
                               [[a1, a2, b1, b2] for a1 in 0:(d - 1), a2 in 0:(d - 1),
                                b1 in 0:(d - 1), b2 in 0:(d - 1)] |> vec
            for vv in vectors
                tv = Tuple(vv)
                tout = QC._matvec(tprep, tv)
                dout = QC._matvec_prepared!(dprep, copyto!(dprep.v, vv))
                @test collect(dout) == collect(tout)
                @test QC._phase(dprep, dprep.v, dprep.vout) == QC._phase(tprep, tv, tout)
            end
        end
    end
end

function dense_apply_allocations(tab, prep)
    for _ in 1:3
        QC._apply_prepared!(tab, prep)
    end
    return @allocated QC._apply_prepared!(tab, prep)
end

# The dense backing through the SHARED pipeline, before any public type is built
# on it: dense gather/scatter, both destabilizer hooks,
# and allocation behavior when applying an already-built preparation.
@testset "Dense preparations run through the shared pipeline without allocating" begin
    jit = QC.JustInTimeInvMod()
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5), sp in (false, true)
        for g in (Fourier(2), Phase(2), SUM(2, 5, 1), CPhase(2, 5, 1), SWAP(2, 5))
            tuple_tab = TT(d, 6; state = :ghz, storephase = sp)
            dense_tab = deepcopy(tuple_tab)
            QC._apply_prepared!(tuple_tab, QC._prepare(g, d, jit, sp))
            QC._apply_prepared!(dense_tab, dense_prep_from_gate(g, d; storephase = sp))
            @test dense_tab.stab == tuple_tab.stab
            if TT === DestabilizerTableau
                @test dense_tab.destab == tuple_tab.destab
                @test dense_tab.xdotz_cache == tuple_tab.xdotz_cache
            end
        end
        tab = TT(d, 12; state = :ghz, storephase = sp)
        prep = dense_prep_from_gate(Fourier(1), d; storephase = sp)
        dense_apply_allocations(tab, prep)
        @test dense_apply_allocations(tab, prep) == 0
    end
end

# Recreate the borrowed wrapper inside the measured application, from arrays
# already owned by a fixture. This is the boundary Task 2 intends to measure.
function fresh_dense_apply!(tab, fixture)
    prep = QC.PreparedDenseClifford(fixture.targets, fixture.F, fixture.a,
        fixture.D, fixture.v, fixture.vout, fixture.zpref, fixture.k,
        fixture.d, fixture.p, fixture.inv2, fixture.fast, fixture.storephase)
    return QC._apply_prepared!(tab, prep)
end
function fresh_dense_apply_allocations(tab, fixture)
    for _ in 1:3
        fresh_dense_apply!(tab, fixture)
    end
    return @allocated fresh_dense_apply!(tab, fixture)
end
@testset "Fresh dense wrapper and application allocate nothing" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5),
        sp in (false, true), fs in (false, true)
        tab = TT(d, 12; state = :ghz, storephase = sp)
        fixture = dense_prep_from_gate(Fourier(1), d; storephase = sp, force_safe = fs)
        fresh_dense_apply_allocations(tab, fixture)
        @test fresh_dense_apply_allocations(tab, fixture) == 0
    end
end
