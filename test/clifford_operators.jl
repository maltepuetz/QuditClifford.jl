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

# Snapshots compare semantic data, derived context and scratch independently.
const OPERATOR_ARRAY_FIELDS = (:targets, :F, :a, :image_xdotz, :v, :vout, :zpref)
operator_snapshot(U) = (U.d, U.inv2, U.fast,
    map(f -> copy(getfield(U, f)), OPERATOR_ARRAY_FIELDS))
function assert_independent(U, V)
    for f in OPERATOR_ARRAY_FIELDS, g in OPERATOR_ARRAY_FIELDS
        @test !Base.mightalias(getfield(U, f), getfield(V, g))
    end
    for (i, f) in enumerate(OPERATOR_ARRAY_FIELDS), g in OPERATOR_ARRAY_FIELDS[(i + 1):end]
        @test !Base.mightalias(getfield(U, f), getfield(U, g))
    end
end

# Identity Clifford on k targets: F = I_{2k}, all raw phases zero. Valid at
# every prime d and trivially symplectic.
eye_int(S::Int) = [i == j ? 1 : 0 for i in 1:S, j in 1:S]
identity_data(k::Int) = (eye_int(2k), zeros(Int, 2k))

# A minimal non-one-based array. Testing the offset rejection this way avoids
# adding an OffsetArrays dependency to the tracked test/Project.toml.
struct ZeroBased{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
Base.size(z::ZeroBased) = size(z.data)
Base.axes(z::ZeroBased) = map(ax -> 0:(length(ax) - 1), axes(z.data))
Base.getindex(z::ZeroBased{T,N}, I::Vararg{Int,N}) where {T,N} =
    z.data[map(i -> i + 1, I)...]

# Report huge shapes without storage. Any element read is a test failure.
struct UnreadableIntArray{N} <: AbstractArray{Int,N}
    dims::NTuple{N,Int}
end
Base.size(a::UnreadableIntArray) = a.dims
Base.getindex(::UnreadableIntArray, I...) = error("input traversed before rejection")

struct BrokenAxes <: AbstractVector{Int} end
Base.size(::BrokenAxes) = (1,)
Base.axes(::BrokenAxes) = error("caller-defined axes failure")

@testset "Raw construction validates and normalizes" begin
    for d in (2, 3, 5)
        F, a = identity_data(2)
        U = CliffordOperator(d, [1, 3], F, a)
        @test U.d == d
        @test U.targets == [1, 3]
        @test U.F == F
        @test U.a == a
        @test U.image_xdotz == zeros(Int, 4)
        @test U.inv2 == (d == 2 ? 0 : invmod(2, d))
        @test length(U.v) == 4 && length(U.vout) == 4 && length(U.zpref) == 2
        @test U.v !== U.vout
        # The public type must not expose Julia's unchecked full-field
        # constructor. Only the internal owned-data boundary may bypass the
        # copying/validation route, even for otherwise valid input arrays.
        fields = map(f -> getfield(U, f), fieldnames(CliffordOperator))
        @test_throws MethodError CliffordOperator(fields...)

        # Entries normalize: identity + d is still the identity, and raw
        # phases reduce mod p.
        @test CliffordOperator(d, [1, 3], F .+ d, a).F == F
        @test CliffordOperator(d, [1, 3], F, a .+ QC.phase_modulus(d)).a == a

        # Targets must be positive and distinct.
        @test_throws ArgumentError CliffordOperator(d, [1, 1], F, a)
        @test_throws ArgumentError CliffordOperator(d, [0, 2], F, a)
        # Shapes.
        @test_throws ArgumentError CliffordOperator(d, [1, 3], F, zeros(Int, 3))
        @test_throws ArgumentError CliffordOperator(d, [1, 3], zeros(Int, 3, 4), a)
        # Non-prime dimension.
        @test_throws ArgumentError CliffordOperator(4, [1, 3], F, a)
        # Non-symplectic F.
        bad = copy(F); bad[1, 2] = 1; bad[2, 1] = 1
        @test_throws ArgumentError CliffordOperator(d, [1, 3], bad, a)
        @test CliffordOperator(d, [1, 3], bad, a; check = false) isa CliffordOperator
    end

    # typemin(Int) is reduced BEFORE any conversion or arithmetic. `check=false`
    # because a constant matrix is not symplectic; normalization still runs.
    for d in (2, 3, 5)
        W = CliffordOperator(d, [1], fill(typemin(Int), 2, 2), fill(typemin(Int), 2);
                             check = false)
        @test all(==(mod(typemin(Int), d)), W.F)
        @test all(==(mod(typemin(Int), QC.phase_modulus(d))), W.a)
    end

    # Qubit parity: a[i] must match image_xdotz[i] mod 2.
    F, a = identity_data(1)
    @test_throws ArgumentError CliffordOperator(2, [1], F, [1, 0])
    @test CliffordOperator(2, [1], F, [1, 0]; check = false) isa CliffordOperator
    # Qubit construction must not evaluate invmod(2, 2).
    @test CliffordOperator(2, [1], F, a).inv2 == 0

    # Empty support is the identity and is valid at every accepted d.
    for d in (2, 3, 5)
        E = CliffordOperator(d, Int[], zeros(Int, 0, 0), Int[])
        @test E.targets == Int[] && size(E.F) == (0, 0) && isempty(E.a)
    end

    # One-based axes are a structural condition, enforced for either `check`.
    # A correctly sized zero-based vector passes every length test and then
    # invalidates the constructor's `1:k` loops.
    F1, a1 = identity_data(1)
    for check in (false, true)
        for args in ((ZeroBased([1]), F1, a1),
                     ([1], ZeroBased(F1), a1), ([1], F1, ZeroBased(a1)))
            @test_throws ArgumentError CliffordOperator(3, args...; check)
        end
        @test_throws ArgumentError CliffordOperator(3, [big(typemax(Int)) + 1], F1, a1; check)
        @test_throws ArgumentError CliffordOperator(3, [-1], F1, a1; check)
        @test_throws ArgumentError CliffordOperator(3, [1, 1], eye_int(4), zeros(Int, 4); check)
        @test_throws ArgumentError CliffordOperator(4, [1], F1, a1; check)
        @test_throws ArgumentError CliffordOperator(3, [1], F1, [0]; check)
        @test_throws ErrorException CliffordOperator(3, BrokenAxes(), F1, a1; check)

        # Correct logical shape, representable element count, impossible bytes.
        kb = isqrt(typemax(Int) ÷ sizeof(Int)) ÷ 2 + 1
        sb = 2kb
        @test big(sb)^2 <= typemax(Int) < big(sb)^2 * sizeof(Int)
        @test_throws ArgumentError CliffordOperator(3, UnreadableIntArray((kb,)),
            UnreadableIntArray((sb, sb)), UnreadableIntArray((sb,)); check)
        # Impossible element counts and malformed ordinary shapes also precede access.
        for kh in (typemax(Int), isqrt(typemax(Int)) ÷ 2 + 1)
            @test_throws ArgumentError CliffordOperator(3, UnreadableIntArray((kh,)),
                F1, a1; check)
        end
        @test_throws ArgumentError CliffordOperator(3, UnreadableIntArray((2,)), F1, a1; check)
    end
    for d in (2, 3, 5)
        b = big(typemax(Int))^3
        Fbig = big.(F1) .+ d * b
        abig = big.(a1) .- QC.phase_modulus(d) * b
        W = CliffordOperator(d, BigInt[1], Fbig, abig)
        @test W.F == F1 && W.a == a1
    end

    # One-based non-contiguous views, transposes and ranges ARE accepted, and
    # materialize in the documented order.
    padded = [1 0 9; 0 1 9; 9 9 9]
    @test CliffordOperator(3, [1], view(padded, 1:2, 1:2), a1).F == F1
    # A nonsymmetric matrix makes accidental transpose/orientation bugs visible.
    shear = [1 0; 1 1]
    @test CliffordOperator(3, [1], transpose(shear), a1).F == [1 1; 0 1]
    @test CliffordOperator(3, view([1, 99], 1:2:2), shear,
                           view([0, 99, 0, 99], 1:2:4)).F == shear
    @test CliffordOperator(3, 1:1, F1, a1).targets == [1]
end

@testset "Materialization, copying and value behaviour" begin
    jit = QC.JustInTimeInvMod()
    for d in (2, 3, 5)
        gates = Any[Fourier(1), Phase(1), PauliGate(1, 1, 1),
                    SUM(1, 2, 1), CPhase(1, 2, 1), SWAP(1, 2)]
        push!(gates, Multiplier(1, d == 2 ? 1 : 2))
        for g in gates
            U = CliffordOperator(g, d)
            targets, tF, ta = QC._clifford_data(g, d, jit)
            S = 2 * length(targets)
            @test U.d == d
            @test U.targets == collect(Int, targets)
            @test U.a == collect(Int, ta)
            for col in 1:S, row in 1:S
                @test U.F[row, col] == tF[col][row]
            end
            # Materialization must satisfy the public constructor's own checks.
            @test CliffordOperator(d, U.targets, U.F, U.a) == U
        end
    end
    # Modulus-dependent gate validation still fires during materialization.
    @test_throws ArgumentError CliffordOperator(Multiplier(1, 3), 3)
    @test_throws ArgumentError CliffordOperator(Fourier(1), 4)

    # Copying is independent in semantic data, cache AND scratch.
    U = CliffordOperator(Fourier(1), 3)
    C = copy(U)
    @test C == U && isequal(C, U) && hash(C) == hash(U)
    for f in (:targets, :F, :a, :image_xdotz, :v, :vout, :zpref)
        @test getfield(C, f) !== getfield(U, f)
    end
    M = CliffordOperator(U, 3)
    @test M == U
    assert_independent(M, U)
    assert_independent(C, U)
    @test_throws ArgumentError CliffordOperator(U, 5)

    # Copying an unchecked operator must not reinstate the skipped check.
    @test_throws ArgumentError CliffordOperator(3, [1], zeros(Int, 2, 2), zeros(Int, 2))
    bad = CliffordOperator(3, [1], zeros(Int, 2, 2), zeros(Int, 2); check = false)
    @test copy(bad) == bad
    @test CliffordOperator(bad, 3) == bad
    badparity = CliffordOperator(2, [1], eye_int(2), [1, 0]; check = false)
    @test copy(badparity) == badparity

    # Equality and hashing ignore cache and scratch.
    A = CliffordOperator(Phase(1), 5)
    B = copy(A)
    fill!(B.v, 7); fill!(B.vout, 9); fill!(B.zpref, 3)
    @test A == B && isequal(A, B) && hash(A) == hash(B)
    @test A != CliffordOperator(Fourier(1), 5)
    @test A != CliffordOperator(Phase(2), 5)
    @test A != CliffordOperator(Phase(1), 3)
    @test Dict(A => :found)[B] === :found
    # Deliberate cache perturbation tests value semantics only; do not execute B.
    fill!(B.image_xdotz, 99)
    @test A == B && isequal(A, B) && hash(A) == hash(B)

    # show exposes semantic data only.
    s = sprint(show, A)
    @test occursin("CliffordOperator", s) && occursin("d=5", s)
    @test !occursin("zpref", s)
end

@testset "Construction copies into independent dense storage" begin
    F, a = identity_data(1)
    t = [2]
    U = CliffordOperator(3, t, F, a)
    t[1] = 99; F[1, 1] = 7; a[1] = 5
    @test U.targets == [2] && U.F[1, 1] == 1 && U.a[1] == 0
    # A range target must materialize as Vector{Int}, not stay a range.
    R = CliffordOperator(3, 1:1, identity_data(1)...)
    @test R.targets isa Vector{Int} && R.targets == [1]
end

@testset "Stored apply! matches named apply!" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5),
        storephase in (false, true)
        gates = Any[Fourier(2), Phase(2), PauliGate(2, 1, 1),
                    SUM(2, 5, 1), CPhase(2, 5, 1), SWAP(2, 5)]
        push!(gates, Multiplier(2, d == 2 ? 1 : 2))
        for g in gates
            named  = TT(d, 6; state = :ghz, storephase = storephase)
            stored = TT(d, 6; state = :ghz, storephase = storephase)
            apply!(named, g)
            apply!(stored, CliffordOperator(g, d))
            @test stored.stab == named.stab
            @test stored.m == named.m
            TT === DestabilizerTableau && @test stored.destab == named.destab
        end
    end
end

# Include all mutable tableau fields when checking failure atomicity.
tableau_snapshot(tab) = map(fieldnames(typeof(tab))) do f
    x = getfield(tab, f)
    # PrecomputedInvMod has identity equality, so compare its table by value.
    x isa QC.PrecomputedInvMod ? (typeof(x), copy(x.lookuptable)) : deepcopy(x)
end

@testset "Stored _prepare dispatches and validates" begin
    U = CliffordOperator(Fourier(1), 3)
    for im in (QC.PrecomputedInvMod(3), QC.JustInTimeInvMod()),
        sp in (false, true), fs in (false, true)
        prep = QC._prepare(U, 3, im, sp; force_safe = fs)
        @test prep isa QC.PreparedDenseClifford
        @test prep.storephase == sp
        @test prep.fast == (U.fast && !fs)
        @test prep.v === U.v && prep.vout === U.vout
    end
    # Dimension mismatch fails before any tableau mutation.
    tab = StabilizerTableau(5, 3; state = :ghz)
    before = copy(tab.stab)
    @test_throws ArgumentError apply!(tab, U)
    @test tab.stab == before
    # Out-of-register target fails before mutation too.
    W = CliffordOperator(Fourier(9), 3)
    tab3 = StabilizerTableau(3, 3; state = :ghz)
    before3 = copy(tab3.stab)
    @test_throws ArgumentError apply!(tab3, W)
    @test tab3.stab == before3
    for TT in (StabilizerTableau, DestabilizerTableau), sp in (false, true)
        tab = TT(3, 3; state = :ghz, storephase = sp)
        mismatched = CliffordOperator(Fourier(1), 5)
        emptywrong = CliffordOperator(5, Int[], zeros(Int, 0, 0), Int[])
        for bad in (W, mismatched, emptywrong)
            snap = tableau_snapshot(tab)
            usnap = operator_snapshot(bad)
            @test_throws ArgumentError apply!(tab, bad)
            @test tableau_snapshot(tab) == snap
            @test operator_snapshot(bad) == usnap
        end
        # The prepared boundary also rejects phase/dimension/bounds mismatches
        # before touching any tableau or borrowed-scratch field.
        for prep in (QC._prepare(U, 3, QC.JustInTimeInvMod(), !sp),
                     QC._prepare(mismatched, 5, QC.JustInTimeInvMod(), sp),
                     QC._prepare(W, 3, QC.JustInTimeInvMod(), sp))
            snap = tableau_snapshot(tab)
            @test_throws ArgumentError QC._apply_prepared!(tab, prep)
            @test tableau_snapshot(tab) == snap
        end
    end
end

function stored_apply_allocations(tab, U)
    for _ in 1:3
        apply!(tab, U)
    end
    return @allocated apply!(tab, U)
end

@testset "Stored apply! is allocation-free after warm-up" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5),
        storephase in (false, true)
        tab = TT(d, 12; state = :ghz, storephase = storephase)
        for g in (Fourier(1), Phase(1), SUM(1, 12), CPhase(1, 12), SWAP(1, 12))
            U = CliffordOperator(g, d)
            stored_apply_allocations(tab, U)
            @test stored_apply_allocations(tab, U) == 0
        end
    end
end

@testset "Empty support applies as the identity and preserves iscanonical" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3),
        sp in (false, true), state in (:ghz, :mixed)
        tab = TT(d, 4; state, storephase = sp)
        canonicalize!(tab)
        @test tab.iscanonical
        before = tableau_snapshot(tab)
        E = CliffordOperator(d, Int[], zeros(Int, 0, 0), Int[])
        @test apply!(tab, E) === tab
        @test tableau_snapshot(tab) == before
    end
end
