using QuditClifford
using Test
using Random

const QC = QuditClifford

# Build a dense prepared view from tuple gate data, so the dense evaluators
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
# already owned by a fixture. This is the boundary a wrapper-construction-plus-
# apply benchmark measures.
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

    # show exposes semantic data only: identical for A and the scratch- and
    # cache-perturbed B.
    s = sprint(show, A)
    @test occursin("CliffordOperator", s) && occursin("d=5", s)
    @test s == sprint(show, B)
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

@testset "Stored conjugate matches named conjugate" begin
    for d in (2, 3, 5)
        gates = Any[Fourier(2), Phase(2), PauliGate(2, 1, 1),
                    SUM(2, 4, 1), CPhase(2, 4, 1), SWAP(2, 4)]
        push!(gates, Multiplier(2, d == 2 ? 1 : 2))
        n = 5
        for g in gates
            U = CliffordOperator(g, d)
            for xz in ([1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                       zeros(Int, 2n))
                op = GeneralPauli(copy(xz), 1)
                named  = conjugate(g, GeneralPauli(copy(xz), 1); d = d)
                stored = conjugate(U, op)
                @test stored.xz == named.xz && stored.phase == named.phase
                explicit = conjugate(U, op, n; d)
                @test explicit.xz == stored.xz && explicit.phase == stored.phase
                @test stored.xz !== op.xz
                # The sparse arity includes phase and inferred/matching dimensions.
                sp = SinglePauli(2, 1, 1)
                a, b = conjugate(U, sp, n), conjugate(g, sp, n; d)
                @test a.xz == b.xz && a.phase == b.phase
                c = conjugate(U, sp, n; d)
                @test c.xz == a.xz && c.phase == a.phase
                @test op.xz == xz && op.phase == 1
            end
        end
    end
end

@testset "Stored conjugate resolves dimension and leaves scratch untouched" begin
    U = CliffordOperator(Fourier(1), 3)
    op = GeneralPauli([1, 0, 0, 0], 0)
    @test conjugate(U, op).xz == conjugate(U, op; d = 3).xz
    @test_throws ArgumentError conjugate(U, op; d = 5)
    # Named gates still require the keyword.
    @test_throws ArgumentError conjugate(Fourier(1), op)

    # Sentinel scratch must survive conjugation bit-identically.
    fill!(U.v, -7); fill!(U.vout, -8); fill!(U.zpref, -9)
    snap = operator_snapshot(U)
    conjugate(U, GeneralPauli([1, 1, 0, 1], 2))
    @test operator_snapshot(U) == snap
    for f in (() -> conjugate(U, GeneralPauli([1, 0, 0], 0)),
              () -> conjugate(U, op, 1),
              () -> conjugate(U, SinglePauli(0, 0, 0), 2),
              () -> conjugate(U, DoublePauli(1, 0, 0, 1, 0, 0), 2),
              () -> conjugate(U, SinglePauli(1, 1, 0), -1),
              () -> conjugate(U, SinglePauli(1, 1, 0), typemax(Int)),
              () -> conjugate(U, SinglePauli(1, 1, 0), 2; d = 5))
        @test_throws ArgumentError f()
        @test operator_snapshot(U) == snap
    end
    badtarget = CliffordOperator(Fourier(9), 3)
    @test_throws ArgumentError conjugate(badtarget, op)
    raw = GeneralPauli([typemin(Int), 8, -7, 10], -5)
    oldraw = (copy(raw.xz), raw.phase)
    normalized = GeneralPauli(mod.(raw.xz, 3), mod(raw.phase, 3))
    a, b = conjugate(U, raw), conjugate(Fourier(1), normalized; d = 3)
    @test a.xz == b.xz && a.phase == b.phase
    @test (raw.xz, raw.phase) == oldraw

    # Empty Clifford support still validates the whole Pauli input.
    E = CliffordOperator(3, Int[], zeros(Int, 0, 0), Int[])
    rawempty = GeneralPauli([-1, 5, 3, -2], -1)
    normalizedempty = conjugate(E, rawempty)
    @test normalizedempty.xz == [2, 2, 0, 1] && normalizedempty.phase == 2
    @test normalizedempty !== rawempty && normalizedempty.xz !== rawempty.xz
    @test rawempty.xz == [-1, 5, 3, -2] && rawempty.phase == -1
    @test_throws ArgumentError conjugate(E, SinglePauli(9, 1, 0), 2)
end

# F_inv must be the actual matrix inverse mod d, independent of the block formula.
function is_matrix_inverse(F::Matrix{Int}, G::Matrix{Int}, d::Int)
    S = size(F, 1)
    P = [mod(sum(big(F[i, r]) * G[r, j] for r in 1:S), d) for i in 1:S, j in 1:S]
    return P == [i == j ? 1 : 0 for i in 1:S, j in 1:S]
end

@testset "inv is the group inverse" begin
    for d in (2, 3, 5)
        gates = Any[Fourier(1), Phase(1), PauliGate(1, 1, 1),
                    SUM(1, 2, 1), CPhase(1, 2, 1), SWAP(1, 2)]
        push!(gates, Multiplier(1, d == 2 ? 1 : 2))
        for g in gates
            U = CliffordOperator(g, d)
            W = inv(U)
            @test W.d == d && W.targets == U.targets
            @test is_matrix_inverse(U.F, W.F, d)
            @test inv(W) == U
            assert_independent(W, U)
            # inv(U) must itself pass the public constructor's algebraic checks.
            @test CliffordOperator(d, W.targets, W.F, W.a) == W
            # Applying U then inv(U) is the identity on a tableau, phases included.
            for TT in (StabilizerTableau, DestabilizerTableau)
                tab = TT(d, 4; state = :ghz)
                before = copy(tab.stab)
                apply!(tab, U); apply!(tab, W)
                @test tab.stab == before
            end
            # Operand and its scratch are untouched.
            fill!(U.v, -1); fill!(U.vout, -2); fill!(U.zpref, -3)
            snap = operator_snapshot(U)
            inv(U)
            @test operator_snapshot(U) == snap
        end
    end
    # Qubit raw phases: Phase(1)'s generator images have exponents (1, 0);
    # Fourier(1)'s have (0, 0).
    P2, F2 = CliffordOperator(Phase(1), 2), CliffordOperator(Fourier(1), 2)
    @test P2.a == [1, 0] && F2.a == [0, 0]
    # Empty support inverts to itself.
    E = CliffordOperator(3, Int[], zeros(Int, 0, 0), Int[])
    @test inv(E) == E
end

@testset "Composition applies its right operand first" begin
    for d in (2, 3, 5)
        pairs = Any[(Fourier(1), Phase(1)), (Phase(1), Fourier(1)),
                    (PauliGate(1, 1, 1), Fourier(1))]
        for (gu, gv) in pairs
            U, V = CliffordOperator(gu, d), CliffordOperator(gv, d)
            UV = U ∘ V
            @test UV.d == d && UV.targets == U.targets
            # The composite must pass the public constructor's algebraic checks.
            @test CliffordOperator(d, UV.targets, UV.F, UV.a) == UV
            for TT in (StabilizerTableau, DestabilizerTableau)
                seq = TT(d, 4; state = :ghz)
                one = TT(d, 4; state = :ghz)
                apply!(seq, V); apply!(seq, U)     # V first
                apply!(one, UV)
                @test one.stab == seq.stab
                TT === DestabilizerTableau && @test one.destab == seq.destab
            end
            # Group laws.
            @test U ∘ inv(U) == CliffordOperator(d, U.targets,
                                                 Matrix{Int}(eye_int(2 * length(U.targets))),
                                                 zeros(Int, 2 * length(U.targets)))
            # Associativity and self-composition.
            W1 = CliffordOperator(d, U.targets, U.F, U.a)   # same support as U
            @test (U ∘ V) ∘ W1 == U ∘ (V ∘ W1)
            @test U ∘ U isa CliffordOperator
            assert_independent(UV, U)
            assert_independent(UV, V)
            # Chained `∘` is left-associative binary and stays in the type.
            @test (U ∘ V ∘ W1) isa CliffordOperator
            # Operands and their scratch survive bit-identically.
            for (i, f) in enumerate((:v, :vout, :zpref))
                fill!(getfield(U, f), -i)
                fill!(getfield(V, f), -i - 3)
            end
            usnap, vsnap = operator_snapshot(U), operator_snapshot(V)
            U ∘ V
            U ∘ U
            @test operator_snapshot(U) == usnap
            @test operator_snapshot(V) == vsnap
        end
    end
    # Mismatches are rejected.
    A = CliffordOperator(Fourier(1), 3)
    @test_throws ArgumentError A ∘ CliffordOperator(Fourier(1), 5)
    @test_throws ArgumentError A ∘ CliffordOperator(Fourier(2), 3)
    @test_throws ArgumentError CliffordOperator(SUM(1, 2, 1), 3) ∘
                               CliffordOperator(SUM(2, 1, 1), 3)
    # Qubit composition: Phase ∘ Fourier has raw phases (0, 1), and its inverse
    # has (1, 0).
    PF = CliffordOperator(Phase(1), 2) ∘ CliffordOperator(Fourier(1), 2)
    @test PF.a == [0, 1]
    @test inv(PF).a == [1, 0]
    for d in (2, 3, 5)
        E = CliffordOperator(d, Int[], zeros(Int, 0, 0), Int[])
        @test inv(E) == E && E ∘ E == E
        assert_independent(E ∘ E, E)
    end
end

# Exact BigInt ordered-product oracle, the dense twin of `phase_reference` in
# test/unitaries.jl. Independent of the production evaluators: it forms the
# quadratic part directly instead of reconstructing it from the D cache.
function phase_reference_dense(F::Matrix{Int}, a::Vector{Int}, v::Vector{Int}, d::Int)
    S = length(v); K = S ÷ 2
    p = d == 2 ? 4 : d
    γ = d == 2 ? 2 : 1
    colxz(i) = sum(big(F[q, i]) * F[K + q, i] for q in 1:K; init = big(0))
    lin = sum(big(a[i]) * v[i] for i in 1:S; init = big(0))
    quad = sum(binomial(big(v[i]), 2) * colxz(i) for i in 1:S; init = big(0))
    quad += sum(sum(big(F[K + r, i]) * F[r, j] for r in 1:K; init = big(0)) *
                v[i] * v[j] for i in 1:S for j in (i + 1):S; init = big(0))
    return Int(mod(lin + γ * quad, p))
end

# Every 2x2 matrix over Z_d with determinant 1: Sp(2, Z_d) = SL(2, Z_d),
# of order d(d^2 - 1) -- 6 at d = 2, 24 at d = 3.
function sl2_matrices(d::Int)
    out = Matrix{Int}[]
    for a in 0:(d - 1), b in 0:(d - 1), c in 0:(d - 1), e in 0:(d - 1)
        mod(a * e - b * c, d) == 1 && push!(out, [a b; c e])
    end
    return out
end

@testset "Every one-qudit Clifford through the actual dense path" begin
    for (d, expected) in ((2, 24), (3, 216))
        mats = sl2_matrices(d)
        @test length(mats) == d * (d^2 - 1)
        built = 0
        for F in mats
            D = [mod(F[1, i] * F[2, i], d) for i in 1:2]
            # Valid raw phases: all of Z_d at odd d; parity-constrained at d = 2.
            phasesets = d == 2 ? [[D[1] + 2s1, D[2] + 2s2] for s1 in 0:1 for s2 in 0:1] :
                                 [[a1, a2] for a1 in 0:(d - 1) for a2 in 0:(d - 1)]
            for a in phasesets
                U = CliffordOperator(d, [1], copy(F), copy(a))
                built += 1
                # Compare the production dense evaluator to the exact oracle on
                # EVERY local Pauli exponent vector.
                prep = QC._prepare(U, d, QC.JustInTimeInvMod(), true)
                for x in 0:(d - 1), z in 0:(d - 1)
                    v = [x, z]
                    out = QC._matvec_prepared!(prep, copyto!(prep.v, v))
                    @test collect(out) == [mod(sum(big(F[i, j]) * v[j] for j in 1:2), d)
                                           for i in 1:2]
                    @test QC._phase(prep, prep.v, prep.vout) ==
                          phase_reference_dense(U.F, U.a, v, d)
                end
                # Basis images recover the raw phases: φ(e_i) == a[i], since
                # the quadratic part vanishes on a basis vector.
                for i in 1:2
                    e = zeros(Int, 2); e[i] = 1
                    QC._matvec_prepared!(prep, copyto!(prep.v, e))
                    @test QC._phase(prep, prep.v, prep.vout) == U.a[i]
                end
                # The safe tier must agree with the default one.
                safe = QC._prepare(U, d, QC.JustInTimeInvMod(), true; force_safe = true)
                copyto!(safe.v, [d - 1, d - 1]); QC._matvec_prepared!(safe, safe.v)
                @test QC._phase(safe, safe.v, safe.vout) ==
                      phase_reference_dense(U.F, U.a, [d - 1, d - 1], d)
                # Exercise the public normalization/gather/scatter entry point too.
                for x in 0:(d - 1), z in 0:(d - 1), h in 0:(QC.phase_modulus(d) - 1)
                    v = [x, z]
                    op = GeneralPauli(v, h)
                    out = conjugate(U, op)
                    @test out.xz == Int.(mod.(big.(U.F) * v, d))
                    @test out.phase == mod(h + phase_reference_dense(U.F, U.a, v, d), QC.phase_modulus(d))
                    @test op.xz == v && op.phase == h
                end
                # Full group identities include raw phases, in both orders.
                W = inv(U)
                Iop = CliffordOperator(d, [1], eye_int(2), zeros(Int, 2))
                @test is_matrix_inverse(U.F, W.F, d)
                @test U ∘ W == Iop && W ∘ U == Iop
                @test inv(W) == U
                @test CliffordOperator(d, W.targets, W.F, W.a) == W
            end
        end
        @test built == expected
    end
end

# Identity on k qubits except the given local 2x2 action at coordinate `q`.
function embed_local(k::Int, q::Int, M::Matrix{Int}, phases::Vector{Int}, d::Int)
    S = 2k
    F = eye_int(S)
    F[q, q]         = M[1, 1]; F[k + q, q]         = M[2, 1]
    F[q, k + q]     = M[1, 2]; F[k + q, k + q]     = M[2, 2]
    a = zeros(Int, S)
    a[q] = phases[1]; a[k + q] = phases[2]
    return CliffordOperator(d, collect(1:k), F, a)
end

@testset "A k = 65 qubit operator exercises prefix coordinates above bit 64" begin
    k = 65; q = 65; n = 65
    # Fourier at coordinate 65: X -> Z, Z -> X. Its ordered-product cross term
    # is NONZERO, which is exactly what a UInt64 prefix would drop, since
    # UInt64(1) << 64 == 0. A Phase embedding would NOT detect that: its phase
    # is linear in the input X coefficient and the relevant cross term is zero.
    UF = embed_local(k, q, [0 1; 1 0], [0, 0], 2)
    @test length(UF.targets) == 65

    yv = zeros(Int, 2n); yv[q] = 1; yv[n + q] = 1
    # Y_65 -> -Y_65: exponents preserved, raw phase 1 -> 3.
    outY = conjugate(UF, GeneralPauli(copy(yv), 1))
    @test outY.xz == yv
    @test outY.phase == 3
    # The same X_65 Z_65 vector with input raw phase zero acquires phase 2.
    @test conjugate(UF, GeneralPauli(copy(yv), 0)).phase == 2

    # Phase after Fourier: input raw phase zero gives output phase 3.
    UP = embed_local(k, q, [1 0; 1 1], [1, 0], 2)
    PF = UP ∘ UF
    @test conjugate(PF, GeneralPauli(copy(yv), 0)).phase == 3

    # Cross-check every one of these against the exact oracle.
    for U in (UF, UP, PF)
        local v = zeros(Int, 2k); v[q] = 1; v[k + q] = 1
        prep = QC._prepare(U, 2, QC.JustInTimeInvMod(), true)
        QC._matvec_prepared!(prep, copyto!(prep.v, v))
        @test QC._phase(prep, prep.v, prep.vout) ==
              phase_reference_dense(U.F, U.a, v, 2)
    end

    # Direct application on Y generators catches a wrong phase even when
    # the same error could cancel in an inverse round trip.
    for TT in (StabilizerTableau, DestabilizerTableau), sp in (false, true)
        tab = TT(2, n; state = :product, basis = :Y, storephase = sp)
        before = copy(tab.stab)
        apply!(tab, UF)
        expected = copy(before)
        sp && (expected[2n + 1, q] = 3)
        @test tab.stab == expected
        if TT === DestabilizerTableau
            @test tab.xdotz_cache == ones(Int, n)
        end
        apply!(tab, inv(UF))
        @test tab.stab == before
    end

    # A one-qudit support whose only physical target is 65 has local arity ONE
    # and does not test this limit at all.
    small = CliffordOperator(2, [65], [0 1; 1 0], [0, 0])
    @test length(small.targets) == 1
end

@testset "Large moduli use exact arithmetic in both tiers" begin
    d = Sys.WORD_SIZE == 64 ? 2147483647 : 32749
    # A counterexample matrix, now reachable through the PUBLIC constructor:
    # an unreduced four-term Int dot wraps to 0 while the true answer is 4.
    M = mod.([-1 -1 -1 -1; 0 -1 0 -1; 0 0 -1 0; 0 0 1 -1], d)
    U = CliffordOperator(d, [1, 2], M, zeros(Int, 4))
    @test !QC.clifford_fast_dots(4, d)     # the guard rejects k = 2 at this d
    @test U.fast == false
    v = fill(d - 1, 4)
    prep = QC._prepare(U, d, QC.JustInTimeInvMod(), true)
    out = QC._matvec_prepared!(prep, copyto!(prep.v, v))
    expected = [Int(mod(sum(big(M[i, j]) * v[j] for j in 1:4), d)) for i in 1:4]
    @test collect(out) == expected
    @test expected[1] == 4
    @test QC._phase(prep, prep.v, prep.vout) == phase_reference_dense(U.F, U.a, v, d)
    # inv and ∘ stay exact at this modulus.
    @test is_matrix_inverse(U.F, inv(U).F, d)
    @test U ∘ inv(U) == CliffordOperator(d, [1, 2], eye_int(4), zeros(Int, 4))
    outpublic = conjugate(U, GeneralPauli(copy(v), d - 1))
    @test outpublic.xz == expected
    @test outpublic.phase == Int(mod(big(d - 1) + phase_reference_dense(U.F, U.a, v, d), d))

    # A k = 1 operator at the same d is inside the fast guard; both tiers must
    # agree with BigInt.
    P = CliffordOperator(d, [1], [1 0; 1 1], [0, 0])
    @test QC.clifford_fast_dots(2, d)
    for fs in (false, true)
        pp = QC._prepare(P, d, QC.JustInTimeInvMod(), true; force_safe = fs)
        w = [d - 1, 0]
        QC._matvec_prepared!(pp, copyto!(pp.v, w))
        @test QC._phase(pp, pp.v, pp.vout) == phase_reference_dense(P.F, P.a, w, d)
    end

    # Above the scalar Int-product boundary, exercise widemul in the actual
    # stored path, including raw validation, inverse and composition phases.
    nearmax = Sys.WORD_SIZE == 64 ? 9223372036854775783 : 2147483647
    for g in (Fourier(1), Phase(1), PauliGate(1, typemin(Int), typemin(Int)),
              SUM(1, 2, nearmax - 1), CPhase(1, 2, nearmax - 1))
        W = CliffordOperator(g, nearmax)
        @test !W.fast
        @test CliffordOperator(nearmax, W.targets, W.F, W.a) == W
        S = length(W.a)
        v = fill(nearmax - 1, S)
        xref = Int.(mod.(big.(W.F) * v, nearmax))
        href = phase_reference_dense(W.F, W.a, v, nearmax)
        for fs in (false, true)
            prep = QC._prepare(W, nearmax, QC.JustInTimeInvMod(), true; force_safe = fs)
            QC._matvec_prepared!(prep, copyto!(prep.v, v))
            @test prep.vout == xref
            @test QC._phase(prep, prep.v, prep.vout) == href
        end
        Iop = CliffordOperator(nearmax, W.targets, eye_int(S), zeros(Int, S))
        @test W ∘ inv(W) == Iop && inv(W) ∘ W == Iop
        WW = W ∘ W
        @test WW.F == Int.(mod.(big.(W.F) * big.(W.F), nearmax))
        for i in 1:S
            @test WW.a[i] == Int(mod(big(W.a[i]) +
                phase_reference_dense(W.F, W.a, W.F[:, i], nearmax), nearmax))
        end
    end
end

# Embed a named action on local support coordinates without using composition.
# The dense identity gets only the selected rows/columns replaced.
function embed_named_on(g, d::Int, targets::Vector{Int})
    local_targets, tF, ta = QC._clifford_data(g, d, QC.JustInTimeInvMod())
    k = length(targets)
    l = length(local_targets)
    inds = vcat(collect(local_targets), k .+ collect(local_targets))
    F, a = eye_int(2k), zeros(Int, 2k)
    for j in 1:(2l)
        a[inds[j]] = ta[j]
        for i in 1:(2l)
            F[inds[i], inds[j]] = tF[j][i]
        end
    end
    return CliffordOperator(d, targets, F, a)
end

function physical_gate(g, targets)
    g isa Fourier && return Fourier(targets[g.qudit])
    g isa Phase && return Phase(targets[g.qudit])
    g isa SUM && return SUM(targets[g.control], targets[g.target], g.a)
    g isa CPhase && return CPhase(targets[g.qudit1], targets[g.qudit2], g.a)
    error("unsupported fixture gate")
end

@testset "Seeded entangling supports with k > 2 and spaced reversed targets" begin
    for d in (2, 3, 5), k in (3, 4)
        rng = Random.MersenneTwister(20260921 + d + k)
        targets = reverse(collect(2:2:2k))
        @test all(abs.(diff(targets)) .> 1)
        U = CliffordOperator(d, targets, eye_int(2k), zeros(Int, 2k))
        # Guaranteed entangling action followed by seeded local and two-site gates.
        gates = AbstractClifford[Fourier(1), SUM(1, 2), CPhase(2, 3)]
        for step in 1:6
            q = rand(rng, 1:k)
            r = mod1(q + 1, k)
            push!(gates, rand(rng, Bool) ? Phase(q) : Fourier(q))
            push!(gates, SUM(q, r, rand(rng, 1:(d - 1))))
        end
        for g in gates
            U = embed_named_on(g, d, targets) ∘ U
        end
        @test CliffordOperator(d, targets, U.F, U.a) == U
        # Verify that the guaranteed SUM embedding really couples local sites.
        entangler = embed_named_on(SUM(1, 2), d, targets)
        @test entangler.F[2, 1] == 1
        n = maximum(targets) + 2
        for TT in (StabilizerTableau, DestabilizerTableau), sp in (false, true),
            state in (:product, :mixed)
            named = TT(d, n; state, storephase = sp)
            if state === :mixed
                measure!(named, SinglePauli(targets[1], 0, 1); outcome = 0)
                measure!(named, SinglePauli(targets[2], 0, 1); outcome = 0)
            end
            one = deepcopy(named)
            old = copy(one.stab)
            oldm = one.m
            for g in gates
                apply!(named, physical_gate(g, targets))
            end
            apply!(one, U)
            @test one.stab == named.stab && one.m == oldm
            @test all(iszero, one.stab[:, (oldm + 1):end])
            for q in setdiff(1:n, targets)
                @test one.stab[q, :] == old[q, :]
                @test one.stab[n + q, :] == old[n + q, :]
            end
            if TT === DestabilizerTableau
                @test one.destab == named.destab
                @test one.xdotz_cache == named.xdotz_cache
            end
            apply!(one, inv(U))
            @test one.stab == old
        end
        for sample in 1:8
            v = rand(rng, 0:(d - 1), 2k)
            expectedxz = Int.(mod.(big.(U.F) * v, d))
            expectedphase = phase_reference_dense(U.F, U.a, v, d)
            for fs in (false, true)
                prep = QC._prepare(U, d, QC.JustInTimeInvMod(), true; force_safe = fs)
                QC._matvec_prepared!(prep, copyto!(prep.v, v))
                beforev, beforeout = copy(prep.v), copy(prep.vout)
                @test prep.vout == expectedxz
                @test QC._phase(prep, prep.v, prep.vout) == expectedphase
                @test prep.v == beforev && prep.vout == beforeout
            end
            # Sequential named conjugation is independent of dense composition.
            xz = zeros(Int, 2n)
            xz[targets] = v[1:k]; xz[n .+ targets] = v[(k + 1):end]
            seq = GeneralPauli(copy(xz), 1)
            for g in gates
                seq = conjugate(physical_gate(g, targets), seq; d)
            end
            out = conjugate(U, GeneralPauli(copy(xz), 1))
            @test out.xz == seq.xz && out.phase == seq.phase
            @test out.xz[targets] == expectedxz[1:k]
            @test out.xz[n .+ targets] == expectedxz[(k + 1):end]
        end
    end
end

function prepared_apply_allocations(tab, prep)
    for _ in 1:3
        QC._apply_prepared!(tab, prep)
    end
    return @allocated QC._apply_prepared!(tab, prep)
end

@testset "Runtime supports allocate nothing during application" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3),
        sp in (false, true), k in (0, 3, 8, 65), state in (:product, :mixed)
        n = max(k, 1)
        tab = TT(d, n; state, storephase = sp)
        U = k == 0 ? CliffordOperator(d, Int[], zeros(Int, 0, 0), Int[]) :
                     embed_named_on(Fourier(k), d, collect(1:k))
        stored_apply_allocations(tab, U)
        @test stored_apply_allocations(tab, U) == 0
        prep = QC._prepare(U, d, tab.inversemod, sp; force_safe = true)
        prepared_apply_allocations(tab, prep)
        @test prepared_apply_allocations(tab, prep) == 0
    end
end

@testset "Safe-tier stored application and cache delta agree with BigInt" begin
    d = Sys.WORD_SIZE == 64 ? 2147483647 : 32749
    M = mod.([-1 -1 -1 -1; 0 -1 0 -1; 0 0 -1 0; 0 0 1 -1], d)
    U = CliffordOperator(d, [1, 2], M, zeros(Int, 4))
    for TT in (StabilizerTableau, DestabilizerTableau), sp in (false, true)
        tab = TT(d, 2; state = :product, basis = :X, storephase = sp,
                 inversemod = QC.JustInTimeInvMod())
        # Named Phase creates a physical, nonzero old x·z using Clifford
        # arithmetic, without general tableau re-initialization at large d.
        apply!(tab, Phase(1))
        apply!(tab, Phase(2))
        old = copy(tab.stab)
        expected = copy(old)
        for j in 1:tab.m
            v = old[1:4, j]
            expected[1:4, j] = Int.(mod.(big.(M) * v, d))
            sp && (expected[5, j] = Int(mod(big(old[5, j]) +
                phase_reference_dense(U.F, U.a, v, d), d)))
        end
        olddual = TT === DestabilizerTableau ? copy(tab.destab) : nothing
        apply!(tab, U)
        @test tab.stab == expected
        if TT === DestabilizerTableau
            @test tab.destab == Int.(mod.(big.(M) * olddual, d))
            for j in 1:tab.m
                @test tab.xdotz_cache[j] == Int(mod(sum(big(tab.stab[q, j]) *
                    tab.stab[2 + q, j] for q in 1:2), d))
            end
        end
        stored_apply_allocations(tab, U)
        @test stored_apply_allocations(tab, U) == 0
    end
end

# Straddles the dense matvec's row/column switch (`_DENSE_MATVEC_ROWWISE_MAX_S`,
# introduced alongside these tests) from both directions, in both arithmetic
# tiers, so neither branch of the restructured kernel silently diverges from
# `mod(F * v, d)`. Symplecticity is irrelevant to a matvec, so `F` is just a
# random canonical matrix -- built directly from `PreparedDenseClifford`,
# matching how the following testset builds it too.
@testset "Dense matvec matches BigInt on both sides of the row/column switch" begin
    rng = Random.MersenneTwister(20260924)
    Smax = QC._DENSE_MATVEC_ROWWISE_MAX_S
    for k in unique([1, 2, Smax ÷ 2 - 1, Smax ÷ 2, Smax ÷ 2 + 1, Smax, 65])
        S = 2k
        for d in (2, 3, 5, 1000000007), force_safe in (false, true)
            fast = !force_safe && QC.clifford_fast_dots(S, d)
            F = rand(rng, 0:(d - 1), S, S)
            v = zeros(Int, S)
            vout = zeros(Int, S)
            zpref = zeros(Int, k)
            prep = QC.PreparedDenseClifford(collect(1:k), F, zeros(Int, S),
                zeros(Int, S), v, vout, zpref, k, d, QC.phase_modulus(d), 0,
                fast, true)
            inputs = Vector{Int}[zeros(Int, S)]
            onehot = zeros(Int, S)
            onehot[rand(rng, 1:S)] = rand(rng, 1:(d - 1))
            push!(inputs, onehot)
            for _ in 1:3
                push!(inputs, rand(rng, 0:(d - 1), S))
            end
            for vv in inputs
                copyto!(prep.v, vv)
                beforev = copy(prep.v)
                out = QC._matvec_prepared!(prep, prep.v)
                @test out === prep.vout
                @test collect(out) == Int.(mod.(big.(F) * vv, d))
                @test prep.v == beforev
            end
        end
    end
end

# The ordered-product BigInt oracle (`phase_reference_dense`, defined above)
# holds for ANY F, symplectic or not, so it doubles as a direct check on the
# branch-free rewrite -- independent of whatever internal form
# `_phase_qubit_dense` uses to reach the same parity.
@testset "Dense qubit phase matches the ordered-product oracle on dense inputs" begin
    rng = Random.MersenneTwister(20260925)
    for k in (1, 2, 3, 8, 33, 100)
        S = 2k
        for _ in 1:5
            F = rand(rng, 0:1, S, S)
            a = rand(rng, 0:3, S)
            v = rand(rng, 0:1, S)
            zpref = fill(1, k)   # stale on purpose: proves the kernel clears it
            beforev = copy(v)
            phase = QC._phase_qubit_dense(v, F, a, k, zpref)
            @test phase == phase_reference_dense(F, a, v, 2)
            @test v == beforev
        end
    end
end

@testset "Stored apply! stays allocation-free just above the row/column switch" begin
    k = QC._DENSE_MATVEC_ROWWISE_MAX_S ÷ 2 + 1   # S = 2k is just past the switch
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3), sp in (false, true)
        tab = TT(d, k; state = :product, storephase = sp)
        U = embed_named_on(Fourier(k), d, collect(1:k))
        stored_apply_allocations(tab, U)
        @test stored_apply_allocations(tab, U) == 0
        prep = QC._prepare(U, d, tab.inversemod, sp; force_safe = true)
        prepared_apply_allocations(tab, prep)
        @test prepared_apply_allocations(tab, prep) == 0
    end
end
