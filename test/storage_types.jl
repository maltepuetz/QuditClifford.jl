using QuditClifford, Test, Random, LinearAlgebra

const st_QC = QuditClifford

st_types(d) = filter(E -> st_QC._eltype_admissible(E, d), (UInt8, UInt16, Int))

# A signed integer type wider than one machine word on neither platform, so
# never `Int` itself.
const st_OtherSigned = Int === Int64 ? Int32 : Int64

# ---- differential driver: every operation on every element type ----

# A random Pauli in reduced form, Hermitian at d = 2: `kind` 1 gives a
# `SinglePauli`, 2 a `DoublePauli`, 3 a `GeneralPauli` on a random support.
function st_rand_pauli(rng, d, n, kind)
    k = kind == 1 ? 1 : kind == 2 ? 2 : rand(rng, 1:n)
    qs = randperm(rng, n)[1:k]
    xs = [rand(rng, 0:d-1) for _ in 1:k]
    zs = [rand(rng, 0:d-1) for _ in 1:k]
    ph = d == 2 ? mod(sum(xs .* zs), 2) + 2 * rand(rng, 0:1) : rand(rng, 0:d-1)
    kind == 1 && return SinglePauli(qs[1], xs[1], zs[1], ph)
    kind == 2 && return DoublePauli(qs[1], xs[1], zs[1], qs[2], xs[2], zs[2], ph)
    v = zeros(Int, 2n)
    for (q, x, z) in zip(qs, xs, zs)
        v[q] = x
        v[n+q] = z
    end
    return GeneralPauli(v, ph)
end

# Generator `j` of `ref` as a `GeneralPauli`: in the stabilizer group, so a
# measurement of it is deterministic and its expectation value is nonzero.
function st_generator_pauli(ref, j)
    S = stabilizer_matrix(ref)
    n = ref.n
    return GeneralPauli(S[1:2n, j], ref.storephase ? S[2n+1, j] : 0)
end

function st_rand_gate(rng, d, n)
    q, q2 = randperm(rng, n)[1:2]
    c = rand(rng, 1:7)
    c == 1 && return Fourier(q)
    c == 2 && return Phase(q)
    c == 3 && return Multiplier(q, rand(rng, 1:d-1))
    c == 4 && return PauliGate(q, rand(rng, 0:d-1), rand(rng, 0:d-1))
    c == 5 && return SUM(q, q2, rand(rng, 1:d-1))
    c == 6 && return CPhase(q, q2, rand(rng, 1:d-1))
    return SWAP(q, q2)
end

# Three disjoint random subsystems, each possibly empty.
function st_rand_parts(rng, n)
    p = randperm(rng, n)
    a = rand(rng, 0:n)
    b = rand(rng, 0:n-a)
    c = rand(rng, 0:n-a-b)
    return p[1:a], p[a+1:a+b], p[a+b+1:a+b+c]
end

# The next operation, drawn from `rng` and the current state of `ref`. It is a
# function of a tableau and that tableau's own RNG, applied to every tableau of
# the run.
function st_rand_op(rng, ref)
    d, n, m = ref.d, ref.n, ref.m
    c = rand(rng, 1:24)
    if c <= 4
        g = st_rand_gate(rng, d, n)
        return (tab, r) -> apply!(tab, g)
    elseif c == 5
        k = rand(rng, (1, 2, 3, 4))
        C = random_clifford(rng, d, randperm(rng, n)[1:k])
        return (tab, r) -> apply!(tab, C)
    elseif c <= 9
        op = (m > 0 && rand(rng, 1:3) == 1) ? st_generator_pauli(ref, rand(rng, 1:m)) :
             st_rand_pauli(rng, d, n, rand(rng, 1:3))
        return rand(rng, Bool) ? ((tab, r) -> measure!(r, tab, op)) :
                                 ((tab, r) -> measure!(tab, op; outcome = 0))
    elseif c <= 11
        op = (m > 0 && rand(rng, Bool)) ? st_generator_pauli(ref, rand(rng, 1:m)) :
             st_rand_pauli(rng, d, n, rand(rng, 1:3))
        return c == 10 ? ((tab, r) -> expect!(tab, op)) : ((tab, r) -> expect_int!(tab, op))
    elseif c == 12
        op = st_rand_pauli(rng, d, n, rand(rng, (1, 3)))
        return (tab, r) -> dephase!(tab, op)
    elseif c == 13
        sites = randperm(rng, n)[1:rand(rng, 1:min(n, 3))]
        return (tab, r) -> traceout!(tab, sites)
    elseif c == 14
        sites = randperm(rng, n)[1:rand(rng, 1:min(n, 3))]
        basis = rand(rng, (:Z, :X, :Y))
        return basis === :Z ? ((tab, r) -> reset!(tab, sites)) :
                              ((tab, r) -> reset!(tab, sites; basis))
    elseif c == 15
        return (tab, r) -> canonicalize!(tab)
    elseif c == 16
        mm = rand(rng, (n, n, rand(rng, 0:n)))
        return (tab, r) -> random_state!(r, tab; m = mm)
    elseif c == 17
        A, _, _ = st_rand_parts(rng, n)
        return (tab, r) -> entanglement_entropy(tab, A)
    elseif c == 18
        return (tab, r) -> entanglement_profile(tab)
    elseif c == 19
        A, B, _ = st_rand_parts(rng, n)
        return (tab, r) -> mutual_information(tab, A, B)
    elseif c == 20
        A, B, C = st_rand_parts(rng, n)
        return (tab, r) -> conditional_mutual_information(tab, A, B, C)
    elseif c == 21
        A, B, C = st_rand_parts(rng, n)
        return (tab, r) -> tripartite_information(tab, A, B, C)
    elseif c == 22
        A, B, _ = st_rand_parts(rng, n)
        return (tab, r) -> entanglement_negativity(tab, A, B)
    elseif c == 23
        return (tab, r) -> is_pure(tab; verify = true)
    end
    # A further gate, so that two-qudit gates make up a good share.
    g = st_rand_gate(rng, d, n)
    return (tab, r) -> apply!(tab, g)
end

# What an operation returned: a tableau stands for itself, and a thrown
# exception is compared by type and message.
function st_call(op, tab, r)
    v = try
        op(tab, r)
    catch e
        return (:error, typeof(e), sprint(showerror, e))
    end
    return v isa AbstractTableau ? (v === tab ? :self : :other) : v
end

# The symplectic products of destabilizer `i` with stabilizer `j` over `1:m`,
# in `Int64` so that no `d` here overflows them.
function st_duality(tab)
    n, m, d = tab.n, tab.m, tab.d
    S = Int64.(stabilizer_matrix(tab)[1:2n, 1:m])
    D = Int64.(destabilizer_matrix(tab)[:, 1:m])
    return mod.(transpose(D[1:n, :]) * S[n+1:2n, :] - transpose(D[n+1:2n, :]) * S[1:n, :], d)
end

function st_xdotz(tab)
    n, d = tab.n, tab.d
    S = Int64.(stabilizer_matrix(tab))
    return [Int(mod(sum(S[q, j] * S[n+q, j] for q in 1:n), d)) for j in 1:tab.m]
end

# Apply `nops` operations drawn from `oprng` to every tableau in `tabs`, the
# first being the `eltype = Int` reference, each with its own copy of `seed`
# as RNG, and compare every tableau with the reference after every operation.
# Returns the number of comparisons that failed.
function st_run!(tabs, seed, oprng, nops)
    ref = tabs[1]
    rngs = [Xoshiro(seed) for _ in tabs]
    d, n = ref.d, ref.n
    bad = 0
    for i in 1:nops
        op = st_rand_op(oprng, ref)
        vals = [st_call(op, tab, r) for (tab, r) in zip(tabs, rngs)]
        Sref = stabilizer_matrix(ref)
        Dref = ref isa DestabilizerTableau ? destabilizer_matrix(ref) : nothing
        ok = true
        for (k, tab) in enumerate(tabs)
            same = stabilizer_matrix(tab) == Sref && tab.m == ref.m &&
                   tab.iscanonical == ref.iscanonical && isequal(vals[k], vals[1])
            Dref === nothing || (same &= destabilizer_matrix(tab) == Dref)
            tab isa DestabilizerTableau && (same &= tab.xdotz_cache[1:tab.m] == st_xdotz(tab))
            if !same
                ok = false
                @error "differs from the Int reference" i eltype = Base.eltype(tab.stab_x) vals[k] vals[1]
            end
        end
        if ref isa DestabilizerTableau && st_duality(ref) != Matrix{Int64}(I, ref.m, ref.m)
            ok = false
            @error "the Int reference breaks destabilizer duality" i
        end
        if d == 2 && ref.storephase &&
           !all(mod(Sref[2n+1, j], 2) ==
                mod(sum(Sref[q, j] * Sref[n+q, j] for q in 1:n), 2) for j in 1:ref.m)
            ok = false
            @error "the Int reference has a non-Hermitian qubit generator" i
        end
        ok || (bad += 1)
        ok || break
    end
    return bad
end

@testset "storage element types" begin

@testset "admissible element types" begin
    @test st_QC._default_eltype.((2, 3, 251, 257, 65521, 65537)) ==
          (UInt8, UInt8, UInt8, UInt16, UInt16, Int)
    for TT in (StabilizerTableau, DestabilizerTableau)
        @test_throws ArgumentError TT(257, 2; eltype = UInt8)
        @test_throws ArgumentError TT(3, 2; eltype = st_OtherSigned)
        @test_throws ArgumentError TT(3, 2; eltype = UInt32)
        @test_throws ArgumentError TT(3, 2; eltype = Float64)
        @test_throws ArgumentError TT(3, 2; eltype = Union{UInt8, UInt16})
        @test_throws ArgumentError TT(3, 2; eltype = Union{})
        @test_throws ArgumentError TT(257, [0 1; 1 0; 0 0; 0 0; 0 0]; eltype = UInt8)
        @test typeof(TT(3, 2; eltype = UInt16)).parameters[2] === UInt16
        @test typeof(TT(3, 2)).parameters[2] === UInt8
        @test typeof(TT(3, 2; eltype = Int)).parameters[2] === Int
        @test st_QC._storage_eltype(TT(3, 2; eltype = UInt8)) === UInt8
    end
end

@testset "presets in every element type" begin
    n = 3
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 251, 257),
        E in st_types(d), storephase in (true, false),
        (state, basis) in ((:mixed, :Z), (:product, [:X, :Y, :Z]), (:ghz, :Z))

        tab = TT(d, n; state, basis, storephase, eltype = E)
        ref = TT(d, n; state, basis, storephase, eltype = Int)
        @test stabilizer_matrix(tab) == stabilizer_matrix(ref)
        @test eltype(stabilizer_matrix(tab)) === Int
        @test size(tab.stab_x) == (n, n)
        @test Base.eltype(tab.stab_x) === E && Base.eltype(tab.stab_z) === E
        if TT === DestabilizerTableau
            @test destabilizer_matrix(tab) == destabilizer_matrix(ref)
            @test eltype(destabilizer_matrix(tab)) === Int
            @test Base.eltype(tab.destab_x) === E && Base.eltype(tab.destab_z) === E
        end
    end
end

@testset "raw input reduced before narrowing" begin
    S = [-1 0; 252 3; 0 -251; 1 1; 5 7]
    for (d, M) in ((251, S), (257, S .+ 257), (257, S .- 257))
        ref = stabilizer_matrix(StabilizerTableau(d, M; eltype = Int, check = false))
        @test all(0 .≤ ref[1:4, :] .< d)
        for E in st_types(d)
            tab = StabilizerTableau(d, M; eltype = E, check = false)
            @test Base.eltype(tab.stab_x) === E
            @test stabilizer_matrix(tab) == ref
            fresh = StabilizerTableau(d, 2; eltype = E)
            st_QC._load_stabilizer_matrix!(fresh, M)
            @test stabilizer_matrix(fresh) == ref
        end
    end
    # The destabilizer loader reduces as well.
    for E in st_types(251)
        tab = DestabilizerTableau(251, 2; eltype = E)
        st_QC._load_destabilizer_matrix!(tab, [-1 0; 252 3; 0 -251; 1 1])
        @test destabilizer_matrix(tab) == [250 0; 1 3; 0 0; 1 1]
    end
end

@testset "copy and copy! keep the element type" begin
    for TT in (StabilizerTableau, DestabilizerTableau), E in (UInt8, UInt16)
        tab = TT(3, 4; state = :ghz, eltype = E)
        c = copy(tab)
        @test typeof(c) === typeof(tab)
        @test stabilizer_matrix(c) == stabilizer_matrix(tab)
        @test c.stab_x !== tab.stab_x

        dst = TT(3, 4; eltype = E)
        @test copy!(dst, tab) === dst
        @test stabilizer_matrix(dst) == stabilizer_matrix(tab) && dst.m == tab.m
        if TT === DestabilizerTableau
            @test destabilizer_matrix(dst) == destabilizer_matrix(tab)
        end

        other = TT(3, 4; eltype = E === UInt8 ? UInt16 : UInt8)
        @test_throws ArgumentError copy!(other, tab)
        @test_throws ArgumentError copy!(TT(3, 4; eltype = Int), tab)

        reset!(tab; state = :product)
        @test typeof(tab) === typeof(c)
        @test Base.eltype(tab.stab_x) === E
        @test stabilizer_matrix(tab) ==
              stabilizer_matrix(TT(3, 4; state = :product, eltype = Int))
    end
end

@testset "pivot workspace has the storage type" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 257), E in st_types(d),
        storephase in (true, false)
        n = 5
        tab = TT(d, n; storephase, eltype = E)
        @test Base.eltype(tab.pivot_workspace) === E
        @test length(tab.pivot_workspace) == 2n
        c = copy(tab)
        @test Base.eltype(c.pivot_workspace) === E && length(c.pivot_workspace) == 2n
        @test c.pivot_workspace !== tab.pivot_workspace
        raw = TT(d, stabilizer_matrix(TT(d, n; state = :ghz, storephase)); eltype = E)
        @test Base.eltype(raw.pivot_workspace) === E && length(raw.pivot_workspace) == 2n
    end
end

@testset "equality and hash ignore the element type" begin
    for d in (3, 251), TT in (StabilizerTableau, DestabilizerTableau)
        a = TT(d, 4; state = :ghz, eltype = UInt8)
        b = TT(d, 4; state = :ghz, eltype = Int)
        @test a == b
        @test hash(a) == hash(b)
        @test length(Set([a, b])) == 1
        apply!(b, SUM(1, 2))
        @test a != b
    end
end

@testset "generator operations on every element-type pair" begin
    rng = Xoshiro(7)
    n, ncap = 37, 6
    for d in (2, 3, 251, 257), Eb in st_types(d), Ev in st_types(d)
        X0 = rand(rng, 0:d-1, n, ncap)
        Z0 = rand(rng, 0:d-1, n, ncap)
        v0 = rand(rng, 0:d-1, 2n)
        a = rand(rng, 1:d-1)
        sites = randperm(rng, n)[1:5]
        blk(T) = st_QC.XZBlock{st_QC.GeneratorMajor}(T.(X0), T.(Z0))
        ints(b) = (Int.(b.x), Int.(b.z))
        ops = (
            (b, v) -> st_QC._gen_addmul!(b, 2, v, a, d),
            (b, v) -> st_QC._gen_submul!(b, 2, v, a, d),
            (b, v) -> st_QC._gen_mulcopy!(b, 2, v, a, d),
            (b, v) -> st_QC._vec_addmul!(v, b, 3, a, d),
            (b, v) -> st_QC._vec_submul!(v, b, 3, a, d),
            (b, v) -> st_QC._gen_addmul_sites!(b, 4, v, sites, 3, a, d),
            (b, v) -> st_QC._gen_submul_sites!(b, 4, v, sites, 3, a, d),
            (b, v) -> st_QC._gen_addmul!(b, 1, b, 5, a, d),
            (b, v) -> st_QC._gen_submul_sites!(b, 1, b, 5, sites, 4, a, d),
            (b, v) -> st_QC._gen_scale!(b, 6, a, d),
            (b, v) -> st_QC._copy_gen!(v, b, 2),
            (b, v) -> st_QC._load_gen!(b, 2, v),
            (b, v) -> st_QC._swap_gens!(b, 1, 4),
            (b, v) -> st_QC._move_gen!(b, 1, 6),
            (b, v) -> st_QC._zero_gen!(b, 5),
        )
        for op in ops
            bref, vref = blk(Int), copy(v0)
            b, v = blk(Eb), Ev.(v0)
            op(bref, vref)
            op(b, v)
            @test ints(b) == ints(bref) && Int.(v) == vref
        end
        b, v = blk(Eb), Ev.(v0)
        bref = blk(Int)
        @test st_QC.dot_xz_col(b, 1, 2, d) == st_QC.dot_xz_col(bref, 1, 2, d)
        @test st_QC.dot_xz_ws_vs_col(v, b, 3, d) == st_QC.dot_xz_ws_vs_col(v0, bref, 3, d)
        @test st_QC.dot_xz_col_vs_zacc(b, 4, v, d) == st_QC.dot_xz_col_vs_zacc(bref, 4, v0, d)
        @test st_QC.dot_xz_ws(v, n, d) == st_QC.dot_xz_ws(v0, n, d)
    end
end

@testset "memory bound" begin
    n = 1024
    slack = 64n * sizeof(Int)
    for d in (3, 2)
        S = StabilizerTableau(d, n; eltype = UInt8)
        D = DestabilizerTableau(d, n; eltype = UInt8)
        @test Base.summarysize(S) <= 2.25 * 2n^2 + slack
        @test Base.summarysize(D) <= 2.25 * 4n^2 + slack
    end
    st_int_matrix_fields(tab) =
        [f for f in fieldnames(typeof(tab)) if getfield(tab, f) isa Matrix{Int}]
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 257), E in st_types(d)
        tab = TT(d, 8; eltype = E)
        @test E === Int || isempty(st_int_matrix_fields(tab))
    end
end

# A pure state reached through random measurements, left non-canonical by a
# final gate, so that `measure!` on a `StabilizerTableau` solves on a copy in
# the rank workspace rather than canonicalizing.
function st_monitored(TT, d, n, E, seed)
    tab = TT(d, n; eltype = E)
    rng = Xoshiro(seed)
    random_state!(rng, tab; m = n)
    for q in 1:n
        if rand(rng) < 0.5
            x = rand(rng, 0:d-1)
            measure!(rng, tab, SinglePauli(q, x, 1, d == 2 ? x : 0))
        end
    end
    apply!(tab, SUM(1, 2))
    apply!(tab, Fourier(3))
    return tab
end

# The product of generators 1 and 2 of `tab` with a phase that makes it
# Hermitian at d = 2: in the stabilizer group up to its phase.
function st_in_span_op(tab)
    n, d = tab.n, tab.d
    S = stabilizer_matrix(tab)
    v = mod.(S[1:2n, 1] .+ S[1:2n, 2], d)
    ph = d == 2 ? mod(sum(v[1:n] .* v[n+1:2n]), 2) : 0
    return GeneralPauli(v, ph)
end

@testset "borrowed workspaces back to back" begin
    n = 12
    A, B, C = [1, 4, 7], [2, 5, 9, 11], [3, 10]
    calls = (
        tab -> is_pure(tab; verify = true),
        tab -> measure!(tab, st_in_span_op(tab); outcome = 0),
        tab -> entanglement_entropy(tab, A),
        tab -> entanglement_negativity(tab, A, B),
        tab -> conditional_mutual_information(tab, A, B, C),
        tab -> is_pure(tab; verify = true),
    )
    for (TT, d, seed) in ((StabilizerTableau, 2, 2101), (StabilizerTableau, 3, 2102),
                          (StabilizerTableau, 251, 2103), (StabilizerTableau, 257, 2104),
                          (DestabilizerTableau, 2, 2201), (DestabilizerTableau, 3, 2202),
                          (DestabilizerTableau, 251, 2203), (DestabilizerTableau, 257, 2204))
        ref = st_monitored(TT, d, n, Int, seed)
        @test !ref.iscanonical
        refvals = Any[]
        refmats = Any[]
        for f in calls
            push!(refvals, f(ref))
            push!(refmats, stabilizer_matrix(ref))
        end
        @test refvals[1] && refvals[end]
        for E in st_types(d)
            tab = st_monitored(TT, d, n, E, seed)
            for (k, f) in enumerate(calls)
                @test f(tab) == refvals[k]
                @test stabilizer_matrix(tab) == refmats[k]
            end
        end
    end
end

@testset "non-commuting input" begin
    # Generator j is Z_j except generators `ja` = X_s and `ia` = Z_s, so the
    # one non-commuting pair is (ja, ia).
    function st_anticommuting(n, ja, ia, s)
        S = zeros(Int, 2n + 1, n)
        for j in 1:n
            S[n + j, j] = 1
        end
        S[:, ja] .= 0
        S[s, ja] = 1
        S[:, ia] .= 0
        S[n + s, ia] = 1
        return S
    end
    st_error(f) = try
        f()
        nothing
    catch e
        e
    end
    for (d, n, ja, ia) in ((2, 70, 3, 69), (2, 5, 2, 4), (3, 5, 2, 4), (257, 5, 2, 4))
        S = st_anticommuting(n, ja, ia, ia)
        for TT in (StabilizerTableau, DestabilizerTableau)
            msgs = String[]
            for E in st_types(d)
                e = st_error(() -> TT(d, S; eltype = E))
                @test e isa ArgumentError
                e isa ArgumentError && push!(msgs, e.msg)
                @test !is_pure(TT(d, S; eltype = E, check = false); verify = true)
            end
            @test all(==(msgs[1]), msgs)
            @test occursin("generators $ja and $ia do not", msgs[1])
        end
    end
end

@testset "differential across element types" begin
    ds = Sys.WORD_SIZE == 64 ? (2, 3, 251, 257, 65521) : (2, 3, 251, 257)
    cases = [(d, n) for d in ds for n in (5, 33)]
    append!(cases, [(3, 300), (251, 300)])
    seed = 0
    for (d, n) in cases, TT in (DestabilizerTableau, StabilizerTableau),
        state in (:mixed, :product), storephase in (n == 5 ? (true, false) : (true,))

        seed += 1
        Es = filter(!=(Int), st_types(d))
        tabs = [TT(d, n; state, storephase, eltype = E) for E in (Int, Es...)]
        nops = n == 300 ? 30 : 120
        @test st_run!(tabs, seed, Xoshiro(10_000 + seed), nops) == 0
    end
end

end
