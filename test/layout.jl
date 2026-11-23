using QuditClifford
using Test
using Random

const ly_QC = QuditClifford

# The storage element types a tableau at dimension `d` admits.
ly_eltypes(d) = filter(E -> ly_QC._eltype_admissible(E, d), (UInt8, UInt16, Int))

# The raw X and Z blocks of a tableau, destabilizers included.
ly_blocks(tab::StabilizerTableau) = (tab.stab_x, tab.stab_z)
ly_blocks(tab::DestabilizerTableau) = (tab.stab_x, tab.stab_z, tab.destab_x, tab.destab_z)

ly_matrices(tab::StabilizerTableau) = (stabilizer_matrix(tab),)
ly_matrices(tab::DestabilizerTableau) = (stabilizer_matrix(tab), destabilizer_matrix(tab))

# Allocations of one switch each way, measured after a warm-up round trip.
function ly_switch_allocs(tab)
    qudit_major!(tab)
    generator_major!(tab)
    a = @allocated qudit_major!(tab)
    b = @allocated generator_major!(tab)
    return a, b
end

# Allocations of `copy!` into `dst` from a source of the other layout, measured
# after a warm-up; the sources alternate so every measured call switches the
# layout of `dst`.
function ly_copy_allocs(dst, src_a, src_b)
    copy!(dst, src_a)
    copy!(dst, src_b)
    a = @allocated copy!(dst, src_a)
    b = @allocated copy!(dst, src_b)
    return a + b
end

@testset "layout keyword and query" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 257), E in ly_eltypes(d)
        n = 4
        presets = (
            (; state = :mixed),
            (; state = :product, basis = :X),
            (; state = :product, basis = :Y),
            (; state = :product, basis = :Z),
            (; state = :ghz),
        )
        for kw in presets
            g = TT(d, n; kw..., eltype = E, layout = :generator)
            q = TT(d, n; kw..., eltype = E, layout = :qudit)
            @test tableau_layout(g) === :generator
            @test tableau_layout(TT(d, n; kw..., eltype = E)) === ly_QC._DEFAULT_LAYOUT[]
            @test tableau_layout(TT(d, n; kw..., eltype = E, layout = :generator)) === :generator
            @test tableau_layout(q) === :qudit
            @test q isa typeof(g)
            @test ly_matrices(q) == ly_matrices(g)
        end
        raw = stabilizer_matrix(TT(d, n; state = :ghz))
        g = TT(d, raw; eltype = E, layout = :generator)
        q = TT(d, raw; eltype = E, layout = :qudit)
        qt = TT(d, permutedims(raw); eltype = E, layout = :qudit)
        @test tableau_layout(g) === :generator
        @test tableau_layout(q) === :qudit
        @test tableau_layout(qt) === :qudit
        @test ly_matrices(q) == ly_matrices(g)
        @test ly_matrices(qt) == ly_matrices(g)
    end
    for TT in (StabilizerTableau, DestabilizerTableau)
        @test_throws ArgumentError TT(3, 4; layout = :rows)
        @test_throws ArgumentError TT(3, 4, :ghz; layout = :rows)
        @test_throws ArgumentError TT(3, stabilizer_matrix(TT(3, 2; state = :ghz)); layout = :rows)
    end
end

@testset "switch round trip" begin
    cases = vcat(
        [(UInt8, 3, n) for n in (1, 2, 5, 255, 256, 257, 512, 1023, 1024)],
        [(UInt16, 257, n) for n in (127, 128, 129)],
        [(Int, 3, n) for n in (31, 32, 33)],
    )
    rng = Xoshiro(20261120)
    for TT in (StabilizerTableau, DestabilizerTableau), (E, d, n) in cases
        tab = TT(d, n; eltype = E, layout = :generator)
        random_state!(rng, tab; m = n ÷ 2)
        P = map(b -> copy(parent(b)), ly_blocks(tab))
        X0 = map(copy, ly_blocks(tab))
        mats = ly_matrices(tab)
        phase, m, canon, cache = copy(tab.phase), tab.m, tab.iscanonical, copy(tab.xdotz_cache)

        @test qudit_major!(tab) === tab
        @test tableau_layout(tab) === :qudit
        @test tab.stab_x == permutedims(X0[1])
        @test all(map((b, b0) -> b == permutedims(b0), ly_blocks(tab), X0))
        @test ly_matrices(tab) == mats
        @test tab.phase == phase
        @test tab.m == m
        @test tab.iscanonical == canon
        @test tab.xdotz_cache == cache

        Q = map(b -> copy(parent(b)), ly_blocks(tab))
        @test qudit_major!(qudit_major!(tab)) === tab
        @test map(parent, ly_blocks(tab)) == Q

        @test generator_major!(tab) === tab
        @test tableau_layout(tab) === :generator
        @test all(map(==, map(parent, ly_blocks(tab)), P))
        @test generator_major!(tab) === tab
        @test all(map(==, map(parent, ly_blocks(tab)), P))
        @test ly_matrices(tab) == mats

        @test ly_switch_allocs(tab) == (0, 0)
        @test tableau_layout(tab) === :generator
    end
end

@testset "copy and copy! across layouts" begin
    rng = Xoshiro(7)
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3), n in (5, 300)
        src = TT(d, n; layout = :generator)
        random_state!(rng, src; m = n ÷ 2)
        q = qudit_major!(copy(src))
        @test tableau_layout(q) === :qudit
        c = copy(q)
        @test tableau_layout(c) === :qudit
        @test ly_matrices(c) == ly_matrices(src)
        @test all(map(==, ly_blocks(c), ly_blocks(q)))

        g = TT(d, n; layout = :generator)
        copy!(g, q)
        @test tableau_layout(g) === :qudit
        @test ly_matrices(g) == ly_matrices(src)
        @test g.m == src.m
        @test ly_copy_allocs(g, q, src) == 0
        @test tableau_layout(g) === :generator

        g2 = TT(d, n; layout = :qudit)
        copy!(g2, src)
        @test tableau_layout(g2) === :generator
        @test ly_matrices(g2) == ly_matrices(src)
        copy!(q, g2)
        @test tableau_layout(q) === :generator
        @test ly_matrices(q) == ly_matrices(src)
    end
end

@testset "reset! keeps layout" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3), E in ly_eltypes(d)
        n = 6
        q = TT(d, n; state = :ghz, eltype = E, layout = :qudit)
        @test reset!(q; state = :product) === q
        @test tableau_layout(q) === :qudit
        @test ly_QC._storage_eltype(q) === E
        fresh = TT(d, n; state = :product, eltype = E)
        @test ly_matrices(q) == ly_matrices(fresh)
        @test q.m == fresh.m
        @test q.xdotz_cache == fresh.xdotz_cache
        reset!(q, :ghz)
        @test tableau_layout(q) === :qudit
        @test ly_matrices(q) == ly_matrices(TT(d, n; state = :ghz, eltype = E))
        reset!(q; state = :product, basis = [:X, :Y, :Z, :X, :Y, :Z])
        @test ly_matrices(q) ==
              ly_matrices(TT(d, n; state = :product, basis = [:X, :Y, :Z, :X, :Y, :Z], eltype = E))
        reset!(q)
        @test q.m == 0
        @test tableau_layout(q) === :qudit
    end
end

@testset "show names the layout" begin
    @test occursin("Storage:          qudit-major, UInt8",
                   sprint(show, StabilizerTableau(3, 2; layout = :qudit)))
    @test occursin("Storage:          generator-major, UInt8",
                   sprint(show, StabilizerTableau(3, 2; layout = :generator)))
    for TT in (StabilizerTableau, DestabilizerTableau), state in (:ghz, :mixed)
        g = TT(3, 4; state, eltype = Int, layout = :generator)
        q = TT(3, 4; state, eltype = Int, layout = :qudit)
        sg, sq = sprint(show, g), sprint(show, q)
        @test occursin("Storage:          qudit-major, $Int", sq)
        @test replace(sq, "qudit-major" => "generator-major") == sg
    end
end

@testset "in-place transpose leaves the padding alone" begin
    for E in (UInt8, UInt16, Int), n in (1, 3, 31, 32, 33, 65, 257)
        A = ly_QC._padded_block(E, n, 1024)
        P = parent(A)
        rng = Xoshiro(n)
        P .= rand(rng, E, size(P)...)
        P0 = copy(P)
        A0 = copy(A)
        ly_QC._transpose_block!(A)
        @test A == permutedims(A0)
        cols = parentindices(A)[2]
        @test P[n+1:end, :] == P0[n+1:end, :]
        @test P[:, setdiff(axes(P, 2), cols)] == P0[:, setdiff(axes(P, 2), cols)]
        ly_QC._transpose_block!(A)
        @test P == P0
    end
end

@testset "dense states of a qudit-major tableau" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
        n = 4
        g = TT(d, n; layout = :generator)
        random_state!(Xoshiro(20261120 + d), g)
        q = qudit_major!(copy(g))
        @test state_vector(q) == state_vector(g)
        @test ket(q) == ket(g)
        @test density_matrix(q) == density_matrix(g)
        traceout!(g, 2)
        traceout!(q, 2)
        @test density_matrix(q) == density_matrix(g)
        @test tableau_layout(q) === :qudit
        @test ly_matrices(q) == ly_matrices(g)
    end
end

@testset "generator-major copy of a block" begin
    for E in (UInt8, UInt16, Int), n in (1, 3, 33, 40, 65, 129)
        g = ly_QC.XZBlock{ly_QC.GeneratorMajor}(ly_QC._stab_blocks(E, n)...)
        rng = Xoshiro(n)
        g.x .= rand(rng, E(0):E(2), n, n)
        g.z .= rand(rng, E(0):E(2), n, n)
        X0, Z0 = copy(g.x), copy(g.z)
        cg = ly_QC._generator_major_copy(g)
        @test cg isa ly_QC.XZBlock{ly_QC.GeneratorMajor, Matrix{E}}
        @test cg.x == X0 && cg.z == Z0
        @test cg.x !== g.x
        ly_QC._transpose_block!(g.x)
        ly_QC._transpose_block!(g.z)
        qb = ly_QC.XZBlock{ly_QC.QuditMajor}(g.x, g.z)
        cq = ly_QC._generator_major_copy(qb)
        @test cq isa ly_QC.XZBlock{ly_QC.GeneratorMajor, Matrix{E}}
        @test cq.x == X0 && cq.z == Z0
        @test all(ly_QC._xget(cq, i, j) == ly_QC._xget(qb, i, j) &&
                  ly_QC._zget(cq, i, j) == ly_QC._zget(qb, i, j) for i in 1:n, j in 1:n)
        cq.x[1, 1] = E(7)
        @test qb.x == permutedims(X0)
    end
end

@testset "workspace x·z dot and matrix loads in both layouts" begin
    for E in (UInt8, UInt16, Int), n in (1, 3, 33), d in (2, 3)
        g = ly_QC.XZBlock{ly_QC.GeneratorMajor}(ly_QC._stab_blocks(E, n)...)
        rng = Xoshiro(10n + d)
        g.x .= rand(rng, E(0):E(d - 1), n, n)
        g.z .= rand(rng, E(0):E(d - 1), n, n)
        ws = rand(rng, E(0):E(d - 1), 2n)
        gm = [ly_QC.dot_xz_ws_vs_col(ws, g, j, d) for j in 1:n]
        ly_QC._transpose_block!(g.x)
        ly_QC._transpose_block!(g.z)
        qb = ly_QC.XZBlock{ly_QC.QuditMajor}(g.x, g.z)
        @test [ly_QC.dot_xz_ws_vs_col(ws, qb, j, d) for j in 1:n] == gm
    end
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3), E in ly_eltypes(d)
        ref = TT(d, 5; state = :Z, eltype = E)
        random_state!(Xoshiro(d), ref)
        q = TT(d, 5; state = :Z, eltype = E, layout = :qudit)
        ly_QC._load_stabilizer_matrix!(q, stabilizer_matrix(ref))
        @test tableau_layout(q) === :qudit
        @test stabilizer_matrix(q) == stabilizer_matrix(ref)
        if TT === DestabilizerTableau
            ly_QC._load_destabilizer_matrix!(q, destabilizer_matrix(ref))
            @test destabilizer_matrix(q) == destabilizer_matrix(ref)
        end
    end
end

# Allocations of one call, the least of three measurements after a warm-up
# call. One function per operation: a shared varargs helper allocates on
# Julia 1.10.
const ly_skip_alloc_compare =
    Sys.WORD_SIZE == 32 && Base.invokelatest(QuditClifford._default_qubit_eltype) === Bits

function ly_stabilizer_matrix_allocs(t)
    stabilizer_matrix(t)
    return minimum(@allocated(stabilizer_matrix(t)) for _ in 1:3)
end
function ly_destabilizer_matrix_allocs(t)
    destabilizer_matrix(t)
    return minimum(@allocated(destabilizer_matrix(t)) for _ in 1:3)
end
function ly_equal_allocs(a, b)
    a == b
    return minimum(@allocated(a == b) for _ in 1:3)
end

@testset "generator reads agree across layouts at sizes off the tile" begin
    for E in (UInt8, UInt16, Int), TT in (StabilizerTableau, DestabilizerTableau),
        n in (1, 2, 5, 31, 33, 65, 257)
        g = TT(3, n; eltype = E, layout = :generator)
        random_state!(Xoshiro(20261121 + n), g)
        @test !g.iscanonical
        q = qudit_major!(copy(g))
        @test stabilizer_matrix(q) == stabilizer_matrix(g)
        @test stabilizer_matrix(q; layout = :qudit) == stabilizer_matrix(g; layout = :qudit)
        if TT === DestabilizerTableau
            @test destabilizer_matrix(q) == destabilizer_matrix(g)
            @test destabilizer_matrix(q; layout = :qudit) ==
                  destabilizer_matrix(g; layout = :qudit)
        end
        # The same state in another generator basis, so `==` and `hash` go
        # through a canonical copy of the non-canonical side.
        gc = copy(g)
        canonicalize!(gc)
        qc = copy(q)
        canonicalize!(qc)
        @test stabilizer_matrix(qc) == stabilizer_matrix(gc)
        for (a, b) in ((q, g), (g, q), (q, q), (q, gc), (gc, q), (q, qc), (qc, q))
            @test a == b
            @test hash(a) == hash(b)
        end
        other = copy(g)
        random_state!(Xoshiro(1 + n), other)
        if other != g
            qo = qudit_major!(copy(other))
            @test q != qo && qo != q && q != other && g != qo
        end
        if n <= 5
            @test ket(q) == ket(g)
            @test state_vector(q) == state_vector(g)
        end
        @test tableau_layout(q) === :qudit && !q.iscanonical
        # Both layouts allocate only the result. Windows reports the same
        # allocations a few bytes apart from call to call, so the counts
        # agree within 64 bytes, far less than any extra n × n buffer at
        # n ≥ 31. On 32-bit runs with packed default storage
        # (`QC_TEST_ELTYPE=bits`) the qudit-major read reports about 3 KB more
        # in one case, so the comparison is skipped in that configuration.
        @test abs(ly_stabilizer_matrix_allocs(q) - ly_stabilizer_matrix_allocs(g)) <= 64 skip =
            ly_skip_alloc_compare
        if TT === DestabilizerTableau
            @test abs(ly_destabilizer_matrix_allocs(q) - ly_destabilizer_matrix_allocs(g)) <= 64
        end
        qq = copy(q)
        @test ly_equal_allocs(q, qq) == 0
        @test ly_equal_allocs(qc, copy(qc)) == 0
    end
end

# The Barrett reduction of `x` evaluated in the lane type `T`, with the product
# formed in the type of `M`.
ly_reduce(::Type{T}, x::Integer, d::Integer, M::Signed, s::Int) where {T<:Signed} =
    ly_QC._barrett_reduce(T(x), T(d), M, s)

const ly_LANES = (Int16, Int32, Int64)

# The type a lane forms its Barrett product in: the lane itself when it has
# its own constant (and always for `Int64`), else the next wider type.
ly_product_type(T, wide) = wide ? widen(T) : T

# `T` holds the kernel's largest unreduced value `X = 2S·d·(d-1)`.
ly_holds(T, S, d) = big(2S) * d * (d - 1) <= typemax(T)

ly_has_constant(W, S, d) = ly_QC._clifford_barrett(W, S, d)[1] != 0

# A lane is usable if it holds `X` and has its own or (below Int64) a widened constant.
ly_lane_ok(T, S, d) = T === Int64 ? ly_has_constant(Int64, S, d) :
    ly_holds(T, S, d) && (ly_has_constant(T, S, d) || ly_has_constant(widen(T), S, d))

ly_lane_type(S, d) =
    (b = ly_QC._clifford_lane_bytes(S, d); b == 0 ? nothing : only(filter(T -> sizeof(T) == b, ly_LANES)))

# Every `x ∈ 0:2S·d·(d-1)` reduces to `mod(x, d)` in the lane chosen for `(S, d)`,
# with the constant `_clifford_lane` returns.
function ly_lane_reduces_exactly(S, d)
    T = ly_lane_type(S, d)
    _, M64, s, wide = ly_QC._clifford_lane(S, d)
    W = ly_product_type(T, wide)
    M = M64 % W
    (M, s) == ly_QC._clifford_barrett(W, S, d) || return false
    for x in 0:(2S * d * (d - 1))
        ly_reduce(T, x, d, M, s) == mod(x, d) || return false
    end
    return true
end

@testset "lane selection" begin
    ds = vcat(filter(ly_QC.Primes.isprime, 2:1031), [4093, 13399, 16381, 23189, 32749, 65521])
    nwide = 0
    for S in (2, 4, 6), d in ds
        b, M, s, wide = ly_QC._clifford_lane(S, d)
        @test b == ly_QC._clifford_lane_bytes(S, d)
        @test b in (0, 2, 4, 8)
        # Every prime up to 1031 has a lane.
        d <= 1031 && @test b != 0
        for T in ly_LANES
            if b == 0 || sizeof(T) < b
                @test !ly_lane_ok(T, S, d)
            elseif sizeof(T) == b
                @test ly_holds(T, S, d)
                # The lane's own constant when it has one, else the widened one.
                @test wide == !ly_has_constant(T, S, d)
                @test (M, s) == Tuple(Int64.(ly_QC._clifford_barrett(ly_product_type(T, wide), S, d)))
                @test M != 0
                @test !(T === Int64 && wide)
                nwide += wide
            end
        end
    end
    @test nwide > 0
    # The lanes of the boundary d in "qudit-major gate kernel".
    @test [ly_QC._clifford_lane_bytes(2, d) for d in (89, 97, 23189)] == [2, 4, 8]
    @test [ly_QC._clifford_lane_bytes(4, d) for d in (61, 67)] == [2, 4]
    @test [ly_QC._clifford_lane_bytes(6, d) for d in (43, 47, 13399)] == [2, 4, 8]
    for S in (2, 6), d in filter(ly_QC.Primes.isprime, 2:251)
        @test ly_lane_reduces_exactly(S, d)
    end
end

@testset "coeff_workspace" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 257), E in ly_eltypes(d),
        layout in (:generator, :qudit)
        n = 7
        tab = TT(d, n; state = :ghz, eltype = E, layout)
        @test size(tab.coeff_workspace) == (n, 3)
        @test eltype(tab.coeff_workspace) === E
        tab.coeff_workspace .= E(1)
        c = copy(tab)
        @test size(c.coeff_workspace) == (n, 3) && all(iszero, c.coeff_workspace)
        @test c.coeff_workspace !== tab.coeff_workspace
    end
end

# Entry-wise equality of the generators, phases and caches of a generator-major
# and a qudit-major tableau, without building the stacked matrices.
function ly_same_state(g, q)
    G, Q = ly_QC.GeneratorMajor(), ly_QC.QuditMajor()
    g.m == q.m || return false
    g.phase == q.phase || return false
    g.xdotz_cache == q.xdotz_cache || return false
    pairs = g isa DestabilizerTableau ?
        ((ly_QC._stab(g, G), ly_QC._stab(q, Q)), (ly_QC._destab(g, G), ly_QC._destab(q, Q))) :
        ((ly_QC._stab(g, G), ly_QC._stab(q, Q)),)
    for (bg, bq) in pairs, j in 1:g.n, t in 1:g.n
        ly_QC._xget(bg, t, j) == ly_QC._xget(bq, t, j) || return false
        ly_QC._zget(bg, t, j) == ly_QC._zget(bq, t, j) || return false
    end
    return true
end

# Gate `i` of a seeded stream on `n` qudits at dimension `d`. Gate 8 is a dense
# 4-qudit operator when `n ≥ 4`.
function ly_stream_gate(rng, d, n, i)
    kmax = min(n, 3)
    tg = randperm(rng, n)
    r = rand(rng, 1:d - 1)
    kind = mod1(i, 9)
    if n == 1 && kind in (1, 2, 3)
        kind = 4
    end
    kind == 1 && return SUM(tg[1], tg[2], r)
    kind == 2 && return CPhase(tg[1], tg[2], r)
    kind == 3 && return SWAP(tg[1], tg[2])
    kind == 4 && return Fourier(tg[1])
    kind == 5 && return Phase(tg[1])
    kind == 6 && return Multiplier(tg[1], r)
    kind == 7 && return PauliGate(tg[1], rand(rng, 0:d - 1), rand(rng, 0:d - 1))
    k = i == 8 && n >= 4 ? 4 : rand(rng, 1:kmax)
    return random_clifford(rng, d, tg[1:k])
end

function ly_check_gate_stream(TT, d, E, n, storephase, seed)
    rng = Xoshiro(seed)
    g = TT(d, n; eltype = E, storephase, layout = :generator)
    random_state!(rng, g; m = max(1, n - 2))
    q = qudit_major!(copy(g))
    ok = ly_same_state(g, q)
    for i in 1:200
        gate = ly_stream_gate(rng, d, n, i)
        if i % 25 == 0
            prep = ly_QC._prepare(gate, d, g.inversemod, storephase; force_safe = true)
            ly_QC._apply_prepared!(g, prep)
            ly_QC._apply_prepared!(q, prep)
        else
            apply!(g, gate)
            apply!(q, gate)
        end
        ok &= ly_same_state(g, q)
        ok || (@error "layouts differ" TT d E n storephase i gate; break)
    end
    ok &= ly_matrices(g) == ly_matrices(q)
    return ok
end

@testset "qudit-major gate kernel" begin
    # Lane boundaries of "lane selection": the largest Int16 d for S = 6, 4, 2
    # (43, 61, 89), the first Int32 d (47, 67, 97), Int32 lanes with `UInt16`
    # storage (4093) and Int64 lanes (13399 for S = 6, 23189 for S = 2). Each
    # lane runs with its own constant (d = 3; 101 and 61 at S = 6 in Int32)
    # and with the widened one (d = 5 at S = 4, 6; 43-97, 251, 4093).
    ds = [2, 3, 5, 43, 47, 61, 67, 89, 97, 101, 251, 257, 4093, 13399, 23189]
    Sys.WORD_SIZE == 64 && push!(ds, 65521)
    seed = 1
    for TT in (StabilizerTableau, DestabilizerTableau), storephase in (true, false),
        d in ds, E in ly_eltypes(d), n in (5, 64, 300)
        seed += 1
        @test ly_check_gate_stream(TT, d, E, n, storephase, seed)
    end
end

# Measured operator `i` of a seeded stream on `n ≥ 4` qudits at dimension `d`:
# every built-in Pauli type, and a dense `GeneralPauli` every fifth call.
function ly_measure_op(rng, d, n, i)
    tg = randperm(rng, n)
    e() = rand(rng, -d:2d)
    ph = rand(rng, 0:3)
    kind = mod1(i, 5)
    if kind == 1
        r = rand(rng)
        r < 0.4 && return SinglePauli(tg[1], 0, 1)
        r < 0.6 && return SinglePauli(tg[1], 1, 0)
        return SinglePauli(tg[1], e(), e(), ph)
    end
    kind == 2 && return DoublePauli(tg[1], e(), e(), tg[2], e(), e(), ph)
    kind == 3 && return TriplePauli(tg[1], e(), e(), tg[2], e(), e(), tg[3], e(), e(), ph)
    kind == 4 && return NPauli(Tuple(tg[1:4]), ntuple(_ -> e(), 4), ntuple(_ -> e(), 4), ph)
    return GeneralPauli(rand(rng, 0:d-1, 2n), ph)
end

@testset "same rng stream in both layouts" begin
    n = 40
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31), storephase in (true, false)
        g = TT(d, n; state = :mixed, storephase, layout = :generator)
        q = TT(d, n; state = :mixed, storephase, layout = :qudit)
        rng_g = Xoshiro(7)
        rng_q = Xoshiro(7)
        src = Xoshiro(1000d + n)
        ok = true
        for i in 1:300
            op = ly_measure_op(src, d, n, i)
            tg = measure!(rng_g, g, op; phase_policy = 1)
            tq = measure!(rng_q, q, op; phase_policy = 1)
            ok &= tg == tq && ly_matrices(g) == ly_matrices(q) && ly_same_state(g, q) &&
                  g.iscanonical == q.iscanonical
            ok || (@error "layouts differ" TT d storephase i op tg tq; break)
            if rand(src) < 0.5
                gate = ly_stream_gate(src, d, n, i)
                apply!(g, gate)
                apply!(q, gate)
            end
        end
        @test ok
        @test tableau_layout(q) === :qudit
        @test rand(rng_g) == rand(rng_q)
    end
end

# Fixtures for the qudit-major canonicalization and dephasing kernels, all
# generator-major: a brickwork circuit with Z measurements at rate `p`, the
# same with a third of the qudits traced out (mixed), a random pure state and
# a GHZ state.
function ly_fixture(TT, d, n, E, storephase, kind, seed)
    rng = Xoshiro(seed)
    tab = TT(d, n; state = kind === :ghz ? :ghz : :Z, eltype = E, storephase,
             layout = :generator)
    kind === :ghz && return tab
    if kind === :random
        random_state!(rng, tab)
        return tab
    end
    for l in 1:8
        for q in (isodd(l) ? 1 : 2):2:(n - 1)
            apply!(tab, random_clifford(rng, d, [q, q + 1]))
        end
        for q in 1:n
            rand(rng) < 0.15 && measure!(rng, tab, SinglePauli(q, 0, 1))
        end
    end
    kind === :mixed && traceout!(tab, randperm(rng, n)[1:(n ÷ 3)])
    return tab
end

ly_same_full(g, q) = ly_same_state(g, q) && g.iscanonical == q.iscanonical &&
    g.pivcol_of_row == q.pivcol_of_row && ly_matrices(g) == ly_matrices(q)

@testset "qudit-major canonicalize!" begin
    L = ly_QC.QuditMajor()
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31), E in ly_eltypes(d),
        storephase in (true, false), kind in (:monitored, :mixed, :random, :ghz), n in (5, 48)
        g = ly_fixture(TT, d, n, E, storephase, kind, 7d + n)
        q = qudit_major!(copy(g))
        # Divisor 1 updates row by row whenever a generator has a nonzero β;
        # `typemax(Int)` takes the column updates whenever one has.
        s = copy(q)
        w = copy(q)
        canonicalize!(g)
        canonicalize!(q)
        ly_QC._canonicalize_qm!(s, L, 1)
        ly_QC._canonicalize_qm!(w, L, typemax(Int))
        ok = ly_same_full(g, q) && ly_same_full(g, s) && ly_same_full(g, w)
        # A canonical tableau stays as it is.
        canonicalize!(g)
        canonicalize!(q)
        ok &= ly_same_full(g, q)
        ok || @error "layouts differ" TT d E storephase kind n
        @test ok
    end
end

@testset "qudit-major canonicalize! dual accumulators" begin
    # The dual sums run in UInt16 at d = 31, UInt32 at d = 251 and UInt64 at
    # d = 65521 (64-bit hosts only: 65520^2 overflows a 32-bit `Int`). Row by
    # row, the path for a `d` at which even UInt64 overflows, and the wider
    # accumulators are also forced at d = 3, on every host.
    L = ly_QC.QuditMajor()
    cases = Any[(31, UInt8), (251, UInt8)]
    Sys.WORD_SIZE == 64 && push!(cases, (65521, UInt16))
    for (d, E) in cases, storephase in (true, false)
        g = ly_fixture(DestabilizerTableau, d, 24, E, storephase, :random, d % 1000)
        q = qudit_major!(copy(g))
        w = copy(q)
        canonicalize!(g)
        canonicalize!(q)
        ly_QC._canonicalize_qm!(w, L, typemax(Int))
        @test ly_same_full(g, q) && ly_same_full(g, w)
    end
    for dual in (ly_QC._QMDualRows(), ly_QC._QMDualAcc{UInt64}()), div in (1, typemax(Int)),
        kind in (:monitored, :random)
        g = ly_fixture(DestabilizerTableau, 3, 40, UInt8, true, kind, 5)
        q = qudit_major!(copy(g))
        canonicalize!(g)
        ly_QC._canonicalize_qm!(ly_QC._stab(q, L), q.phase, ly_QC._destab(q, L), dual, q, div)
        @test ly_same_full(g, q)
    end
end

@testset "qudit-major canonicalize! at m = 0 and n = 1" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31), storephase in (true, false)
        cases = Any[TT(d, 7; state = :mixed, storephase, layout = :generator)]
        for state in (:mixed, :Z, :ghz)
            push!(cases, TT(d, 1; state, storephase, layout = :generator))
        end
        g1 = TT(d, 1; state = :Z, storephase, layout = :generator)
        apply!(g1, random_clifford(Xoshiro(d), d, [1]))
        push!(cases, g1)
        for g in cases
            q = qudit_major!(copy(g))
            canonicalize!(g)
            canonicalize!(q)
            @test ly_same_full(g, q)
        end
    end
end

@testset "qudit-major paired-dual update, forced sparse and dense" begin
    # `_qm_orthogonalize_duals!` with divisor 1 updates only the rows of the
    # paired duals whenever one is nonzero; `typemax(Int)` always updates whole
    # columns. Both must equal the generic update on a generator-major copy.
    G, Q = ly_QC.GeneratorMajor(), ly_QC.QuditMajor()
    for d in (2, 3, 31), E in ly_eltypes(d), kind in (:monitored, :random)
        n = 64
        g = ly_fixture(DestabilizerTableau, d, n, E, true, kind, 17d)
        ok = true
        for col in (1, g.m ÷ 2, g.m)
            g2 = copy(g)
            a = qudit_major!(copy(g))
            b = copy(a)
            supp = g2.support_workspace
            nsupp = ly_QC._column_support!(supp, ly_QC._stab(g2, G), col)
            sites = g2.zacc_workspace
            nsites = ly_QC._column_support!(sites, ly_QC._destab(g2, G), col)
            ly_QC._orthogonalize_duals!(g2, G, col, g2.m, supp, nsupp, sites, nsites,
                                        ly_QC._pivot_support_is_sparse(nsites, n))
            for (t, dv) in ((a, 1), (b, typemax(Int)))
                ns = ly_QC._column_support!(t.support_workspace, ly_QC._stab(t, Q), col)
                nd = ly_QC._column_support!(t.zacc_workspace, ly_QC._destab(t, Q), col)
                ly_QC._qm_orthogonalize_duals!(t, Q, col, t.m, t.support_workspace, ns,
                                               t.zacc_workspace, nd, dv)
                ok &= ly_matrices(t) == ly_matrices(g2)
            end
        end
        ok || @error "paired-dual update differs" d E kind
        @test ok
    end
end

@testset "traceout! and reset! with sites in a workspace" begin
    # The qudit-major `dephase!` writes c_workspace and zacc_workspace; sites
    # stored there are copied first, so both layouts agree.
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3), ws in (:zacc, :c)
        n = 24
        g = ly_fixture(TT, d, n, ly_QC._default_eltype(d), true, :monitored, 19d)
        q = qudit_major!(copy(g))
        for (t, f!) in ((g, traceout!), (q, traceout!))
            w = ws === :zacc ? t.zacc_workspace : t.c_workspace
            w[1:3] .= (2, 9, 17)
            f!(t, view(w, 1:3))
        end
        @test ly_same_full(g, q)
        for t in (g, q)
            w = ws === :zacc ? t.zacc_workspace : t.c_workspace
            w[1:2] .= (4, 11)
            reset!(t, view(w, 1:2); basis = :X)
        end
        @test ly_same_full(g, q)
    end
end

@testset "qudit-major canonicalize! inside expect! and measure!" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31), storephase in (true, false)
        n = 40
        g = ly_fixture(TT, d, n, ly_QC._default_eltype(d), storephase, :mixed, 3d)
        q = qudit_major!(copy(g))
        rng = Xoshiro(d)
        ok = true
        for i in 1:60
            op = ly_measure_op(rng, d, n, i)
            if isodd(i)
                ok &= expect_int!(g, op) == expect_int!(q, op)
            else
                ok &= measure!(Xoshiro(i), g, op; phase_policy = 1) ==
                      measure!(Xoshiro(i), q, op; phase_policy = 1)
            end
            ok &= ly_same_full(g, q)
            ok || (@error "layouts differ" TT d storephase i op; break)
        end
        @test ok
    end
end

@testset "qudit-major dephase!, traceout! and reset!" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31), E in ly_eltypes(d),
        storephase in (true, false), kind in (:monitored, :random)
        n = 48
        g = ly_fixture(TT, d, n, E, storephase, kind, 11d)
        q = qudit_major!(copy(g))
        rng = Xoshiro(5d)
        ok = true
        for i in 1:30
            r = rand(rng, 1:4)
            if r == 1
                op = ly_measure_op(rng, d, n, i)
                dephase!(g, op)
                dephase!(q, op)
            elseif r == 2
                v = rand(rng, 0:(d - 1), 2n)
                dephase!(g, v)
                dephase!(q, v)
            elseif r == 3
                s = randperm(rng, n)[1:rand(rng, 1:4)]
                traceout!(g, s)
                traceout!(q, s)
            else
                s = randperm(rng, n)[1:rand(rng, 1:4)]
                basis = rand(rng, (:X, :Y, :Z))
                reset!(g, s; basis)
                reset!(q, s; basis)
            end
            ok &= ly_same_full(g, q)
            ok || (@error "layouts differ" TT d E storephase kind i r; break)
        end
        @test ok
    end
end

# Allocations of one call on a copy of `tab`, measured after a warm-up call on
# another copy. One function per operation: a shared varargs helper allocates
# on Julia 1.10.
ly_canonicalize_allocs(t) = @allocated canonicalize!(t)
ly_dephase_allocs(t, op) = @allocated dephase!(t, op)
ly_traceout_allocs(t, sites) = @allocated traceout!(t, sites)
function ly_allocs(f, tab, args...)
    f(copy(tab), args...)
    return f(copy(tab), args...)
end

@testset "canonicalize!, dephase! and traceout! allocate nothing" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31), E in ly_eltypes(d),
        layout in (:generator, :qudit)
        n = 48
        g = ly_fixture(TT, d, n, E, true, :mixed, 13d)
        layout === :qudit && qudit_major!(g)
        @test ly_allocs(ly_canonicalize_allocs, g) == 0
        @test ly_allocs(ly_dephase_allocs, g, SinglePauli(5, 1, 0)) == 0
        @test ly_allocs(ly_traceout_allocs, g, [3, 9]) == 0
    end
end
