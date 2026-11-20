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
    for E in (UInt8, UInt16, Int), n in (3, 40)
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

# The Barrett reduction of `x` evaluated in the lane type `T`.
ly_reduce(::Type{T}, x::Integer, d::Integer, M::T, s::Int) where {T<:Signed} =
    ly_QC._barrett_reduce(T(x), T(d), M, s)

const ly_LANES = (Int16, Int32, Int64)

ly_lane_type(S, d) =
    (b = ly_QC._clifford_lane_bytes(S, d); b == 0 ? nothing : only(filter(T -> sizeof(T) == b, ly_LANES)))

# Every `x ∈ 0:2S·d·(d-1)` reduces to `mod(x, d)` in the lane chosen for `(S, d)`.
function ly_lane_reduces_exactly(S, d)
    T = ly_lane_type(S, d)
    M, s = ly_QC._clifford_barrett(T, S, d)
    for x in 0:(2S * d * (d - 1))
        ly_reduce(T, x, d, M, s) == mod(x, d) || return false
    end
    return true
end

@testset "lane selection" begin
    ds = vcat(filter(ly_QC.Primes.isprime, 2:1031), [4093, 16381, 32749, 65521])
    for S in (2, 4, 6), d in ds
        b = ly_QC._clifford_lane_bytes(S, d)
        @test b in (0, 2, 4, 8)
        # Every prime up to 1031 has a lane.
        d <= 1031 && @test b != 0
        for T in ly_LANES
            has = ly_QC._clifford_barrett(T, S, d)[1] != 0
            if b == 0 || sizeof(T) < b
                @test !has
            elseif sizeof(T) == b
                @test has
            end
        end
    end
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

# Gate `i` of a seeded stream on `n` qudits at dimension `d`. Gate 7 is a dense
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
    # Lane boundaries of "lane selection": the largest Int16 d for S = 6 (3)
    # and S = 2 (5), the first Int32 d (5 for S = 4, 6; 7 for S = 2), the first
    # Int64 d (47, 67, 97 for S = 6, 4, 2) and the largest Int32 d (109, 179).
    ds = [2, 3, 5, 7, 47, 61, 67, 97, 109, 179, 251, 257]
    Sys.WORD_SIZE == 64 && push!(ds, 65521)
    seed = 1
    for TT in (StabilizerTableau, DestabilizerTableau), storephase in (true, false),
        d in ds, E in ly_eltypes(d), n in (5, 64, 300)
        seed += 1
        @test ly_check_gate_stream(TT, d, E, n, storephase, seed)
    end
end
