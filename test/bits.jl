using QuditClifford, Test, Random
const bt_QC = QuditClifford

# `st_padding_zero` lives in test/storage_types.jl, which the suite includes
# first; a standalone run of this file includes it here.
isdefined(Main, :st_padding_zero) || include(joinpath(@__DIR__, "storage_types.jl"))

const bt_TYPES = (StabilizerTableau, DestabilizerTableau)

# Every observable of `t` that must match the `UInt8` tableau `u`.
function bt_same(t, u)
    ok = stabilizer_matrix(t) == stabilizer_matrix(u) && t.m == u.m &&
         t.iscanonical == u.iscanonical && t.pivcol_of_row == u.pivcol_of_row
    t isa DestabilizerTableau && (ok &= destabilizer_matrix(t) == destabilizer_matrix(u) &&
                                         collect(t.xdotz_cache) == collect(u.xdotz_cache))
    return ok
end

# The preset constructor arguments of the construction test.
function bt_presets(n)
    mixedbasis = [(:X, :Y, :Z)[mod1(q, 3)] for q in 1:n]
    return ((:mixed, :Z), (:product, :X), (:product, :Y), (:product, :Z),
            (:product, mixedbasis), (:ghz, :Z))
end

@testset "Bits" begin

@testset "Bits construction and matrices" begin
    seed = 0
    for TT in bt_TYPES, layout in (:generator, :qudit), n in (1, 2, 63, 64, 65, 130),
        storephase in (true, false)

        for (state, basis) in bt_presets(n)
            t = TT(2, n; state, basis, storephase, eltype = Bits, layout)
            u = TT(2, n; state, basis, storephase, eltype = UInt8, layout)
            @test tableau_eltype(t) === Bits
            @test tableau_layout(t) === layout
            @test bt_same(t, u)
            @test t == u && hash(t) == hash(u)
            @test st_padding_zero(t)
        end

        seed += 1
        r = TT(2, n; storephase, eltype = UInt8, layout = :generator)
        random_state!(Xoshiro(seed), r; m = max(n - 1, 1))
        S = stabilizer_matrix(r)
        t = TT(2, S; m = r.m, eltype = Bits, layout)
        u = TT(2, S; m = r.m, eltype = UInt8, layout)
        @test tableau_eltype(t) === Bits && tableau_layout(t) === layout
        @test bt_same(t, u)
        @test t == u && hash(t) == hash(u) && t == r
        @test st_padding_zero(t)
        # The transposed raw matrix gives the same tableau.
        @test bt_same(TT(2, permutedims(S); m = r.m, eltype = Bits, layout), u)
    end
end

@testset "Bits shared methods" begin
    for TT in bt_TYPES, layout in (:generator, :qudit), n in (3, 65)
        t = TT(2, n; state = :ghz, eltype = Bits, layout)
        c = copy(t)
        @test typeof(c) === typeof(t) && tableau_layout(c) === layout
        @test tableau_eltype(c) === Bits && bt_same(c, t) && st_padding_zero(c)
        @test parent(c.stab_x) !== parent(t.stab_x)
        @test iszero(c.word_workspace) && size(c.word_workspace) == (32, 4)

        dst = TT(2, n; eltype = Bits, layout = layout === :qudit ? :generator : :qudit)
        copy!(dst, t)
        @test tableau_layout(dst) === layout && bt_same(dst, t) && st_padding_zero(dst)
        copy!(dst, t)
        @test (@allocated copy!(dst, t)) == 0

        u = TT(2, n; state = :product, eltype = UInt8, layout)
        @test reset!(t; state = :product) === t
        @test tableau_layout(t) === layout && tableau_eltype(t) === Bits
        @test bt_same(t, u) && st_padding_zero(t)
        reset!(t, :ghz)
        @test bt_same(t, TT(2, n; state = :ghz, eltype = UInt8, layout))
        basis = [(:X, :Y, :Z)[mod1(q, 3)] for q in 1:n]
        reset!(t; state = :product, basis)
        @test bt_same(t, TT(2, n; state = :product, basis, eltype = UInt8, layout))
        @test st_padding_zero(t)

        name = layout === :qudit ? "qudit-major" : "generator-major"
        @test occursin("    Storage:          $name, Bits\n", sprint(show, t))
    end
end

@testset "Bits admissibility" begin
    for TT in bt_TYPES
        @test_throws ArgumentError TT(3, 4; eltype = Bits)
        @test_throws ArgumentError TT(3, [0 1; 1 0; 0 0; 0 0; 0 0]; eltype = Bits)
        bits_tab = TT(2, 4; eltype = Bits)
        u8_tab = TT(2, 4; eltype = UInt8)
        @test_throws ArgumentError copy!(bits_tab, u8_tab)
        @test_throws ArgumentError copy!(u8_tab, bits_tab)
        err = try copy!(bits_tab, u8_tab); nothing catch e; e end
        @test err isa ArgumentError && occursin("Bits", err.msg) && occursin("UInt8", err.msg)
    end
    # The admissible list names `Bits` at d = 2 only.
    bt_msg(d) = try bt_QC._check_eltype(Float64, d); "" catch e; e.msg end
    @test occursin("use one of UInt8, UInt16, $(nameof(Int)), Bits.", bt_msg(2))
    @test occursin("use one of UInt8, UInt16, $(nameof(Int)).", bt_msg(3))
    @test bt_QC._default_eltype(2) === bt_QC._default_qubit_eltype() === UInt8
    @test tableau_eltype(StabilizerTableau(2, 3)) === UInt8
    @test tableau_eltype(DestabilizerTableau(257, 3)) === UInt16
    @test tableau_eltype(StabilizerTableau(3, 3; eltype = Int)) === Int
end

# The no-keyword constructors at a literal d = 2, as a caller's compiled code
# sees them.
bt_default_stab(n) = StabilizerTableau(2, n)
bt_default_destab(n) = DestabilizerTableau(2, n)

@testset "default qubit construction is inferred" begin
    # Inference, not a particular element type: it holds for any default.
    for f in (bt_default_stab, bt_default_destab)
        t = @inferred f(5)
        @test isconcretetype(typeof(t))
        @test tableau_eltype(t) === bt_QC._default_qubit_eltype()
    end
end

# The mod-4 add of `δ` at the indices `sel` of a two-plane phase, written on
# whole words: the addend is the plane pair `(l, h)` of `δ`, all ones or zero
# on the selected bits of each word, through the carry formula.
function bt_plane_add!(v, sel, δ)
    P = v.planes
    for w in axes(P, 1)
        m = UInt64(0)
        for j in sel
            (j - 1) >> 6 + 1 == w && (m |= UInt64(1) << ((j - 1) & 63))
        end
        l = (δ & 1 == 1 ? typemax(UInt64) : UInt64(0)) & m
        h = (δ >> 1 == 1 ? typemax(UInt64) : UInt64(0)) & m
        p0 = P[w, 1]
        P[w, 1] = p0 ⊻ l
        P[w, 2] = P[w, 2] ⊻ h ⊻ (p0 & l)
    end
    return v
end

# Whether the bits past `len` in the last word of every plane are zero.
function bt_tail_zero(v)
    r = v.len & 63
    (r == 0 || isempty(v.planes)) && return true
    return all(v.planes[end, b] >> r == 0 for b in axes(v.planes, 2))
end

@testset "plane arithmetic" begin
    len = 130
    sel = (1, 64, 65, 130)
    rng = Xoshiro(5)
    for p in 0:3, δ in 0:3
        v = bt_QC._BitPlanes(zeros(UInt64, cld(len, 64), 2), len)
        base = rand(rng, 0:3, len)
        for j in 1:len
            v[j] = j in sel ? p : base[j]
        end
        bt_plane_add!(v, sel, δ)
        @test all(v[j] == mod(p + δ, 4) for j in sel)
        @test all(v[j] == base[j] for j in 1:len if !(j in sel))
        @test bt_tail_zero(v)
    end
    for k in (1, 2), len in (1, 63, 64, 65, 130)
        v = bt_QC._BitPlanes(zeros(UInt64, cld(len, 64), k), len)
        @test size(v) == (len,) && length(v) == len && eltype(v) === Int
        for j in 1:len, x in 0:(2^k - 1)
            v[j] = x
            @test v[j] == x
        end
        vals = rand(rng, 0:(2^k - 1), len)
        for j in 1:len
            v[j] = vals[j]
        end
        @test collect(v) == vals && bt_tail_zero(v)
        v[len] = 2^k + 1                      # stored mod 2^k
        @test v[len] == 1
        c = copy(v)
        @test c == v && c.planes !== v.planes
        fill!(v, 2^k - 1)
        @test all(==(2^k - 1), v) && bt_tail_zero(v)
        fill!(v, 0)
        @test all(iszero, v.planes)
    end
    @test length(bt_QC._BitPlanes(zeros(UInt64, 0, 2), 0)) == 0
end

@testset "Bits leaves" begin
    for (L, Lm) in ((bt_QC.GeneratorMajor, :generator), (bt_QC.QuditMajor, :qudit)), n in (65, 130)
        rng = Xoshiro(n + (Lm === :qudit ? 1 : 0))
        x, z = bt_QC._stab_bit_blocks(n)
        nw = cld(n, 64)
        @test size(x) == size(z) == (nw, 64nw)
        b = bt_QC.BitBlock{L}(x, z, n)
        @test bt_QC._nq(b) == n && bt_QC._ncap(b) == n
        X = rand(rng, 0:1, n, n)
        Z = rand(rng, 0:1, n, n)
        for j in 1:n, q in 1:n
            bt_QC._xset!(b, X[q, j], q, j)
            bt_QC._zset!(b, Z[q, j], q, j)
        end
        @test all(bt_QC._xget(b, q, j) == X[q, j] && bt_QC._zget(b, q, j) == Z[q, j]
                  for q in 1:n, j in 1:n)
        # Clearing a set bit leaves its neighbours alone.
        q1 = findfirst(==(1), X[:, 2])
        bt_QC._xset!(b, 0, q1, 2)
        @test bt_QC._xget(b, q1, 2) == 0
        bt_QC._xset!(b, 1, q1, 2)
        @test all(bt_QC._xget(b, q, j) == X[q, j] for q in 1:n, j in 1:n)

        for _ in 1:20
            i, j = rand(rng, 1:n, 2)
            @test bt_QC.dot_xz_col(b, i, j, 2) == mod(sum(X[:, i] .* Z[:, j]), 2)
            @test bt_QC.dot_xz_col(b, j, 2) == mod(sum(X[:, j] .* Z[:, j]), 2)
            ws = UInt8.(rand(rng, 0:1, 2n))
            zacc = rand(rng, 0:1, n)
            @test bt_QC.dot_xz_ws_vs_col(ws, b, j, 2) == mod(sum(Int.(ws[1:n]) .* Z[:, j]), 2)
            @test bt_QC.dot_xz_col_vs_zacc(b, j, zacc, 2) == mod(sum(X[:, j] .* zacc), 2)
        end

        # Raw words.
        rawbit(A, r, c) = Int((A[(r - 1) >> 6 + 1, c] >> ((r - 1) & 63)) & 1)
        if L === bt_QC.GeneratorMajor
            @test all(rawbit(b.x, q, j) == X[q, j] && rawbit(b.z, q, j) == Z[q, j]
                      for q in 1:n, j in 1:n)
        else
            @test all(rawbit(b.x, j, q) == X[q, j] && rawbit(b.z, j, q) == Z[q, j]
                      for q in 1:n, j in 1:n)
        end
        # Nothing was written outside the n × n bit square.
        @test all(rawbit(x, r, c) == 0 for r in 1:64nw, c in 1:64nw if r > n || c > n)
        @test all(iszero, view(x, :, n+1:64nw)) && all(iszero, view(z, :, n+1:64nw))
    end
end

end
