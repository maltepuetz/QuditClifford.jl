using QuditClifford, Test, Random
const bt_QC = QuditClifford

# The `st_` helpers live in test/storage_helpers.jl.
isdefined(Main, :st_padding_zero) || include(joinpath(@__DIR__, "storage_helpers.jl"))

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
    # The packed loader reduces entries outside 0:1 mod 2 and phases mod 4.
    for TT in bt_TYPES, layout in (:generator, :qudit)
        n = 65
        S = stabilizer_matrix(TT(2, n; state = :ghz, eltype = UInt8))
        W = copy(S)
        W[1:2n, :] .+= 2 .* rand(Xoshiro(n), -3:3, 2n, n)
        W[2n + 1, :] .+= 4 .* rand(Xoshiro(n + 1), -3:3, n)
        t = TT(2, n; state = :product, eltype = Bits, layout)
        bt_QC._load_stabilizer_matrix!(t, W)
        @test stabilizer_matrix(t) == S && st_padding_zero(t)
    end
end

# The bytes a packed constructor call allocates, after a warm-up call.
bt_construction_allocs(f) = (f(); @allocated f())

@testset "packed construction allocates about the final size" begin
    # No unpacked tableau is built on the way: a preset allocates the tableau
    # itself, a raw matrix that plus O(n²/64) words of elimination scratch.
    n = 1024
    for TT in bt_TYPES, layout in (:generator, :qudit)
        unpacked = Base.summarysize(TT(2, n; eltype = UInt8, layout))
        for state in (:product, :ghz)
            size = Base.summarysize(TT(2, n; state, eltype = Bits, layout))
            a = bt_construction_allocs(() -> TT(2, n; state, eltype = Bits, layout))
            @test a <= 1.25 * size
            @test a < unpacked ÷ 2
        end
        S = stabilizer_matrix(TT(2, n; state = :ghz, eltype = UInt8))
        size = Base.summarysize(TT(2, S; eltype = Bits, layout))
        a = bt_construction_allocs(() -> TT(2, S; eltype = Bits, layout))
        @test a <= 2 * size
        @test a < unpacked ÷ 2
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

# `set_operator!` with a plain exponent vector, which nothing in `src` calls on
# a packed tableau (`measure!` and `dephase!` wrap a vector in a
# `GeneralPauli` first), against the `UInt8` method.
@testset "Bits set_operator! from a vector" begin
    for TT in bt_TYPES, layout in (:generator, :qudit), n in (1, 3, 64, 65),
        storephase in (true, false)

        t = TT(2, n; state = :ghz, storephase, eltype = Bits, layout)
        u = TT(2, n; state = :ghz, storephase, eltype = UInt8, layout)
        L = bt_QC._layout_marker(t)
        rng = Xoshiro(100n + storephase)
        for i in unique((1, n))
            # Entries outside 0:1 and phases outside 0:3 are reduced.
            op = rand(rng, -5:5, 2n + 1)
            @test bt_QC.set_operator!(t, L, i, op) === nothing
            bt_QC.set_operator!(u, L, i, op)
            @test stabilizer_matrix(t) == stabilizer_matrix(u) && st_padding_zero(t)
            @test stabilizer_matrix(t)[1:2n, i] == mod.(op[1:2n], 2)
            storephase && @test stabilizer_matrix(t)[2n + 1, i] == mod(op[2n + 1], 4)
            # Without a phase entry the stored phase is cleared.
            op2 = view(rand(rng, 0:1, 2n + 1), 1:2n)
            bt_QC.set_operator!(t, L, i, op2)
            bt_QC.set_operator!(u, L, i, op2)
            @test stabilizer_matrix(t) == stabilizer_matrix(u) && st_padding_zero(t)
            storephase && @test stabilizer_matrix(t)[2n + 1, i] == 0
        end
        before = stabilizer_matrix(t)
        @test_throws ArgumentError bt_QC.set_operator!(t, L, 1, zeros(Int, 2n - 1))
        @test_throws ArgumentError bt_QC.set_operator!(t, L, 1, zeros(Int, 2n + 2))
        @test stabilizer_matrix(t) == before
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
    # test/runtests.jl checks that the shipped default is `UInt8`, before the
    # suite may switch it to `Bits`.
    @test bt_QC._default_eltype(2) === bt_QC._default_qubit_eltype()
    @test tableau_eltype(StabilizerTableau(2, 3)) === bt_QC._default_qubit_eltype()
    @test tableau_eltype(StabilizerTableau(2, 3; eltype = UInt8)) === UInt8
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

# Entry (q, j) of the X block `x` (word columns, qudit-major) and of the Z block
# `z`, read from the raw words: bit `(j-1) & 63` of word `(j-1) >> 6 + 1` of column `q`.
bt_raw_entry(A, q, j) = Int((A[(j - 1) >> 6 + 1, q] >> ((j - 1) & 63)) & UInt64(1))

function bt_blocks(tab)
    tab isa DestabilizerTableau ?
        (tab.stab_x, tab.stab_z, tab.destab_x, tab.destab_z) : (tab.stab_x, tab.stab_z)
end

# Dense random generators in every block of the generator-major `tab`, all
# `n` of them live (`random_state!` is too slow to fill thousands of qudits):
# the bits of the `n × n` square are random and the padding stays zero.
function bt_random_blocks!(rng, tab)
    n = tab.n
    for A in bt_blocks(tab), c in 1:n, w in axes(A, 1)
        A[w, c] = rand(rng, UInt64) & bt_QC._tail_mask(min(n - 64 * (w - 1), 64))
    end
    tab.m = n
    return tab
end

@testset "Bits switch round trip" begin
    for TT in bt_TYPES, n in (1, 2, 63, 64, 65, 127, 128, 129, 1984, 1985, 2047, 2048, 2049)
        rng = Xoshiro(20261121 + n)
        tab = TT(2, n; eltype = Bits, layout = :generator)
        n <= 129 ? random_state!(rng, tab; m = n ÷ 2) : bt_random_blocks!(rng, tab)
        @test st_padding_zero(tab)
        S = stabilizer_matrix(tab)
        D = TT === DestabilizerTableau ? destabilizer_matrix(tab) : nothing
        ph = copy(collect(tab.phase))
        cache = TT === DestabilizerTableau ? collect(tab.xdotz_cache) : nothing
        m, canon = tab.m, tab.iscanonical
        P = map(A -> copy(parent(A)), bt_blocks(tab))
        planes = TT === DestabilizerTableau ? copy(tab.xdotz_cache.planes) : nothing
        qudit_major!(tab)
        @test tableau_layout(tab) === :qudit
        @test stabilizer_matrix(tab) == S && collect(tab.phase) == ph
        @test tab.m == m && tab.iscanonical == canon
        if TT === DestabilizerTableau
            @test destabilizer_matrix(tab) == D && collect(tab.xdotz_cache) == cache
            @test tab.xdotz_cache.planes == planes
        end
        raw = bt_blocks(tab)
        pairs = n <= 130 ? vec([(q, j) for q in 1:n, j in 1:n]) :
                [(rand(rng, 1:n), rand(rng, 1:n)) for _ in 1:20000]
        n > 130 && push!(pairs, (n, n), (1, n), (n, 1), (64, 65), (65, 64))
        @test all(bt_raw_entry(raw[1], q, j) == S[q, j] && bt_raw_entry(raw[2], q, j) == S[n+q, j]
                  for (q, j) in pairs)
        if TT === DestabilizerTableau
            @test all(bt_raw_entry(raw[3], q, j) == D[q, j] && bt_raw_entry(raw[4], q, j) == D[n+q, j]
                      for (q, j) in pairs)
        end
        @test st_padding_zero(tab)
        generator_major!(tab)
        @test tableau_layout(tab) === :generator
        @test all(parent(A) == p for (A, p) in zip(bt_blocks(tab), P))
        @test st_padding_zero(tab)
        @test stabilizer_matrix(tab) == S
        qudit_major!(tab); generator_major!(tab)
        @test (@allocated qudit_major!(tab)) == 0
        @test (@allocated generator_major!(tab)) == 0
    end
    # The switch needs two 64-word tiles of scratch, which the workspace has
    # for every n, and refuses a shorter buffer.
    tab = StabilizerTableau(2, 3; eltype = Bits)
    @test length(tab.word_workspace) >= 128
    S = stabilizer_matrix(tab)
    @test_throws ArgumentError bt_QC._transpose_bit_block!(tab.stab_x, zeros(UInt64, 127))
    @test stabilizer_matrix(tab) == S
end

@testset "_t64! is a transpose" begin
    rng = Xoshiro(64)
    ref(a) = UInt64[sum(((a[b+1] >> (i - 1)) & UInt64(1)) << (b) for b in 0:63; init = UInt64(0))
                    for i in 1:64]
    for _ in 1:100
        a = rand(rng, UInt64, 64)
        b = copy(a)
        bt_QC._t64!(b, 0)
        @test b == ref(a)
        bt_QC._t64!(b, 0)
        @test b == a
    end
    a = rand(rng, UInt64, 192); b = copy(a)
    bt_QC._t64!(b, 64)
    @test b[1:64] == a[1:64] && b[129:192] == a[129:192] && b[65:128] == ref(a[65:128])
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


# A random qubit Pauli of every operator type on `n` qudits, Hermitian:
# `SinglePauli`, `DoublePauli`, `TriplePauli`, `NPauli`, `GeneralPauli`, or a
# plain exponent vector with a phase entry.
function bt_rand_pauli(rng, n)
    kind = rand(rng, 1:6)
    k = kind <= 3 ? kind : kind == 4 ? rand(rng, 1:min(n, 5)) : n
    k = min(k, n)
    qs = randperm(rng, n)[1:k]
    xs = rand(rng, 0:1, k)
    zs = rand(rng, 0:1, k)
    ph = mod(sum(xs .* zs), 2) + 2 * rand(rng, 0:1)
    if kind == 1 || k == 1
        return SinglePauli(qs[1], xs[1], zs[1], ph)
    elseif kind == 2 || k == 2
        return DoublePauli(qs[1], xs[1], zs[1], qs[2], xs[2], zs[2], ph)
    elseif kind == 3 || (kind == 4 && k == 3)
        return TriplePauli(qs[1], xs[1], zs[1], qs[2], xs[2], zs[2], qs[3], xs[3], zs[3], ph)
    elseif kind == 4
        return NPauli(Tuple(qs), Tuple(xs), Tuple(zs), ph)
    end
    v = zeros(Int, 2n)
    v[qs] .= xs
    v[n .+ qs] .= zs
    return kind == 5 ? GeneralPauli(v, ph) : vcat(v, ph)
end

# The next operation for the packed-against-`UInt8` comparison, drawn from
# `rng` and the state of `ref`.
function bt_rand_op(rng, ref)
    n, m = ref.n, ref.m
    c = rand(rng, 1:20)
    if c <= 4
        q = rand(rng, 1:n)
        g = rand(rng, (Fourier(q), Phase(q), PauliGate(q, rand(rng, 0:1), rand(rng, 0:1))))
        if n >= 2 && rand(rng, Bool)
            q1, q2 = randperm(rng, n)[1:2]
            g = rand(rng, (SUM(q1, q2), CPhase(q1, q2), SWAP(q1, q2)))
        end
        return (t, r) -> apply!(t, g)
    elseif c <= 6
        # Tuple-backed up to three qudits, dense for four.
        C = random_clifford(rng, 2, randperm(rng, n)[1:min(n, rand(rng, 1:4))])
        return (t, r) -> apply!(t, C)
    elseif c <= 10
        op = (m > 0 && rand(rng, 1:4) == 1) ? st_generator_pauli(ref, rand(rng, 1:m)) :
             bt_rand_pauli(rng, n)
        return rand(rng, Bool) ? ((t, r) -> measure!(r, t, op)) :
                                 ((t, r) -> measure!(t, op; outcome = 1))
    elseif c == 11
        op = (m > 0 && rand(rng, Bool)) ? st_generator_pauli(ref, rand(rng, 1:m)) :
             bt_rand_pauli(rng, n)
        op isa AbstractVector && (op = GeneralPauli(op[1:2n], op[end]))
        return rand(rng, Bool) ? ((t, r) -> expect!(t, op)) : ((t, r) -> expect_int!(t, op))
    elseif c == 12
        op = bt_rand_pauli(rng, n)
        return (t, r) -> dephase!(t, op)
    elseif c == 13
        sites = randperm(rng, n)[1:rand(rng, 1:min(n, 3))]
        return (t, r) -> traceout!(t, sites)
    elseif c == 14
        sites = randperm(rng, n)[1:rand(rng, 1:min(n, 3))]
        basis = rand(rng, (:Z, :X, :Y))
        return (t, r) -> reset!(t, sites; basis)
    elseif c == 15
        return (t, r) -> canonicalize!(t)
    elseif c == 16
        mm = rand(rng, (n, rand(rng, 0:n)))
        return (t, r) -> random_state!(r, t; m = mm)
    elseif c == 17
        A, B, C = st_rand_parts(rng, n)
        f = rand(rng, (t -> entanglement_entropy(t, A), t -> entanglement_profile(t),
                       t -> mutual_information(t, A, B),
                       t -> conditional_mutual_information(t, A, B, C),
                       t -> tripartite_information(t, A, B, C),
                       t -> entanglement_negativity(t, A, B), bt_QC.is_independent))
        return (t, r) -> f(t)
    elseif c == 18
        return (t, r) -> is_pure(t; verify = true)
    end
    q = rand(rng, 1:n)
    return (t, r) -> apply!(t, Fourier(q))
end

@testset "same results, Bits vs UInt8" begin
    seed = 0
    for TT in bt_TYPES, layout in (:generator, :qudit), storephase in (true, false),
        (n, nops) in ((130, 400), (1, 60), (63, 100))

        seed += 1
        t = TT(2, n; storephase, eltype = Bits, layout)
        u = TT(2, n; storephase, eltype = UInt8, layout)
        oprng = Xoshiro(30_000 + seed)
        rt, ru = Xoshiro(seed), Xoshiro(seed)
        bad = 0
        for i in 1:nops
            op = bt_rand_op(oprng, u)
            vt = st_call(op, t, rt)
            vu = st_call(op, u, ru)
            ok = isequal(vt, vu) && bt_same(t, u) && st_padding_zero(t) &&
                 tableau_layout(t) === layout && rt == ru
            if !ok
                bad += 1
                @error "Bits differs from UInt8" TT layout storephase n i vt vu
                break
            end
        end
        @test bad == 0
    end
end

# A generator-major `Int` block and a packed block in layout `L` holding the
# same random `n × n` entries 0 or 1.
function bt_block_pair(rng, L, n)
    X = rand(rng, 0:1, n, n)
    Z = rand(rng, 0:1, n, n)
    ref = bt_QC.XZBlock{bt_QC.GeneratorMajor}(copy(X), copy(Z))
    b = bt_QC.BitBlock{L}(bt_QC._stab_bit_blocks(n)..., n)
    for j in 1:n, q in 1:n
        bt_QC._xset!(b, X[q, j], q, j)
        bt_QC._zset!(b, Z[q, j], q, j)
    end
    return ref, b
end

bt_entries(b, n) = ([bt_QC._xget(b, q, j) for q in 1:n, j in 1:n],
                    [bt_QC._zget(b, q, j) for q in 1:n, j in 1:n])

@testset "Bits composites" begin
    for L in (bt_QC.GeneratorMajor, bt_QC.QuditMajor), n in (1, 63, 65, 130), Ev in (UInt8, Int)
        rng = Xoshiro(n + 7 * (L === bt_QC.QuditMajor))
        sites = randperm(rng, n)[1:min(n, 5)]
        ns = length(sites)
        js = rand(rng, 1:n, 3)
        j, k = js[1], n == 1 ? 1 : rand(rng, setdiff(1:n, js[1]))
        for a in (1, 0)
            ops = (
                (b, v) -> bt_QC._gen_addmul!(b, j, v, a, 2),
                (b, v) -> bt_QC._gen_submul!(b, j, v, a, 2),
                (b, v) -> bt_QC._gen_mulcopy!(b, j, v, a, 2),
                (b, v) -> bt_QC._vec_addmul!(v, b, j, a, 2),
                (b, v) -> bt_QC._vec_submul!(v, b, j, a, 2),
                (b, v) -> bt_QC._gen_addmul_sites!(b, j, v, sites, ns, a, 2),
                (b, v) -> bt_QC._gen_submul_sites!(b, j, v, sites, ns - 1, a, 2),
                (b, v) -> j == k || bt_QC._gen_addmul!(b, j, b, k, a, 2),
                (b, v) -> j == k || bt_QC._gen_submul!(b, j, b, k, a, 2),
                (b, v) -> j == k || bt_QC._gen_addmul_sites!(b, j, b, k, sites, ns, a, 2),
                (b, v) -> j == k || bt_QC._gen_submul_sites!(b, k, b, j, sites, ns, a, 2),
                (b, v) -> bt_QC._gen_scale!(b, j, a, 2),
                (b, v) -> bt_QC._copy_gen!(v, b, j),
                (b, v) -> bt_QC._load_gen!(b, j, v),
                (b, v) -> bt_QC._swap_gens!(b, j, k),
                (b, v) -> bt_QC._move_gen!(b, j, k),
                (b, v) -> bt_QC._zero_gen!(b, j),
                (b, v) -> bt_QC._rset!(b, 1 - bt_QC._rget(b, n + sites[1], j), n + sites[1], j),
            )
            for op in ops
                ref, b = bt_block_pair(rng, L, n)
                v0 = Ev.(rand(rng, 0:1, 2n))
                vref, v = Int.(v0), copy(v0)
                op(ref, vref)
                op(b, v)
                @test bt_entries(b, n) == (ref.x, ref.z) && Int.(v) == vref
                # Nothing is written outside the n × n bit square.
                nw = cld(n, 64)
                @test all(iszero, view(b.x, :, n+1:64nw)) && all(iszero, view(b.z, :, n+1:64nw))
                @test all(A[w, c] >> (n - 64 * (w - 1)) == 0
                          for A in (b.x, b.z), c in 1:64nw, w in 1:nw if n - 64 * (w - 1) < 64)
            end
        end
    end
end

# `k` distinct targets on `n` qudits, drawn from the word edges 1, 64, 65, `n`
# half the time, otherwise uniformly.
function bt_targets(rng, n, k)
    edges = filter(<=(n), unique([1, 64, 65, n]))
    pool = rand(rng, Bool) ? vcat(shuffle(rng, edges), randperm(rng, n)) : randperm(rng, n)
    return unique(pool)[1:k]
end

# The next gate of the kernel comparison: a named gate, a random Clifford on
# 1-3 qudits or a dense one on 4, or a named gate prepared off the fast tier.
# Returns a function of the tableau.
function bt_rand_gate(rng, n)
    c = rand(rng, 1:10)
    if c <= 3
        q = bt_targets(rng, n, 1)[1]
        g = rand(rng, (Fourier(q), Phase(q), PauliGate(q, rand(rng, 0:1), rand(rng, 0:1))))
        return t -> apply!(t, g)
    elseif c <= 6
        q1, q2 = bt_targets(rng, n, 2)
        g = rand(rng, (SUM(q1, q2), CPhase(q1, q2), SWAP(q1, q2)))
        return t -> apply!(t, g)
    elseif c <= 8
        C = random_clifford(rng, 2, bt_targets(rng, n, rand(rng, 1:min(n, 3))))
        return t -> apply!(t, C)
    elseif c == 9
        C = random_clifford(rng, 2, bt_targets(rng, n, min(n, 4)))
        return t -> apply!(t, C)
    end
    q1, q2 = bt_targets(rng, n, 2)
    g = rand(rng, (Fourier(q1), Phase(q1), PauliGate(q1, rand(rng, 0:1), rand(rng, 0:1)),
                   SUM(q1, q2), CPhase(q1, q2), SWAP(q1, q2)))
    return t -> bt_QC._apply_prepared!(t, bt_QC._prepare(g, 2, t.inversemod, t.storephase;
                                                         force_safe = true))
end

@testset "bit gate kernel" begin
    seed = 0
    for TT in bt_TYPES, layout in (:generator, :qudit), storephase in (true, false),
        n in (5, 64, 65, 130, 300)

        seed += 1
        t = TT(2, n; storephase, eltype = Bits, layout)
        u = TT(2, n; storephase, eltype = UInt8, layout)
        mm = isodd(seed) ? n : n - 3
        random_state!(Xoshiro(seed), t; m = mm)
        random_state!(Xoshiro(seed), u; m = mm)
        rng = Xoshiro(40_000 + seed)
        bad = 0
        for i in 1:300
            f = bt_rand_gate(rng, n)
            f(t)
            f(u)
            ok = bt_same(t, u) && t.phase == u.phase && st_padding_zero(t)
            if !ok
                bad += 1
                @error "bit gate kernel differs from UInt8" TT layout storephase n i
                break
            end
        end
        @test bad == 0
        L = bt_QC._layout_marker(t)
        U = random_clifford(Xoshiro(seed), 2, [1, 2, 3])
        for prep in (bt_QC._prepare(Fourier(1), 2, t.inversemod, storephase),
                     bt_QC._prepare(SUM(1, 2), 2, t.inversemod, storephase),
                     bt_QC._prepare_tuple(U, storephase, Val(3)))
            m = which(bt_QC._apply_columns!, (typeof(t), typeof(prep), typeof(L)))
            @test basename(String(m.file)) == "apply_bits.jl"
        end
    end
end

function bt_apply_allocations(t, g)
    for _ in 1:3
        apply!(t, g)
    end
    return @allocated apply!(t, g)
end

@testset "bit gate kernel allocates nothing" begin
    for TT in bt_TYPES, layout in (:generator, :qudit), storephase in (true, false)
        t = TT(2, 130; state = :ghz, storephase, eltype = Bits, layout)
        for g in (Fourier(64), SUM(1, 130), random_clifford(Xoshiro(3), 2, [1, 65, 130]),
                  CliffordOperator(CPhase(64, 65), 2))
            bt_apply_allocations(t, g)
            @test bt_apply_allocations(t, g) == 0
        end
    end
end

# A random Hermitian qubit Pauli for the outcome-stream test: one of the sparse
# types, or a dense `GeneralPauli` with every exponent random.
function bt_stream_pauli(rng, n)
    if rand(rng, 1:5) <= 4
        op = bt_rand_pauli(rng, n)
        while op isa AbstractVector || op isa GeneralPauli
            op = bt_rand_pauli(rng, n)
        end
        return op
    end
    v = rand(rng, 0:1, 2n)
    return GeneralPauli(v, mod(sum(v[1:n] .* v[n+1:2n]), 2) + 2 * rand(rng, 0:1))
end

# The reference is the per-generator `_measure_generic!` on `UInt8` storage in
# the same layout, so qudit-major the packed kernel is not compared with
# another kernel.
@testset "same rng stream, Bits vs UInt8" begin
    for TT in bt_TYPES, layout in (:generator, :qudit)
        n = 70
        t = TT(2, n; state = :mixed, eltype = Bits, layout)
        u = TT(2, n; state = :mixed, eltype = UInt8, layout)
        rt, ru = Xoshiro(7), Xoshiro(7)
        oprng = Xoshiro(50_000 + n)
        outs_t = Int[]
        outs_u = Int[]
        ok = true
        for i in 1:300
            if rand(oprng) < 0.3
                g = bt_rand_gate(oprng, n)
                g(t)
                g(u)
            end
            op = bt_stream_pauli(oprng, n)
            push!(outs_t, measure!(rt, t, op))
            push!(outs_u, bt_QC._measure_generic!(ru, u, bt_QC._layout_marker(u), op, nothing, 0))
            ok &= outs_t[end] == outs_u[end] && stabilizer_matrix(t) == stabilizer_matrix(u) &&
                  st_padding_zero(t)
            ok || (@error "Bits outcome stream differs" TT layout i op; break)
        end
        @test ok
        @test outs_t == outs_u
        @test bt_same(t, u)
        @test rand(rt) == rand(ru)
    end
end

@testset "bit measurement kernels against the generic functions" begin
    # measure! and dephase! on packed storage (`t`) against `_measure_generic!`
    # and `_dephase_generic!` on `UInt8` storage in the same layout (`u`), and
    # against `_dephase_generic!` on packed storage (`g`), after every call.
    seed = 0
    for TT in bt_TYPES, layout in (:generator, :qudit), storephase in (true, false),
        n in (5, 64, 65, 130)

        seed += 1
        t = TT(2, n; storephase, eltype = Bits, layout)
        g = TT(2, n; storephase, eltype = Bits, layout)
        u = TT(2, n; storephase, eltype = UInt8, layout)
        mm = isodd(seed) ? n : n - 3
        foreach(x -> random_state!(Xoshiro(seed), x; m = mm), (t, g, u))
        L = bt_QC._layout_marker(t)
        rt, rg, ru = Xoshiro(seed), Xoshiro(seed), Xoshiro(seed)
        rng = Xoshiro(80_000 + seed)
        bad = 0
        for i in 1:150
            c = rand(rng, 1:10)
            vt = vg = vu = nothing
            if c <= 2
                f = bt_rand_gate(rng, n)
                foreach(f, (t, g, u))
            elseif c <= 7
                op = bt_stream_pauli(rng, n)
                vt = measure!(rt, t, op)
                vg = measure!(rg, g, op)
                vu = bt_QC._measure_generic!(ru, u, L, op, nothing, 0)
            else
                op = bt_stream_pauli(rng, n)
                dephase!(t, op)
                bt_QC._dephase_generic!(g, L, op)
                bt_QC._dephase_generic!(u, L, op)
            end
            ok = vt == vg == vu && bt_same(t, g) && bt_same(t, u) && rt == rg == ru &&
                 st_padding_zero(t) && st_padding_zero(g)
            if !ok
                bad += 1
                @error "packed kernel differs from the generic function" TT layout storephase n i
                break
            end
        end
        @test bad == 0
    end

    # The packed dual update against `_orthogonalize_duals_generic!` on the
    # same packed tableau, on its site-restricted and its dense path: generator
    # `col` is overwritten with an operand, as measure! does, and the other
    # duals are made orthogonal to it.
    for layout in (:generator, :qudit), n in (5, 64, 65, 130)
        t = DestabilizerTableau(2, n; eltype = Bits, layout)
        random_state!(Xoshiro(n), t; m = n - 2)
        L = bt_QC._layout_marker(t)
        rng = Xoshiro(90_000 + n)
        changed = 0
        for trial in 1:12
            col = rand(rng, 1:t.m)
            op = bt_stream_pauli(rng, n)
            kernel, sparse, dense = copy(t), copy(t), copy(t)
            for (x, f!, sp) in ((kernel, bt_QC._orthogonalize_duals!, true),
                                (sparse, bt_QC._orthogonalize_duals_generic!, true),
                                (dense, bt_QC._orthogonalize_duals_generic!, false))
                bt_QC.set_operator!(x, L, col, op)
                nsupp = bt_QC._column_support!(x.support_workspace, bt_QC._stab(x, L), col)
                nsites = bt_QC._column_support!(x.zacc_workspace, bt_QC._destab(x, L), col)
                f!(x, L, col, x.m, x.support_workspace, nsupp, x.zacc_workspace, nsites, sp)
            end
            @test destabilizer_matrix(kernel) == destabilizer_matrix(sparse) ==
                  destabilizer_matrix(dense)
            @test stabilizer_matrix(kernel) == stabilizer_matrix(sparse)
            @test st_padding_zero(kernel) && st_padding_zero(sparse) && st_padding_zero(dense)
            changed += destabilizer_matrix(kernel) != destabilizer_matrix(t)
        end
        @test changed > 0
    end
end

# A canonicalization fixture in storage `E`, generator-major: a random pure
# state, a brickwork circuit with Z measurements at rate 0.15 on a pure state
# (`:monitored`), or random measurements on the maximally mixed state
# (`:measured`, so m < n). A final gate layer leaves it out of canonical form.
function bt_canon_fixture(TT, n, kind, storephase, E)
    rng = Xoshiro(1000n + findfirst(==(kind), (:random, :monitored, :measured)))
    tab = TT(2, n; state = kind === :measured ? :mixed : :Z, storephase, eltype = E,
             layout = :generator)
    if kind === :random
        random_state!(rng, tab)
    elseif kind === :monitored
        for l in 1:8
            for q in (isodd(l) ? 1 : 2):2:(n - 1)
                apply!(tab, random_clifford(rng, 2, [q, q + 1]))
            end
            for q in 1:n
                rand(rng) < 0.15 && measure!(rng, tab, SinglePauli(q, 0, 1))
            end
        end
    else
        for _ in 1:(n ÷ 2)
            q = rand(rng, 1:n)
            x = rand(rng, 0:1)
            measure!(rng, tab, SinglePauli(q, x, 1, x))
            n >= 2 && apply!(tab, random_clifford(rng, 2, randperm(rng, n)[1:2]))
        end
    end
    for q in 1:2:(n - 1)
        apply!(tab, random_clifford(rng, 2, [q, q + 1]))
    end
    return tab
end

# `bt_same` plus the phases, the x·z cache of either type and the padding.
bt_canon_same(t, u) = bt_same(t, u) && t.iscanonical && collect(t.xdotz_cache) == collect(u.xdotz_cache) &&
                      (!t.storephase || collect(t.phase) == collect(u.phase)) && st_padding_zero(t)

function bt_canon_allocations(f!, t)
    for _ in 1:2
        f!(t)
    end
    return @allocated f!(t)
end

@testset "bit canonicalize kernels" begin
    QM, GM = bt_QC.QuditMajor(), bt_QC.GeneratorMajor()
    gm_kernel!(t) = bt_QC._canonicalize_gm_bits!(t, GM)
    for TT in bt_TYPES, storephase in (true, false), n in (5, 64, 65, 128, 130, 300),
        kind in (:random, :monitored, :measured)

        u = bt_canon_fixture(TT, n, kind, storephase, UInt8)
        canonicalize!(u)
        ui = bt_canon_fixture(TT, n, kind, storephase, Int)
        canonicalize!(ui)

        # qudit-major: switched to generator-major, the kernel, switched back
        q = qudit_major!(bt_canon_fixture(TT, n, kind, storephase, Bits))
        canonicalize!(q)
        @test tableau_layout(q) === :qudit
        @test bt_canon_same(q, u)
        @test bt_canon_same(q, ui)

        g = bt_canon_fixture(TT, n, kind, storephase, Bits)
        gm_kernel!(g)
        @test bt_canon_same(g, u)
        @test bt_canon_same(g, ui)
        @test stabilizer_matrix(q) == stabilizer_matrix(g) && collect(q.phase) == collect(g.phase)

        if n == 130 && kind === :monitored
            @test bt_canon_allocations(canonicalize!, q) == 0
            @test bt_canon_allocations(gm_kernel!, g) == 0
        end
    end
    for TT in bt_TYPES, n in (1, 65)
        # m = 0: nothing to reduce.
        u = TT(2, n; state = :mixed, eltype = UInt8, layout = :generator)
        canonicalize!(u)
        q = TT(2, n; state = :mixed, eltype = Bits, layout = :qudit)
        canonicalize!(q)
        g = TT(2, n; state = :mixed, eltype = Bits, layout = :generator)
        gm_kernel!(g)
        @test bt_canon_same(q, u) && bt_canon_same(g, u)
        @test tableau_layout(q) === :qudit
        for (TL, L) in ((:qudit, QM), (:generator, GM))
            s = TT(2, n; eltype = Bits, layout = TL)
            m = which(bt_QC._canonicalize!, (typeof(s), typeof(L)))
            @test basename(String(m.file)) == "canonicalize_bits.jl"
        end
        # the qudit-major method is the switched route
        s = TT(2, n; eltype = Bits, layout = :qudit)
        @test occursin("_canonicalize_bits_switched!",
                       sprint(io -> code_lowered(bt_QC._canonicalize!, (typeof(s), typeof(QM))) .|> (c -> print(io, c))))
    end
end


# Flip the X (`which = 1`), Z (2) entry or add 2 to the phase (3) of
# generator `j` at qudit `q`.
function bt_perturb!(t, which, q, j)
    L = bt_QC._layout_marker(t)
    b = bt_QC._stab(t, L)
    if which == 1
        bt_QC._xset!(b, 1 - bt_QC._xget(b, q, j), q, j)
    elseif which == 2
        bt_QC._zset!(b, 1 - bt_QC._zget(b, q, j), q, j)
    else
        t.phase[j] = mod(t.phase[j] + 2, 4)
    end
    return t
end

bt_eq_allocations(a, b) = (a == b; @allocated a == b)

@testset "bit equality kernels" begin
    for TT in bt_TYPES, storephase in (true, false), n in (5, 64, 65, 128, 130),
        kind in (:monitored, :measured),
        layout in (:generator, :qudit)

        u = bt_canon_fixture(TT, n, kind, storephase, UInt8)
        t = bt_canon_fixture(TT, n, kind, storephase, Bits)
        layout === :qudit && qudit_major!(t)
        @test t == copy(t)
        @test bt_eq_allocations(t, copy(t)) == 0
        c = copy(t); canonicalize!(c)
        @test t == c
        m = t.m
        m == 0 && continue
        rng = Xoshiro(n + m)
        for j in unique((1, m, rand(rng, 1:m), min(m, 64), min(m, 65))), which in (1, 2, 3)
            which == 3 && !storephase && continue
            q = rand(rng, 1:n)
            t2 = bt_perturb!(copy(t), which, q, j)
            u2 = bt_perturb!(copy(u), which, q, j)
            @test (t == t2) == (u == u2)
            @test (t2 == t) == (u2 == u)
            Lt, Lu = bt_QC._layout_marker(t), bt_QC._layout_marker(u)
            @test bt_QC._active_columns_equal(bt_QC._stab(t, Lt), t.phase, bt_QC._stab(t2, Lt), t2.phase, m, storephase) ==
                  bt_QC._active_columns_equal(bt_QC._stab(u, Lu), u.phase, bt_QC._stab(u2, Lu), u2.phase, m, storephase) ==
                  false
        end
    end
    for (TL, B) in ((:generator, bt_QC._GMBits), (:qudit, bt_QC._QMBits))
        t = StabilizerTableau(2, 5; eltype = Bits, layout = TL)
        P = typeof(t.phase)
        m = which(bt_QC._active_columns_equal, (B, P, B, P, Int, Bool))
        @test basename(String(m.file)) == "bit_block.jl"
    end
end


# An operand for the span tests: a product of a random subset of the generators
# of `t`, moved out of the span by one flipped exponent a quarter of the time;
# an operand measured earlier (`pool`); or a random sparse or dense Pauli with
# X, Y and Z components. Phases are arbitrary, so many operands are not
# Hermitian.
function bt_span_pauli(rng, t, n, pool)
    c = rand(rng, 1:6)
    if c <= 2 && t.m > 0
        S = stabilizer_matrix(t)
        v = zeros(Int, 2n)
        for j in 1:t.m
            rand(rng, Bool) && (v .⊻= S[1:2n, j])
        end
        rand(rng, 1:4) == 1 && (v[rand(rng, 1:2n)] ⊻= 1)
        return GeneralPauli(v, rand(rng, 0:3))
    elseif c == 3 && !isempty(pool)
        return rand(rng, pool)
    elseif c == 4
        return GeneralPauli(rand(rng, -2:3, 2n), rand(rng, -3:6))
    elseif c == 5
        return SinglePauli(rand(rng, 1:n), rand(rng, 0:1), rand(rng, 0:1), rand(rng, 0:3))
    end
    k = rand(rng, 1:min(n, 4))
    return NPauli(Tuple(randperm(rng, n)[1:k]), Tuple(rand(rng, -2:3, k)),
                  Tuple(rand(rng, -2:3, k)), rand(rng, -3:6))
end

# The value of `f()`, or the message of the `ArgumentError` it throws.
bt_try(f) = try
    (:ok, f())
catch e
    e isa ArgumentError || rethrow()
    (:err, e.msg)
end

# `measure!` with the commuting branches taken through the generic
# `_measure_commuting!`, whatever the storage.
function bt_generic_measure!(rng, g, op, outcome, phase_policy)
    L = bt_QC._layout_marker(g)
    bt_QC._commutes_with_generators(g, L, op) ||
        return measure!(rng, g, op; outcome, phase_policy)
    kop = bt_QC._measurement_phase(g, op, outcome, phase_policy)
    return bt_QC._measure_commuting!(rng, g, L, op, kop, outcome)
end

# Bit `q` of word column `c` of a word workspace.
bt_ws_bit(ws, q, c) = Int((ws[(q - 1) >> 6 + 1, c] >> ((q - 1) & 63)) & UInt64(1))

@testset "bit span kernels" begin
    # The packed commuting branches of measure! and expect_int! against the
    # generic pipeline on a packed copy (`g`) and against `_measure_generic!`
    # and `_expect_int_generic!` on `UInt8` storage (`u`), after every call.
    seed = 0
    for TT in bt_TYPES, layout in (:generator, :qudit), storephase in (true, false)
        counts = Dict{Symbol,Int}()
        hit(k) = (counts[k] = get(counts, k, 0) + 1)
        bad = 0
        for n in (1, 5, 63, 64, 65, 130), pure in (true, false)
            seed += 1
            mm = pure ? n : rand(Xoshiro(seed), 0:n)
            t = TT(2, n; storephase, eltype = Bits, layout)
            g = TT(2, n; storephase, eltype = Bits, layout)
            u = TT(2, n; storephase, eltype = UInt8, layout)
            for x in (t, g, u)
                random_state!(Xoshiro(seed), x; m = mm)
            end
            L = bt_QC._layout_marker(t)
            rt, rg, ru = Xoshiro(seed), Xoshiro(seed), Xoshiro(seed)
            rng = Xoshiro(60_000 + seed)
            pool = Any[]
            for i in 1:(n <= 5 ? 250 : 90)
                c = rand(rng, 1:20)
                vt = vg = vu = nothing
                if c <= 3
                    U = random_clifford(rng, 2, randperm(rng, n)[1:min(n, rand(rng, 1:2))])
                    foreach(x -> apply!(x, U), (t, g, u))
                elseif c <= 5
                    foreach(canonicalize!, (t, g, u))
                elseif c == 6
                    # A fresh state that withholds up to three generators of a
                    # pure one; they commute with the rest and lie outside its
                    # span, so measuring one from the pool appends it.
                    r = StabilizerTableau(2, n; storephase, eltype = UInt8)
                    random_state!(rng, r)
                    S = stabilizer_matrix(r)
                    k = rand(rng, 1:min(n, 3))
                    Sm = hcat(S[:, 1:(n - k)], zeros(Int, size(S, 1), k))
                    t, g, u = (TT(2, Sm; m = n - k, storephase, eltype = E, layout)
                               for E in (Bits, Bits, UInt8))
                    empty!(pool)
                    for j in (n - k + 1):n
                        push!(pool, GeneralPauli(S[1:2n, j], storephase ? S[2n + 1, j] : 0))
                    end
                elseif c <= 14
                    op = bt_span_pauli(rng, t, n, pool)
                    outcome = rand(rng, 1:4) == 1 ? rand(rng, 0:1) : nothing
                    pp = rand(rng, 1:2)
                    commutes = bt_QC._commutes_with_generators(g, L, op)
                    m0 = g.m
                    vt = bt_try(() -> measure!(rt, t, op; outcome, phase_policy = pp))
                    vg = bt_try(() -> bt_generic_measure!(rg, g, op, outcome, pp))
                    vu = bt_try(() -> bt_QC._measure_generic!(ru, u, L, op, outcome, pp))
                    if !commutes
                        hit(:random)
                    elseif g.m > m0
                        hit(:append)
                        # The generic residual tells which unit vector the dual grew from.
                        g isa DestabilizerTableau &&
                            hit(any(!iszero, g.res_workspace[1:n]) ? :append_x : :append_z)
                    else
                        hit(vg[1] === :ok ? :fixed : :fixed_nonhermitian)
                        op isa GeneralPauli || hit(:fixed_sparse)
                        t isa StabilizerTableau && hit(t.iscanonical ? :fixed_canonical : :fixed_solve)
                    end
                    length(pool) < 6 ? push!(pool, op) : (pool[rand(rng, 1:6)] = op)
                else
                    op = bt_span_pauli(rng, t, n, pool)
                    vt = bt_try(() -> expect_int!(t, op))
                    vg = bt_try(() -> bt_QC._expect_int_generic!(g, L, op))
                    vu = bt_try(() -> bt_QC._expect_int_generic!(u, L, op))
                    hit(vg == (:ok, -1) ? :expect_out : :expect_in)
                    vg[2] in (1, 3) && hit(:expect_imaginary)
                end
                ok = vt == vg == vu && bt_same(t, g) && bt_same(t, u) && rt == rg == ru &&
                     st_padding_zero(t)
                if !ok
                    bad += 1
                    @error "packed span kernel differs" TT layout storephase n pure i vt vg vu
                    break
                end
            end
        end
        @test bad == 0
        for k in (:random, :append, :fixed, :fixed_sparse, :expect_in, :expect_out)
            @test (k => get(counts, k, 0) > 0) == (k => true)
        end
        if storephase
            @test get(counts, :fixed_nonhermitian, 0) > 0
            @test get(counts, :expect_imaginary, 0) > 0
        end
        if TT === DestabilizerTableau
            @test get(counts, :append_x, 0) > 0 && get(counts, :append_z, 0) > 0
        else
            @test get(counts, :fixed_canonical, 0) > 0 && get(counts, :fixed_solve, 0) > 0
        end
    end

    # The pieces against the generic ones on the same tableau: the coefficient
    # mask, the residual and the phase exponent, with the qudit-major kernel
    # forced through its sparse and its dense path.
    QM = bt_QC.QuditMajor()
    paths = Dict{Symbol,Int}()
    for TT in bt_TYPES, layout in (:generator, :qudit), storephase in (true, false),
        n in (1, 5, 63, 64, 65, 130, 200, 330), kind in (:random, :local)

        seed += 1
        rng = Xoshiro(70_000 + seed)
        t = TT(2, n; storephase, eltype = Bits, layout, state = :mixed)
        if kind === :random
            random_state!(rng, t; m = rand(rng, (n, max(n - 2, 0))))
        else
            for q in 1:2:n
                measure!(rng, t, SinglePauli(q, 0, 1))
            end
            for q in 1:(n - 1)
                rand(rng, Bool) && apply!(t, random_clifford(rng, 2, [q, q + 1]))
            end
        end
        t isa StabilizerTableau && canonicalize!(t)
        L = bt_QC._layout_marker(t)
        ws = t.word_workspace
        stab = bt_QC._stab(t, L)
        P = bt_QC._bit_phase_planes(t)
        m = t.m
        nwm = bt_QC._nwords(m)
        ok = true
        for trial in 1:12
            op = bt_span_pauli(rng, t, n, ())
            in_ref = bt_QC.in_span_and_coeffs!(t, L, op)
            c_ref = t.c_workspace[1:m] .& 1
            res_ref = t.res_workspace[1:2n] .& 1
            k_ref = bt_QC.phase_exponent_from_coeffs!(t, L)
            cut = bt_QC._bit_span_sparse_cut(nwm, storephase)
            for c in (cut, nwm, -1)
                opc = bt_QC._commutation_operand!(t, op)
                bt_QC._bit_pack_operand!(ws, opc, n)
                bt_QC._bit_coeff_mask!(t, L, opc)
                ok &= [bt_ws_bit(ws, j, 1) for j in 1:m] == c_ref
                ok &= all(iszero, [bt_ws_bit(ws, j, 1) for j in (m + 1):(64 * nwm)])
                if L === QM
                    nz = count(!iszero, ws[1:nwm, 1])
                    paths[nz <= cut ? :sparse : :dense] = 1
                    in_span, k = bt_QC._qm_bit_residual_phase!(ws, stab, P, m, t.zacc_workspace, c)
                else
                    in_span, k = bt_QC._gm_bit_residual_phase!(ws, stab, P, m)
                end
                ok &= in_span == in_ref && k == k_ref
                ok &= [bt_ws_bit(ws, q, 3) for q in 1:n] == res_ref[1:n] &&
                      [bt_ws_bit(ws, q, 4) for q in 1:n] == res_ref[(n + 1):(2n)]
                ok &= (in_span, k) == bt_QC._bit_residual_phase!(
                    (bt_QC._bit_pack_operand!(ws, opc, n); bt_QC._bit_coeff_mask!(t, L, opc); t), L)
                # The phase alone, as a pure state takes it.
                storephase || continue
                # The kernel left the residual where the operand was.
                remask!() = (bt_QC._bit_pack_operand!(ws, opc, n); bt_QC._bit_coeff_mask!(t, L, opc))
                L === QM || continue
                remask!()
                pairs = bt_QC._qm_bit_pair_parity!(ws, stab, m, t.zacc_workspace, c)
                ok &= mod(bt_QC._bit_mask_phase(ws, P, m) + 2 * pairs, 4) == k_ref
                remask!()
                ok &= bt_QC._qm_bit_mask_product_phase!(t, L) == k_ref
                paths[nz == 1 ? :one_word : nz <= bt_QC._BIT_SPAN_PROBE_WORDS ?
                      :probed : :unprobed] = 1
                count(!iszero, c_ref) == 1 && (paths[:one_generator] = 1)
            end
            ok || (@error "packed span pieces differ" TT layout storephase n kind trial op; break)
        end
        @test ok
        for O in (SinglePauli, GeneralPauli)
            mth = which(bt_QC._expect_int!, (typeof(t), typeof(L), O))
            @test basename(String(mth.file)) == "span_decomposition_bits.jl"
        end
    end
    for k in (:sparse, :dense, :one_word, :probed, :unprobed, :one_generator)
        @test (k => haskey(paths, k)) == (k => true)
    end
end

function bt_span_allocations(f, t)
    for _ in 1:3
        f(t)
    end
    return @allocated f(t)
end

@testset "bit span kernels allocate nothing" begin
    for TT in bt_TYPES, layout in (:generator, :qudit), storephase in (true, false)
        n = 130
        r = TT(2, n; storephase, eltype = UInt8)
        random_state!(Xoshiro(5), r)
        measure!(Xoshiro(6), r, SinglePauli(7, 0, 1))
        apply!(r, random_clifford(Xoshiro(7), 2, [1, 2]))
        S = stabilizer_matrix(r)
        zq = SinglePauli(7, 0, 1)
        v = reduce((a, b) -> a .⊻ b, (S[1:2n, j] for j in 1:3:n))
        dense = GeneralPauli(v, mod(sum(v[1:n] .* v[(n + 1):(2n)]), 2))
        rng = Xoshiro(8)

        # Pure state: fixed outcomes and expectation values.
        t = TT(2, S; eltype = Bits, layout, storephase)
        for canonical in (TT === StabilizerTableau ? (false, true) : (false,))
            canonical && canonicalize!(t)
            for op in (zq, dense)
                @test bt_span_allocations(x -> measure!(rng, x, op), t) == 0
                (canonical || TT === DestabilizerTableau) &&
                    @test bt_span_allocations(x -> expect_int!(x, op), t) == 0
            end
        end

        # One generator short: fixed outcomes, and the append of the missing one
        # (dense) or of a single-qubit operand on a qubit the state leaves mixed.
        Sm = hcat(S[:, 1:(n - 1)], zeros(Int, size(S, 1)))
        last = GeneralPauli(S[1:2n, n], storephase ? S[2n + 1, n] : 0)
        a = TT(2, Sm; m = n - 1, eltype = Bits, layout, storephase)
        w = copy(a)
        @test bt_span_allocations(x -> expect_int!(x, last), a) == 0
        @test bt_span_allocations(x -> (copy!(x, a); measure!(rng, x, last)), w) == 0
        @test w.m == n
        b = TT(2, n; state = :mixed, eltype = Bits, layout, storephase)
        for q in 1:(n - 1)
            measure!(rng, b, DoublePauli(q, 0, 1, q + 1, 0, 1))
        end
        w = copy(b)
        for op in (SinglePauli(64, 0, 1), SinglePauli(65, 1, 0, 0), DoublePauli(1, 0, 1, 130, 0, 1))
            @test bt_span_allocations(x -> (copy!(x, b); measure!(rng, x, op)), w) == 0
        end
    end
end

end
