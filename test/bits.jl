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

end
