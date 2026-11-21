using QuditClifford, Test, Random
const ts_QC = QuditClifford

ts_eye(n) = Int.((1:n) .== (1:n)')

@testset "stabilizer_matrix" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)],
        d in (2, 3), sp in (true, false)
        @testset "$label d=$d storephase=$sp" begin
            n = 3
            tab = TT(d, n; state = :product, basis = :Z, storephase = sp)
            S = stabilizer_matrix(tab)
            @test S == [zeros(Int, n, n); ts_eye(n); zeros(Int, Int(sp), n)]
            @test stabilizer_matrix(tab; layout = :qudit) == permutedims(S)
            @test_throws ArgumentError stabilizer_matrix(tab; layout = :rows)
            S[n + 1, 1] = 0
            @test stabilizer_matrix(tab)[n + 1, 1] == 1          # a fresh copy
            @test stabilizer_matrix(TT(d, n; storephase = sp)) == zeros(Int, 2n + Int(sp), n)
            S2 = copy(stabilizer_matrix(tab)); S2[1, 2] = 1      # X on qudit 1 in generator 2
            ts_QC._load_stabilizer_matrix!(tab, S2)
            @test stabilizer_matrix(tab) == S2
            @test_throws ArgumentError ts_QC._load_stabilizer_matrix!(tab, zeros(Int, 2n + 2, n))
        end
    end
    for d in (2, 3)
        n = 3
        tab = DestabilizerTableau(d, n; state = :product, basis = :Z)
        D = destabilizer_matrix(tab); S = stabilizer_matrix(tab)
        @test size(D) == (2n, n)
        pair(a, b) = mod(sum(a[1:n] .* b[n+1:2n]) - sum(a[n+1:2n] .* b[1:n]), d)
        @test [pair(D[:, j], S[:, k]) for j in 1:n, k in 1:n] == ts_eye(n)
        @test destabilizer_matrix(tab; layout = :qudit) == permutedims(D)
        ts_QC._load_destabilizer_matrix!(tab, D)
        @test destabilizer_matrix(tab) == D
        @test_throws ArgumentError destabilizer_matrix(tab; layout = :rows)
        @test_throws ArgumentError ts_QC._load_destabilizer_matrix!(tab, zeros(Int, 2n + 1, n))
        D2 = copy(D); D2[1, 2] = mod(D2[1, 2] + 1, d)
        ts_QC._load_destabilizer_matrix!(tab, D2)
        @test destabilizer_matrix(tab) == D2
    end
end

@testset "accessor layer" begin
    for d in (2, 3, 5), n in (1, 4, 7)
        rng = MersenneTwister(1000d + n)
        rv(len) = rand(rng, 0:d-1, len)
        S = rand(rng, 0:d-1, 2n + 1, n)
        (b, ph) = ts_QC._stacked_views(S, n, true)
        @test ts_QC._nq(b) == n && ts_QC._ncap(b) == n
        @test ph == S[2n+1, :]
        @test length(ts_QC._stacked_views(S, n, false)[2]) == 0
        @test all(ts_QC._xget(b, q, j) == S[q, j] && ts_QC._zget(b, q, j) == S[n+q, j]
                  for q in 1:n, j in 1:n)
        @test all(ts_QC._rget(b, r, j) == S[r, j] for r in 1:2n, j in 1:n)
        @test ts_QC._xgen(b, 1) == S[1:n, 1] && ts_QC._zgen(b, 1) == S[n+1:2n, 1]
        jc = 2 % n + 1
        v = zeros(Int, 2n); ts_QC._copy_gen!(v, b, jc); @test v == S[1:2n, jc]

        S0 = copy(S); ts_QC._xset!(b, 0, 1, 1); ts_QC._zset!(b, 0, 1, 1); ts_QC._rset!(b, 1, n + 1, n)
        @test S[1, 1] == 0 && S[n+1, n] == 1 && (n == 1 || S[n+1, 1] == 0) && S[2n+1, :] == S0[2n+1, :]
        copyto!(S, S0)

        w = rv(2n); u0 = rv(2n); zacc = rv(n); genws = rv(2n)
        for a in 0:d-1
            copyto!(S, S0)
            ts_QC._gen_addmul!(b, 1, w, a, d)
            @test S[1:2n, 1] == mod.(S0[1:2n, 1] .+ a .* w, d)
            @test S[2n+1, :] == S0[2n+1, :] && S[:, 2:n] == S0[:, 2:n]
            copyto!(S, S0)
            ts_QC._gen_submul!(b, 1, w, a, d)
            @test S[1:2n, 1] == mod.(S0[1:2n, 1] .- a .* w, d)
            copyto!(S, S0)
            ts_QC._gen_mulcopy!(b, 1, w, a, d)
            @test S[1:2n, 1] == mod.(a .* w, d)
            copyto!(S, S0)
            ts_QC._gen_scale!(b, 1, a, d)
            @test S[1:2n, 1] == mod.(a .* S0[1:2n, 1], d)
            copyto!(S, S0)
            u = copy(u0); ts_QC._vec_submul!(u, b, 1, a, d)
            @test u == mod.(u0 .- a .* S0[1:2n, 1], d)
            u = copy(u0); ts_QC._vec_addmul!(u, b, 1, a, d)
            @test u == mod.(u0 .+ a .* S0[1:2n, 1], d)
            if n >= 4
                j, k = 3, 4
                copyto!(S, S0)
                ts_QC._gen_submul!(b, j, b, k, a, d)
                @test S[1:2n, j] == mod.(S0[1:2n, j] .- a .* S0[1:2n, k], d)
                @test S[1:2n, k] == S0[1:2n, k]
                copyto!(S, S0)
                ts_QC._gen_addmul!(b, j, b, k, a, d)
                @test S[1:2n, j] == mod.(S0[1:2n, j] .+ a .* S0[1:2n, k], d)
            end
            # a second block as source
            S2 = rand(rng, 0:d-1, 2n + 1, n); (b2, _) = ts_QC._stacked_views(S2, n, true)
            copyto!(S, S0)
            ts_QC._gen_addmul!(b, 1, b2, n, a, d)
            @test S[1:2n, 1] == mod.(S0[1:2n, 1] .+ a .* S2[1:2n, n], d)
            copyto!(S, S0)
            ts_QC._gen_submul!(b, 1, b2, n, a, d)
            @test S[1:2n, 1] == mod.(S0[1:2n, 1] .- a .* S2[1:2n, n], d)

            sites = sort(randperm(rng, n)[1:rand(rng, 0:n)])
            ws = zeros(Int, 2n)
            ws[sites] .= rv(length(sites)); ws[n .+ sites] .= rv(length(sites))
            copyto!(S, S0)
            ts_QC._gen_addmul_sites!(b, 1, ws, sites, length(sites), a, d)
            @test S[1:2n, 1] == mod.(S0[1:2n, 1] .+ a .* ws, d)
            copyto!(S, S0)
            ts_QC._gen_submul_sites!(b, 1, ws, sites, length(sites), a, d)
            @test S[1:2n, 1] == mod.(S0[1:2n, 1] .- a .* ws, d)
            S2[:, n] .= 0
            S2[sites, n] .= rv(length(sites)); S2[n .+ sites, n] .= rv(length(sites))
            copyto!(S, S0)
            ts_QC._gen_addmul_sites!(b, 1, b2, n, sites, length(sites), a, d)
            @test S[1:2n, 1] == mod.(S0[1:2n, 1] .+ a .* S2[1:2n, n], d)
            copyto!(S, S0)
            ts_QC._gen_submul_sites!(b, 1, b2, n, sites, length(sites), a, d)
            @test S[1:2n, 1] == mod.(S0[1:2n, 1] .- a .* S2[1:2n, n], d)
            @test S[2n+1, :] == S0[2n+1, :]

            # nonzero source outside the sites: non-site entries must stay put
            if length(sites) < n
                outside = setdiff(1:n, sites)
                nsel = length(sites)
                wf = rv(2n)
                S2[1:2n, n] .= rv(2n)
                S2p = copy(S2)
                rows = vcat(sites, n .+ sites)
                orows = vcat(outside, n .+ outside)
                copyto!(S, S0)
                ts_QC._gen_addmul_sites!(b, 1, wf, sites, nsel, a, d)
                @test S[orows, 1] == S0[orows, 1]
                @test S[rows, 1] == mod.(S0[rows, 1] .+ a .* wf[rows], d)
                copyto!(S, S0)
                ts_QC._gen_submul_sites!(b, 1, wf, sites, nsel, a, d)
                @test S[orows, 1] == S0[orows, 1]
                @test S[rows, 1] == mod.(S0[rows, 1] .- a .* wf[rows], d)
                copyto!(S, S0)
                ts_QC._gen_addmul_sites!(b, 1, b2, n, sites, nsel, a, d)
                @test S[orows, 1] == S0[orows, 1]
                @test S[rows, 1] == mod.(S0[rows, 1] .+ a .* S2p[rows, n], d)
                copyto!(S, S0)
                ts_QC._gen_submul_sites!(b, 1, b2, n, sites, nsel, a, d)
                @test S[orows, 1] == S0[orows, 1]
                @test S[rows, 1] == mod.(S0[rows, 1] .- a .* S2p[rows, n], d)
                @test S2 == S2p
            end
        end

        copyto!(S, S0)
        if n >= 2
            ts_QC._swap_gens!(b, 1, 2)
            @test S[1:2n, 1] == S0[1:2n, 2] && S[1:2n, 2] == S0[1:2n, 1]
            @test S[2n+1, :] == S0[2n+1, :]
            copyto!(S, S0)
            ts_QC._move_gen!(b, 1, 2); ts_QC._zero_gen!(b, 2)
            @test S[1:2n, 1] == S0[1:2n, 2] && all(iszero, S[1:2n, 2]) && S[2n+1, :] == S0[2n+1, :]
        end
        copyto!(S, S0)
        ts_QC._load_gen!(b, 1, w)
        @test S[1:2n, 1] == w && S[2n+1, :] == S0[2n+1, :]
        copyto!(S, S0)

        i, j = 1, n
        @test ts_QC.dot_xz_col(b, i, j, d) == mod(sum(S[1:n, i] .* S[n+1:2n, j]), d)
        @test ts_QC.dot_xz_col(b, j, d) == mod(sum(S[1:n, j] .* S[n+1:2n, j]), d)
        @test ts_QC.dot_xz_col_vs_zacc(b, j, zacc, d) == mod(sum(S[1:n, j] .* zacc), d)
        @test ts_QC.dot_xz_ws_vs_col(genws, b, j, d) == mod(sum(genws[1:n] .* S[n+1:2n, j]), d)

        ts_QC._zero!(b)
        @test all(iszero, S[1:2n, :]) && S[2n+1, :] == S0[2n+1, :]
        A = view(rand(rng, 0:d-1, 3, 3), 1:2, :)
        ts_QC._zero!(A); @test all(iszero, A)
    end
    # A matrix with no rows still gives an empty phase view, of the same type.
    (b0, ph0) = ts_QC._stacked_views(zeros(Int, 0, 0), 0, false)
    @test ts_QC._nq(b0) == 0 && isempty(ph0)
    @test typeof(ph0) == typeof(ts_QC._stacked_views(zeros(Int, 1, 0), 0, true)[2])
    @testset "tableau views $label d=$d sp=$sp" for
            (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)],
            d in (2, 3), sp in (true, false)
        n = 4
        tab = TT(d, n; state = :product, basis = :Z, storephase = sp)
        ts_QC._rset!(ts_QC._stab(tab, ts_QC._layout_marker(tab)), 1, 1, 2)
        Sm = stabilizer_matrix(tab)
        b = ts_QC._stab(tab, ts_QC._layout_marker(tab))
        @test all(ts_QC._rget(b, r, j) == Sm[r, j] for r in 1:2n, j in 1:n)
        @test ts_QC._phase(tab) == (sp ? Sm[2n+1, :] : Int[])
        if TT === DestabilizerTableau
            Dm = destabilizer_matrix(tab)
            bd = ts_QC._destab(tab, ts_QC._layout_marker(tab))
            @test all(ts_QC._rget(bd, r, j) == Dm[r, j] for r in 1:2n, j in 1:n)
        end
    end
end

const ts_TYPES = [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]

# The storage arrays of a tableau, workspaces excluded.
ts_arrays(t::StabilizerTableau) = Any[t.stab_x, t.stab_z, t.phase]
ts_arrays(t::DestabilizerTableau) = Any[t.stab_x, t.stab_z, t.phase, t.destab_x, t.destab_z]

ts_matrices(t::StabilizerTableau) = (stabilizer_matrix(t),)
ts_matrices(t::DestabilizerTableau) = (stabilizer_matrix(t), destabilizer_matrix(t))

function ts_random_gate(rng, d, n)
    q = rand(rng, 1:n)
    r = mod1(q + rand(rng, 1:(n - 1)), n)
    k = rand(rng, 1:7)
    k == 1 && return Fourier(q)
    k == 2 && return Phase(q)
    k == 3 && return Multiplier(q, rand(rng, 1:(d - 1)))
    k == 4 && return PauliGate(q, rand(rng, 0:(d - 1)), rand(rng, 0:(d - 1)))
    k == 5 && return SUM(q, r, rand(rng, 1:(d - 1)))
    k == 6 && return CPhase(q, r, rand(rng, 1:(d - 1)))
    return SWAP(q, r)
end

# A Hermitian (at d = 2) one- or two-qudit Pauli.
function ts_random_pauli(rng, d, n)
    q = rand(rng, 1:n)
    r = mod1(q + rand(rng, 1:(n - 1)), n)
    x1, z1, x2, z2 = rand(rng, 0:(d - 1), 4)
    (x1, z1) == (0, 0) && (z1 = 1)
    ph = d == 2 ? mod(x1 * z1 + x2 * z2, 2) : 0
    return rand(rng, Bool) ? SinglePauli(q, x1, z1, d == 2 ? mod(x1 * z1, 2) : 0) :
                             DoublePauli(q, x1, z1, r, x2, z2, ph)
end

@testset "removed fields" begin
    for (label, TT) in ts_TYPES
        tab = TT(3, 2; state = :ghz)
        @test_throws ArgumentError tab.stab
        err = try tab.stab; nothing catch e; e end
        @test err isa ArgumentError && occursin("stabilizer_matrix", err.msg)
        if TT === DestabilizerTableau
            @test_throws ArgumentError tab.destab
            err = try tab.destab; nothing catch e; e end
            @test err isa ArgumentError && occursin("destabilizer_matrix", err.msg)
        end
        @test tab.m == 2 && tab.n == 2        # other fields read as before
    end
end

@testset "split storage" begin
    for (label, TT) in ts_TYPES, d in (2, 3), sp in (true, false), n in (1, 4)
        tab = TT(d, n; state = :ghz, storephase = sp, layout = :generator, eltype = UInt8)
        @test size(tab.stab_x) == size(tab.stab_z) == (n, n)
        @test length(tab.phase) == (sp ? n : 0)
        @test tab.phase isa Vector{Int}
        if TT === DestabilizerTableau
            @test size(tab.destab_x) == size(tab.destab_z) == (n, n)
        end
        S = stabilizer_matrix(tab)
        @test tab.stab_x == S[1:n, :] && tab.stab_z == S[n+1:2n, :]
        sp && @test tab.phase == S[2n+1, :]
        raw = TT(d, S; storephase = sp, eltype = UInt8)
        @test size(raw.stab_x) == (n, n) && length(raw.phase) == (sp ? n : 0)
        @test stabilizer_matrix(raw) == S
    end
end

# Line slot (byte offset mod 4096, rounded down to 128) of a block's first entry.
ts_slot(A) = Int(UInt(pointer(A)) % 4096 ÷ 128 * 128)

@testset "padding rule in bytes" begin
    for T in (Int, Int64, Int32, UInt16, UInt8)
        sz = sizeof(T)
        for n in (1, 7, 31, 32, 33, 64, 100, 255, 256, 257, 511, 512, 513, 1023, 1024)
            L = ts_QC._padded_rows(n, sz)
            if n * sz < ts_QC._PAD_MIN_COL_BYTES
                @test L == n
            else
                # Stride an odd multiple of 128 bytes, at most 2·128 bytes of padding.
                @test mod(L * sz, 128) == 0 && isodd(L * sz ÷ 128)
                @test n <= L < n + 256 ÷ sz
            end
            n in (1, 33, 512, 1024) || continue
            blocks = [ts_QC._padded_block(T, n, s) for s in (0, 1024, 2048, 3072)]
            @test all(b -> size(b) == (n, n) && eltype(b) === T && iszero(b), blocks)
            @test allunique(map(b -> objectid(parent(b)), blocks))
            @test all(b -> size(parent(b), 1) == L, blocks)
            if L != n
                @test map(ts_slot, blocks) == [0, 1024, 2048, 3072]
            else
                @test all(b -> parent(b) == b, blocks)       # no shift, no spare columns
            end
        end
    end
    @test Int === Int32 || ts_QC._padded_rows(512, sizeof(Int)) == 528
    @test_throws ArgumentError ts_QC._padded_rows(100, 3)
    @test_throws ArgumentError ts_QC._padded_rows(100, 256)
end

@testset "padded block layout" begin
    for (label, TT) in ts_TYPES, E in (UInt8, UInt16, Int),
        n in (1, 31, 32, 33, 64, 511, 512, 513, 1024)
        tab = TT(2, n; state = :ghz, eltype = E)
        fields = TT === DestabilizerTableau ?
            (tab.stab_x, tab.stab_z, tab.destab_x, tab.destab_z) : (tab.stab_x, tab.stab_z)
        L = ts_QC._padded_rows(n, sizeof(E))
        @test allunique(map(b -> objectid(parent(b)), fields))
        @test all(b -> size(b) == (n, n) && size(parent(b), 1) == L, fields)
        L != n && @test map(ts_slot, fields) == (0, 2048, 1024, 3072)[1:length(fields)]
        # A copy has the same slots in parents of its own.
        c = copy(tab)
        cf = TT === DestabilizerTableau ?
            (c.stab_x, c.stab_z, c.destab_x, c.destab_z) : (c.stab_x, c.stab_z)
        L != n && @test map(ts_slot, cf) == map(ts_slot, fields)
        @test all(i -> parent(cf[i]) !== parent(fields[i]), eachindex(cf))
        @test c == tab
        # Zeroing one block leaves the others and the spare columns alone.
        if n > 1
            apply!(tab, Fourier(1))
            before = copy(tab.stab_x)
            ts_QC._zero!(tab.stab_z)
            @test iszero(tab.stab_z) && tab.stab_x == before
            @test iszero(parent(tab.stab_z))
        end
    end
end

@testset "copy shares no storage" begin
    for (label, TT) in ts_TYPES, d in (2, 3), sp in (true, false)
        n = 4
        rng = Xoshiro(17d + sp)
        tab = TT(d, n; state = :ghz, storephase = sp)
        apply!(tab, Fourier(2))
        before = ts_matrices(tab)

        c = copy(tab)
        @test !any(a === b for a in ts_arrays(c), b in ts_arrays(tab))
        apply!(c, SUM(1, 3))
        measure!(rng, c, SinglePauli(4, 1, 0))
        @test ts_matrices(tab) == before

        dst = TT(d, n; storephase = sp)
        copy!(dst, tab)
        @test ts_matrices(dst) == before
        @test !any(a === b for a in ts_arrays(dst), b in ts_arrays(tab))
        apply!(dst, Phase(1))
        measure!(rng, dst, SinglePauli(3, 1, 0))
        @test ts_matrices(tab) == before
    end
end

@testset "phase-less tableau tracks the phased one" begin
    for (label, TT) in ts_TYPES, d in (2, 3)
        n = 5
        rng = Xoshiro(4242 + d)
        tp = TT(d, n; storephase = true)
        tq = TT(d, n; storephase = false)
        steps = shuffle(rng, vcat(fill(:gate, 30), fill(:measure, 10)))
        for s in steps
            if s === :gate
                g = ts_random_gate(rng, d, n)
                apply!(tp, g); apply!(tq, g)
            else
                op = ts_random_pauli(rng, d, n)
                measure!(tp, op; outcome = 0); measure!(tq, op; outcome = 0)
            end
            @test stabilizer_matrix(tq) == stabilizer_matrix(tp)[1:2n, :]
            @test tq.m == tp.m
            TT === DestabilizerTableau && @test destabilizer_matrix(tq) == destabilizer_matrix(tp)
        end
        @test tp.m > 0
    end
end

@testset "tiny sizes" begin
    for (label, TT) in ts_TYPES, d in (2, 3)
        @testset "$label d=$d n=1" begin
            for (state, basis) in ((:mixed, :Z), (:product, :X), (:product, :Y), (:product, :Z), (:ghz, :Z))
                tab = state === :product ? TT(d, 1; state, basis) : TT(d, 1; state)
                ent = entanglement_entropy(tab, [1])
                @test ent == 1 - tab.m        # pure on the whole system, else maximally mixed
                apply!(tab, Fourier(1))
                c = copy(tab)
                @test c == tab
                S = stabilizer_matrix(tab)
                if tab.m == 1
                    # The generator itself: a deterministic outcome, state unchanged.
                    op = SinglePauli(1, S[1, 1], S[2, 1], S[3, 1])
                    @test measure!(tab, op) == 0
                    @test stabilizer_matrix(tab) == S
                    # The conjugate basis: a random outcome.
                    other = S[1, 1] == 0 ? SinglePauli(1, 1, 0) : SinglePauli(1, 0, 1)
                    @test measure!(tab, other; outcome = 1) == 1
                    @test tab.m == 1
                    @test measure!(tab, other) == 1
                else
                    @test measure!(tab, SinglePauli(1, 0, 1); outcome = 1) == 1
                    @test tab.m == 1 && measure!(tab, SinglePauli(1, 0, 1)) == 1
                end
                @test entanglement_entropy(tab, [1]) == 0
                canonicalize!(tab)
                @test tab.iscanonical && copy(tab) == tab
                @test tab != c
            end
        end
        @testset "$label d=$d m=0" begin
            for n in (1, 3)
                tab = TT(d, n; state = :mixed)
                @test tab.m == 0
                @test stabilizer_matrix(tab) == zeros(Int, 2n + 1, n)
                @test expect!(tab, SinglePauli(1, 0, 1)) == 0
                @test !is_pure(tab)
                @test entanglement_entropy(tab, [1]) == 1
                @test entanglement_entropy(tab, collect(1:n)) == n
                @test copy(tab) == tab
                @test measure!(tab, SinglePauli(1, 1, 0); outcome = 0) == 0
                @test tab.m == 1 && stabilizer_matrix(tab)[:, 1] == [1; zeros(Int, 2n - 1); 0]
            end
        end
    end
end

@testset "layout marker" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
        tab = TT(d, 5; state = :ghz, layout = :generator, eltype = UInt8)
        @test tab.layout === QuditClifford.GeneratorMajor()
        @test fieldtype(TT, :layout) === Union{QuditClifford.GeneratorMajor, QuditClifford.QuditMajor}
        @test QuditClifford._layout_marker(tab) === QuditClifford.GeneratorMajor()
        @test QuditClifford._stab(tab, QuditClifford.GeneratorMajor()) isa QuditClifford.XZBlock{QuditClifford.GeneratorMajor}
        @test !hasmethod(QuditClifford._stab, Tuple{TT})
    end
    @test !hasmethod(QuditClifford._destab, Tuple{DestabilizerTableau})
end

@testset "no raw storage access outside the storage files" begin
    srcdir = joinpath(pkgdir(QuditClifford), "src")
    allowed = ("tableau_storage.jl", "stabilizer_tableau.jl", "destabilizer_tableau.jl", "layout.jl")
    for f in readdir(srcdir)
        endswith(f, ".jl") || continue
        text = read(joinpath(srcdir, f), String)
        f in allowed || @test !occursin(r"\.(stab|destab)_[xz]\b", text)
        @test !occursin(r"\.(stab|destab)\b", text)
        @test !occursin(r"\b_(de)?stab\(\s*[A-Za-z_][A-Za-z_0-9]*\s*\)", text)
    end
end

@testset "no raw block access outside the block and kernel files" begin
    srcdir = joinpath(pkgdir(QuditClifford), "src")
    # Explicit layout construction: the block files (bit_block.jl for its
    # unpacked copies), the storage files and the constructors (which build
    # generator-major and then switch). `BitBlock`s are built only in the bit
    # block file and the storage file.
    construct_ok = ("xz_block.jl", "bit_block.jl", "tableau_storage.jl", "layout.jl",
                    "stabilizer_tableau.jl")
    # Raw `b.x` / `b.z` access, with the number of allowed matches per file:
    # the layout-specific kernels, the packed block file and its kernel files,
    # and the storage files that build tableaux from blocks. Kernel files that
    # do not exist count as zero matches.
    raw_ok = Dict(
        "bit_block.jl" => 56,
        "rank_workspaces.jl" => 14,
        "span_decomposition.jl" => 9,
        "rank_copies_bits.jl" => 8,
        "entanglement_entropy.jl" => 6,
        "check_purity.jl" => 4,
        "apply_qudit_major.jl" => 2,
        "measure_qudit_major.jl" => 2,
        "canonicalize_qudit_major.jl" => 4,
        "stabilizer_tableau.jl" => 2,
        "destabilizer_tableau.jl" => 6,
        "tableau_storage.jl" => 4,
    )
    raw = r"\bb\.x\b|\bb\.z\b|\.x\[|\.z\["
    for f in readdir(srcdir)
        endswith(f, ".jl") || continue
        text = read(joinpath(srcdir, f), String)
        f in construct_ok ||
            @test !occursin(r"XZBlock\{(GeneratorMajor|QuditMajor)\}\(", text)
        f in ("bit_block.jl", "tableau_storage.jl") ||
            @test !occursin(r"BitBlock\{\w+\}\(", text)
        f == "xz_block.jl" && continue
        @test (f => count(raw, text)) == (f => get(raw_ok, f, 0))
    end
end

@testset "no 32-bit-unsafe shifts in bit code" begin
    srcdir = joinpath(pkgdir(QuditClifford), "src")
    # `1 << b` shifts an `Int`, which drops bits 32-63 under `Int32`.
    for f in ("bit_block.jl", "rank_copies_bits.jl", "apply_bits.jl", "measure_bits.jl", "canonicalize_bits.jl")
        path = joinpath(srcdir, f)
        isfile(path) || continue
        for (i, line) in enumerate(eachline(path))
            @test (f, i, occursin(r"(?<![\w)])1\s*<<", line)) == (f, i, false)
        end
    end
end
