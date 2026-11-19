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
        # agree with the stacked-matrix methods
        @test ts_QC.dot_xz_col(b, i, j, d) == ts_QC.dot_xz_col(S, n, i, j, d)

        ts_QC._zero!(b)
        @test all(iszero, S[1:2n, :]) && S[2n+1, :] == S0[2n+1, :]
        A = view(rand(rng, 0:d-1, 3, 3), 1:2, :)
        ts_QC._zero!(A); @test all(iszero, A)
    end
    @testset "tableau views $label d=$d sp=$sp" for
            (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)],
            d in (2, 3), sp in (true, false)
        n = 4
        tab = TT(d, n; state = :product, basis = :Z, storephase = sp)
        ts_QC._rset!(ts_QC._stab(tab), 1, 1, 2)
        Sm = stabilizer_matrix(tab)
        b = ts_QC._stab(tab)
        @test all(ts_QC._rget(b, r, j) == Sm[r, j] for r in 1:2n, j in 1:n)
        @test ts_QC._phase(tab) == (sp ? Sm[2n+1, :] : Int[])
        if TT === DestabilizerTableau
            Dm = destabilizer_matrix(tab)
            bd = ts_QC._destab(tab)
            @test all(ts_QC._rget(bd, r, j) == Dm[r, j] for r in 1:2n, j in 1:n)
        end
    end
end
