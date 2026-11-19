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
    end
end
