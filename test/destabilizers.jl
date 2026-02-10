using QuditClifford
using Test

@inline function _symp(A::AbstractMatrix{<:Integer}, colA::Int, B::AbstractMatrix{<:Integer}, colB::Int, n::Int)
    s = 0
    @inbounds for q in 1:n
        s += A[q, colA] * B[n + q, colB]
        s -= A[n + q, colA] * B[q, colB]
    end
    return s
end

@testset "Destabilizers" begin
    @testset "Duality after init" begin
        for (d, n) in ((2, 3), (3, 2))
            tab = DestabilizerTableau(d, n; state=:product, basis=:Z)
            @test tab.m == n
            for i in 1:n
                for j in 1:n
                    val = mod(_symp(tab.destab, i, tab.stab, j, tab.n), d)
                    @test val == (i == j ? 1 : 0)
                end
            end
        end
    end

    @testset "Noncommuting measurement duality" begin
        tab = DestabilizerTableau(2, 1; state=:product, basis=:Z)
        out = measure!(tab, SinglePauli(1, 1, 0); outcome=1) # measure X
        @test out == 1
        @test tab.m == 1
        val = mod(_symp(tab.destab, 1, tab.stab, 1, tab.n), tab.d)
        @test val == 1
    end

    @testset "Commuting append duality" begin
        tab = DestabilizerTableau(3, 1; state=:mixed)
        out = measure!(tab, SinglePauli(1, 0, 1); outcome=2) # measure Z
        @test out == 2
        @test tab.m == 1
        val = mod(_symp(tab.destab, 1, tab.stab, 1, tab.n), tab.d)
        @test val == 1
    end

    @testset "Expectation agreement" begin
        ops = [
            SinglePauli(1, 0, 1),
            SinglePauli(1, 1, 0),
            DoublePauli(1, 0, 1, 2, 0, 1),
        ]

        tab = StabilizerTableau(2, 2; state=:product, basis=:Z)
        ref_vals = map(op -> expect_int!(tab, op), ops)

        tab = DestabilizerTableau(2, 2; state=:product, basis=:Z)
        for (idx, op) in pairs(ops)
            @test ref_vals[idx] == expect_int!(tab, op)
        end
    end
end
