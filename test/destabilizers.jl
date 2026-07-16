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

    @testset "Commuting append projects and reorthogonalizes destabilizers" begin
        # Start with S₁ = X₁X₂. Measuring the commuting operator X₁ leaves
        # residual X₂. Constructing its dual first proposes Z₂, which must be
        # projected against S₁; the old dual must then be reorthogonalized to X₁.
        raw = zeros(Int, 5, 2)
        raw[1, 1] = 1
        raw[2, 1] = 1
        tab = DestabilizerTableau(2, raw; m=1, storephase=true)

        @test measure!(tab, SinglePauli(1, 1, 0); outcome=0) == 0
        @test tab.m == 2
        @test tab.stab[1:4, 1] == Int[1, 1, 0, 0]
        @test tab.stab[1:4, 2] == Int[1, 0, 0, 0]
        @test tab.destab[:, 1] == Int[0, 0, 0, 1] # Z₂
        @test tab.destab[:, 2] == Int[0, 0, 1, 1] # Z₁Z₂

        for i in 1:tab.m, j in 1:tab.m
            val = mod(_symp(tab.destab, i, tab.stab, j, tab.n), tab.d)
            @test val == (i == j ? 1 : 0)
        end
    end

    @testset "Noncommuting replacement reorthogonalizes unaffected duals" begin
        # For S₁=X₁, S₂=X₂ and P=Z₁X₂, replacing S₁ makes the
        # old dual of S₂ fail to commute with P. The update must correct that dual.
        tab = DestabilizerTableau(2, 2; state=:product, basis=:X)
        op = DoublePauli(1, 0, 1, 2, 1, 0)

        @test measure!(tab, op; outcome=0) == 0
        @test tab.stab[1:4, 1] == Int[0, 1, 1, 0]
        @test tab.stab[1:4, 2] == Int[0, 1, 0, 0]
        @test tab.destab[:, 1] == Int[1, 0, 0, 0] # X₁
        @test tab.destab[:, 2] == Int[1, 0, 0, 1] # X₁Z₂

        for i in 1:tab.m, j in 1:tab.m
            val = mod(_symp(tab.destab, i, tab.stab, j, tab.n), tab.d)
            @test val == (i == j ? 1 : 0)
        end
    end

    @testset "Canonicalization preserves duality" begin
        for d in (2, 3)
            raw = zeros(Int, 5, 2)
            raw[3, 1] = 1      # Z1
            raw[3, 2] = 1      # Z1 Z2
            raw[4, 2] = 1

            tab = DestabilizerTableau(d, copy(raw); m=2, storephase=true)
            canonicalize!(tab)

            for i in 1:tab.m
                for j in 1:tab.m
                    val = mod(_symp(tab.destab, i, tab.stab, j, tab.n), d)
                    @test val == (i == j ? 1 : 0)
                end
            end

            ref = StabilizerTableau(d, copy(raw); m=2, storephase=true)
            op = copy(raw[:, 2])
            @test expect_int!(tab, op) == expect_int!(ref, op)
        end
    end

    @testset "Mixed-state canonicalization clears inactive caches" begin
        raw = zeros(Int, 7, 3)
        raw[4, 1] = 1 # Z₁, with two inactive capacity columns
        tab = DestabilizerTableau(3, raw; m=1, storephase=true)
        fill!(tab.xdotz_cache, 2)

        canonicalize!(tab)

        @test tab.iscanonical
        @test tab.xdotz_cache[1] == 0
        @test tab.xdotz_cache[2:3] == [0, 0]
        @test mod(_symp(tab.destab, 1, tab.stab, 1, tab.n), tab.d) == 1
    end

    @testset "Purity checks preserve duality" begin
        d = 3
        raw = zeros(Int, 5, 2)
        raw[3, 1] = 1      # Z1
        raw[3, 2] = 1      # Z1 Z2
        raw[4, 2] = 1

        tab = DestabilizerTableau(d, copy(raw); m=2, storephase=true)
        op = copy(raw[:, 2])
        expected = expect_int!(tab, op)
        @test QuditClifford.is_pure(tab)
        @test expect_int!(tab, op) == expected

        for i in 1:tab.m
            for j in 1:tab.m
                val = mod(_symp(tab.destab, i, tab.stab, j, tab.n), d)
                @test val == (i == j ? 1 : 0)
            end
        end
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
