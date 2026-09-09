using QuditClifford
using Test
using Random

@inline function _symp(A::AbstractMatrix{<:Integer}, colA::Int, B::AbstractMatrix{<:Integer}, colB::Int, n::Int)
    s = 0
    @inbounds for q in 1:n
        s += A[q, colA] * B[n + q, colB]
        s -= A[n + q, colA] * B[q, colB]
    end
    return s
end

@inline function _assert_duality(tab)
    for i in 1:tab.m, j in 1:tab.m
        val = mod(_symp(tab.destab, i, tab.stab, j, tab.n), tab.d)
        @test val == (i == j ? 1 : 0)
    end
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

    @testset "Noncommuting replacement corrects several duals at once" begin
        # S_j = X_j with duals D_j = Z_j (up to normalisation). P = Z1 X2 X3
        # anticommutes with S1 only, so S1 is the pivot -- but P also fails to
        # commute with BOTH D2 and D3, so the update must correct two duals,
        # not just the first one it meets.
        for d in (2, 3)
            tab = DestabilizerTableau(d, 3; state=:product, basis=:X)
            d2_before = copy(tab.destab[:, 2])
            d3_before = copy(tab.destab[:, 3])

            @test measure!(tab, TriplePauli(1, 0, 1, 2, 1, 0, 3, 1, 0);
                           outcome=0, phase_policy=1) == 0

            @test tab.destab[:, 2] != d2_before   # both duals really moved
            @test tab.destab[:, 3] != d3_before
            _assert_duality(tab)
        end
    end

    @testset "Noncommuting replacement duality for x-only and z-only ops" begin
        # The measured operator's support must be read from BOTH halves of the
        # tableau column. Each case below is built so the pivot is qudit 2 and
        # the dual of qudit 1 genuinely needs correcting, so reading only one
        # half of the column leaves duality broken rather than merely untested.
        #
        #   z-only: S = {Z1, X2}, D = {X1, Z2}, P = Z1 Z2 -> <D1,P> != 0
        #   x-only: S = {X1, Z2}, D = {Z1, X2}, P = X1 X2 -> <D1,P> != 0
        for d in (2, 3)
            for (basis, op) in (([:Z, :X], DoublePauli(1, 0, 1, 2, 0, 1)),
                                ([:X, :Z], DoublePauli(1, 1, 0, 2, 1, 0)))
                tab = DestabilizerTableau(d, 2; state=:product, basis=basis)
                d1_before = copy(tab.destab[:, 1])

                @test measure!(tab, op; outcome=0, phase_policy=1) == 0

                @test tab.destab[:, 1] != d1_before   # the dual really moved
                _assert_duality(tab)
            end
        end
    end

    @testset "Noncommuting replacement duality for a dense GeneralPauli" begin
        # Dense operator: support covers every qudit, in both halves.
        for d in (2, 3)
            n = 4
            tab = DestabilizerTableau(d, n; state=:product, basis=:Z)
            xz = ones(Int, 2n)
            op = GeneralPauli(n, d, vcat(xz, 0))
            @test measure!(tab, op; outcome=0, phase_policy=1) == 0
            _assert_duality(tab)
        end
    end

    @testset "Randomized measurement circuit preserves duality and outcomes" begin
        # The destabilizer update must agree with a plain StabilizerTableau on
        # every returned outcome while keeping <D_j, S_k> = delta_jk throughout.
        for d in (2, 3)
            n = 6
            rng = Random.MersenneTwister(20260909)
            dtab = DestabilizerTableau(d, n; state=:product, basis=:Z)
            stab = StabilizerTableau(d, n; state=:product, basis=:Z)

            for _ in 1:150
                q1 = rand(rng, 1:n)
                q2 = rand(rng, 1:n)
                op = if q1 == q2
                    SinglePauli(q1, rand(rng, 0:d-1), rand(rng, 0:d-1))
                else
                    DoublePauli(q1, rand(rng, 0:d-1), rand(rng, 0:d-1),
                                q2, rand(rng, 0:d-1), rand(rng, 0:d-1))
                end
                outcome = rand(rng, 0:d-1)

                got = measure!(dtab, op; outcome=outcome, phase_policy=1)
                want = measure!(stab, op; outcome=outcome, phase_policy=1)

                @test got == want
                @test dtab.m == stab.m
                _assert_duality(dtab)
            end
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
