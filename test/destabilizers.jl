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

# Preset states get their dual basis and x·z cache written directly instead of
# through the generic O(n^3) rebuild. Three properties pin that: duality
# itself; the cache checked against its definition rather than against itself;
# and equality with what the generic rebuild computes, so that changing the
# dual basis has to be a deliberate act rather than a silent one.
@testset "Preset dual basis and x·z cache" begin
    for d in (2, 3, 5), n in (1, 2, 3, 5)
        specs = Any[(:ghz, :Z), (:product, :X), (:product, :Y), (:product, :Z)]
        n >= 3 && push!(specs, (:product, [:X, :Y, :Z][mod1.(1:n, 3)]))

        for (state, basis) in specs
            fresh = DestabilizerTableau(d, n; state=state, basis=basis)
            after_reset = reset!(DestabilizerTableau(d, n; state=:mixed);
                                 state=state, basis=basis)

            for tab in (fresh, after_reset)
                @test tab.m == n
                _assert_duality(tab)

                for j in 1:tab.m
                    @test tab.xdotz_cache[j] ==
                          QuditClifford.dot_xz_col(tab.stab, n, j, d)
                end

                direct = copy(tab.destab)
                QuditClifford.rebuild_destabilizers!(tab)
                @test tab.destab == direct
            end
        end
    end
end

# The dual column is rescaled by the pivot when the stabilizer column is scaled
# by its inverse. Every other canonicalization test here happens to have
# pivot == 1 -- forced at d = 2, accidental at d = 3 -- which makes that
# rescaling an identity operation and leaves the site unguarded. This one uses
# Z_1^2, so the pivot is 2 and inv(2, 3) = 2.
@testset "Canonicalization rescales duals when the pivot is not 1" begin
    raw = zeros(Int, 5, 2)
    raw[3, 1] = 2      # S_1 = Z_1^2  -> pivot 2
    raw[3, 2] = 1      # S_2 = Z_1 Z_2^2
    raw[4, 2] = 2

    tab = DestabilizerTableau(3, raw; m=2, storephase=true)
    _assert_duality(tab)

    canonicalize!(tab)

    @test tab.iscanonical
    _assert_duality(tab)

    # The exact canonical form, not just "it is canonical and dual". S_1 = Z_1^2
    # normalizes to Z_1 and then clears Z_1 out of S_2, leaving Z_2^2 -> Z_2.
    # Phases stay 0 because both generators are Z-type, so every x.z term is 0.
    @test tab.stab[:, 1] == [0, 0, 1, 0, 0]
    @test tab.stab[:, 2] == [0, 0, 0, 1, 0]
    @test tab.destab[:, 1] == [1, 0, 0, 0]
    @test tab.destab[:, 2] == [0, 1, 0, 0]
    @test tab.pivcol_of_row == [0, 0, 1, 2]     # pivot metadata
    @test tab.xdotz_cache[1:2] == [0, 0]

    # RCEF structurally: each pivot is 1 and is alone in its row.
    for r in 1:(2 * tab.n)
        c = tab.pivcol_of_row[r]
        c == 0 && continue
        @test tab.stab[r, c] == 1
        for jj in 1:tab.m
            jj == c || @test tab.stab[r, jj] == 0
        end
    end

    # A StabilizerTableau from the same raw input canonicalizes identically.
    ref = StabilizerTableau(3, copy(raw); m=2, storephase=true)
    canonicalize!(ref)
    @test ref.stab == tab.stab
    @test ref.pivcol_of_row == tab.pivcol_of_row
end

# The commuting-append branch of measure! (case 3: commutes, not in span) with
# a projection that actually has work to do. Every other destabilizer circuit
# test starts from a pure state with m == n, where this branch cannot fire, and
# the two mixed-state tests that do reach it are degenerate: at d = 2 the sign
# of the projection does not matter and inv(beta) is always 1, and at n = 1,
# m = 0 both loops have zero iterations.
#
# S_1 = X_1 X_2^e commutes with X_1^a but does not span it, so measuring X_1^a
# appends. The dual constructed for the residual is Z_1, which does NOT commute
# with S_1 -- that is what makes the projection loop run -- and beta != 1 for
# a != 1, which is what makes the inv(beta) scaling observable.
@testset "Commuting append with a non-trivial projection at odd d" begin
    for e in (1, 2), a in (1, 2)
        n = 2
        raw = zeros(Int, 2n + 1, n)
        raw[1, 1] = 1
        raw[2, 1] = e                       # S_1 = X_1 X_2^e

        tab = DestabilizerTableau(3, raw; m=1, storephase=true)
        _assert_duality(tab)

        measure!(tab, SinglePauli(1, a, 0); outcome=0, phase_policy=1)

        @test tab.m == 2
        _assert_duality(tab)
    end

    # Two existing generators, so the projection and the re-orthogonalization
    # loops both run more than once.
    n = 3
    raw = zeros(Int, 2n + 1, n)
    raw[1, 1] = 1; raw[2, 1] = 1            # S_1 = X_1 X_2
    raw[2, 2] = 1; raw[3, 2] = 2            # S_2 = X_2 X_3^2

    tab = DestabilizerTableau(3, raw; m=2, storephase=true)
    _assert_duality(tab)

    measure!(tab, SinglePauli(1, 2, 0); outcome=0, phase_policy=1)

    @test tab.m == 3
    _assert_duality(tab)
end

# The two modular tiers that no other test in this file reaches. d = 5 accepts
# Barrett and d = 131 rejects it, so these run the shift-reduce and the
# divide-twice paths of src/modular.jl respectively -- through measure!, not
# through the kernel directly. The state invariants are what would break if a
# tier computed the wrong residue.
@testset "Measurement across the modular tiers" begin
    @test QuditClifford._barrett_valid(5)        # Barrett tier
    @test !QuditClifford._barrett_valid(131)     # divide-twice fallback

    # DoublePauli(1,1,0, 2,3,0) on a d=5 Z-basis product state anticommutes
    # with both Z_1 and Z_2, with commutators 4 and 2, so the second generator
    # is multiplied by the replaced one to the power
    # mod(-2 * inv(4, 5), 5) = 2 -- a genuine Barrett multiplier, not 0/1/d-1.
    cases = ((5, DoublePauli(1, 1, 0, 2, 3, 0), 2),
             (131, DoublePauli(1, 1, 0, 2, 3, 0), nothing))

    for (d, op, want_mult) in cases, TT in (StabilizerTableau, DestabilizerTableau)
        n = 3
        tab = TT(d, n; state=:product, basis=:Z)

        if want_mult !== nothing
            c1 = mod(QuditClifford.commutation_col(tab.stab, 1, op), d)
            c2 = mod(QuditClifford.commutation_col(tab.stab, 2, op), d)
            @test mod(-c2 * invmod(c1, d), d) == want_mult
        end

        out = measure!(tab, op; outcome=0, phase_policy=1)
        @test out == 0
        @test tab.m == n

        # generators stay mutually commuting
        for i in 1:tab.m, j in 1:tab.m
            @test mod(QuditClifford.commutation_col(tab.stab, i, tab.stab[:, j]), d) == 0
        end
        # inactive capacity stays zeroed
        @test all(tab.stab[:, (tab.m + 1):end] .== 0)

        if TT === DestabilizerTableau
            _assert_duality(tab)
            for j in 1:tab.m
                @test tab.xdotz_cache[j] == QuditClifford.dot_xz_col(tab.stab, n, j, d)
            end
            @test all(tab.xdotz_cache[(tab.m + 1):end] .== 0)
        end

        # canonicalization at the same prime agrees between the two types
        canonicalize!(tab)
        @test tab.iscanonical
        TT === DestabilizerTableau && _assert_duality(tab)
    end
end
