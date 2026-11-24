using QuditClifford
using Test
using Random
include(joinpath(@__DIR__, "env_switch.jl"))
isdefined(Main, :st_padding_zero) || include(joinpath(@__DIR__, "storage_helpers.jl"))

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
        val = mod(_symp(destabilizer_matrix(tab), i, stabilizer_matrix(tab), j, tab.n), tab.d)
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
                    val = mod(_symp(destabilizer_matrix(tab), i, stabilizer_matrix(tab), j, tab.n), d)
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
        val = mod(_symp(destabilizer_matrix(tab), 1, stabilizer_matrix(tab), 1, tab.n), tab.d)
        @test val == 1
    end

    @testset "Commuting append duality" begin
        tab = DestabilizerTableau(3, 1; state=:mixed)
        out = measure!(tab, SinglePauli(1, 0, 1); outcome=2) # measure Z
        @test out == 2
        @test tab.m == 1
        val = mod(_symp(destabilizer_matrix(tab), 1, stabilizer_matrix(tab), 1, tab.n), tab.d)
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
        @test stabilizer_matrix(tab)[1:4, 1] == Int[1, 1, 0, 0]
        @test stabilizer_matrix(tab)[1:4, 2] == Int[1, 0, 0, 0]
        @test destabilizer_matrix(tab)[:, 1] == Int[0, 0, 0, 1] # Z₂
        @test destabilizer_matrix(tab)[:, 2] == Int[0, 0, 1, 1] # Z₁Z₂

        for i in 1:tab.m, j in 1:tab.m
            val = mod(_symp(destabilizer_matrix(tab), i, stabilizer_matrix(tab), j, tab.n), tab.d)
            @test val == (i == j ? 1 : 0)
        end
    end

    @testset "Noncommuting replacement reorthogonalizes unaffected duals" begin
        # For S₁=X₁, S₂=X₂ and P=Z₁X₂, replacing S₁ makes the
        # old dual of S₂ fail to commute with P. The update must correct that dual.
        tab = DestabilizerTableau(2, 2; state=:product, basis=:X)
        op = DoublePauli(1, 0, 1, 2, 1, 0)

        @test measure!(tab, op; outcome=0) == 0
        @test stabilizer_matrix(tab)[1:4, 1] == Int[0, 1, 1, 0]
        @test stabilizer_matrix(tab)[1:4, 2] == Int[0, 1, 0, 0]
        @test destabilizer_matrix(tab)[:, 1] == Int[1, 0, 0, 0] # X₁
        @test destabilizer_matrix(tab)[:, 2] == Int[1, 0, 0, 1] # X₁Z₂

        for i in 1:tab.m, j in 1:tab.m
            val = mod(_symp(destabilizer_matrix(tab), i, stabilizer_matrix(tab), j, tab.n), tab.d)
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
            d2_before = copy(destabilizer_matrix(tab)[:, 2])
            d3_before = copy(destabilizer_matrix(tab)[:, 3])

            @test measure!(tab, TriplePauli(1, 0, 1, 2, 1, 0, 3, 1, 0);
                           outcome=0, phase_policy=1) == 0

            @test destabilizer_matrix(tab)[:, 2] != d2_before   # both duals really moved
            @test destabilizer_matrix(tab)[:, 3] != d3_before
            _assert_duality(tab)
        end
    end

    @testset "Dephasing preserves duality" begin
        rng = Xoshiro(2024)
        n, d = 6, 3
        tab = DestabilizerTableau(d, n)
        random_state!(rng, tab; m=4)
        for _ in 1:30
            q = randperm(rng, n)
            op = DoublePauli(q[1], rand(rng, 0:d-1), rand(rng, 0:d-1),
                             q[2], rand(rng, 0:d-1), rand(rng, 0:d-1))
            dephase!(tab, op)
            _assert_duality(tab)
            q = rand(rng, 1:n)
            measure!(tab, SinglePauli(q, rand(rng, 0:d-1), rand(rng, 0:d-1)); outcome=rand(rng, 0:d-1))
            _assert_duality(tab)
        end
    end

    @testset "Trace-out preserves duality" begin
        rng = Xoshiro(2025)
        for d in (2, 3), m in (3, 5), _ in 1:5
            n = 5
            tab = DestabilizerTableau(d, n)
            random_state!(rng, tab; m)
            traceout!(tab, randperm(rng, n)[1:rand(rng, 1:n)])
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
                d1_before = copy(destabilizer_matrix(tab)[:, 1])

                @test measure!(tab, op; outcome=0, phase_policy=1) == 0

                @test destabilizer_matrix(tab)[:, 1] != d1_before   # the dual really moved
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
                    val = mod(_symp(destabilizer_matrix(tab), i, stabilizer_matrix(tab), j, tab.n), d)
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
        @test mod(_symp(destabilizer_matrix(tab), 1, stabilizer_matrix(tab), 1, tab.n), tab.d) == 1
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
        @test is_pure(tab; verify=true)
        @test expect_int!(tab, op) == expected

        for i in 1:tab.m
            for j in 1:tab.m
                val = mod(_symp(destabilizer_matrix(tab), i, stabilizer_matrix(tab), j, tab.n), d)
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
                          QuditClifford.dot_xz_col(QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), j, d)
                end

                direct = copy(destabilizer_matrix(tab))
                QuditClifford.rebuild_destabilizers!(tab, QuditClifford._layout_marker(tab))
                @test destabilizer_matrix(tab) == direct
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
    @test stabilizer_matrix(tab)[:, 1] == [0, 0, 1, 0, 0]
    @test stabilizer_matrix(tab)[:, 2] == [0, 0, 0, 1, 0]
    @test destabilizer_matrix(tab)[:, 1] == [1, 0, 0, 0]
    @test destabilizer_matrix(tab)[:, 2] == [0, 1, 0, 0]
    @test tab.pivcol_of_row == [0, 0, 1, 2]     # pivot metadata
    @test tab.xdotz_cache[1:2] == [0, 0]

    # RCEF structurally: each pivot is 1 and is alone in its row.
    for r in 1:(2 * tab.n)
        c = tab.pivcol_of_row[r]
        c == 0 && continue
        @test stabilizer_matrix(tab)[r, c] == 1
        for jj in 1:tab.m
            jj == c || @test stabilizer_matrix(tab)[r, jj] == 0
        end
    end

    # A StabilizerTableau from the same raw input canonicalizes identically.
    ref = StabilizerTableau(3, copy(raw); m=2, storephase=true)
    canonicalize!(ref)
    @test stabilizer_matrix(ref) == stabilizer_matrix(tab)
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
            c1 = mod(QuditClifford.commutation_col(QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), 1, op), d)
            c2 = mod(QuditClifford.commutation_col(QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), 2, op), d)
            @test mod(-c2 * invmod(c1, d), d) == want_mult
        end

        out = measure!(tab, op; outcome=0, phase_policy=1)
        @test out == 0
        @test tab.m == n

        # generators stay mutually commuting
        for i in 1:tab.m, j in 1:tab.m
            @test mod(QuditClifford.commutation_col(QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), i, stabilizer_matrix(tab)[:, j]), d) == 0
        end
        # inactive capacity stays zeroed
        @test all(stabilizer_matrix(tab)[:, (tab.m + 1):end] .== 0)

        if TT === DestabilizerTableau
            _assert_duality(tab)
            for j in 1:tab.m
                @test tab.xdotz_cache[j] == QuditClifford.dot_xz_col(QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), j, d)
            end
            @test all(tab.xdotz_cache[(tab.m + 1):end] .== 0)
        end

        # canonicalization at the same prime agrees between the two types
        canonicalize!(tab)
        @test tab.iscanonical
        TT === DestabilizerTableau && _assert_duality(tab)
    end
end

@testset "DestabilizerTableau carries no rebuild-only workspaces" begin
    # A DestabilizerTableau is a StabilizerTableau plus its dual basis, which is
    # one 2n x n Int matrix, so the ratio belongs near 1.5. Anything much above
    # that means per-tableau scratch has crept back in -- `rebuild_destabilizers!`
    # needs five n^2-sized matrices and runs at most once per tableau, so they
    # belong to the call and not to the struct.
    for n in (16, 64, 256)
        stab_bytes = Base.summarysize(StabilizerTableau(3, n; state = :product))
        destab_bytes = Base.summarysize(DestabilizerTableau(3, n; state = :product))
        @test destab_bytes < 1.75 * stab_bytes
    end

    # The rebuild owns that scratch, so assert it still produces a dual basis.
    for d in (2, 3), n in (2, 5)
        raw = zeros(Int, 2n + 1, n)
        for j in 1:n
            raw[n+j, j] = 1          # Z_j: commuting, independent
        end
        tab = DestabilizerTableau(d, raw; m = n, storephase = true)
        _assert_duality(tab)
    end
end

@testset "Duality and cache after Clifford gates" begin
    n = 3
    for d in (2, 3, 5), storephase in (true, false), kind in (:ghz, :mixed)
        tab = if kind === :ghz
            DestabilizerTableau(d, n; state = :ghz, storephase = storephase)
        else
            t = DestabilizerTableau(d, n; state = :mixed, storephase = storephase)
            measure!(t, SinglePauli(1, 0, 1); outcome = 0)
            t
        end
        # Phase(2) and CPhase leave a nonzero target x·z behind, which is what
        # makes the cache delta non-trivial.
        for g in (Fourier(1), Phase(2), SUM(1, 3, 1), SWAP(1, 3), CPhase(1, 2, 1))
            apply!(tab, g)

            for a in 1:tab.m, b in 1:tab.m
                s = sum(destabilizer_matrix(tab)[q, a] * stabilizer_matrix(tab)[n + q, b] -
                        destabilizer_matrix(tab)[n + q, a] * stabilizer_matrix(tab)[q, b] for q in 1:n)
                @test mod(s, d) == (a == b ? 1 : 0)
            end

            for j in 1:tab.m
                @test tab.xdotz_cache[j] ==
                      QuditClifford.dot_xz_col(QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), j, d)
            end
            @test all(tab.xdotz_cache[(tab.m + 1):n] .== 0)
            @test all(destabilizer_matrix(tab)[:, (tab.m + 1):n] .== 0)
        end

        # A deterministic odd-prime query that actually consumes the cache:
        # measuring a generator of the state is in-span and must not move it.
        if d != 2 && storephase && tab.m > 0
            x = [stabilizer_matrix(tab)[q, 1] for q in 1:n]
            z = [stabilizer_matrix(tab)[n + q, 1] for q in 1:n]
            op = GeneralPauli(vcat(x, z), 0)
            before = copy(stabilizer_matrix(tab))
            @test measure!(tab, op) == mod(-stabilizer_matrix(tab)[2n + 1, 1], d)
            @test stabilizer_matrix(tab) == before
        end
    end
end

@testset "Stored Clifford application preserves duality and the live cache" begin
    for d in (2, 3, 5), sp in (false, true), mixed in (false, true)
        tab = DestabilizerTableau(d, 6; state = mixed ? :mixed : :product,
                                 basis = mixed ? :Z : :X, storephase = sp)
        if mixed
            measure!(tab, SinglePauli(2, 1, 0); outcome = 0)
            measure!(tab, SinglePauli(5, 1, 0); outcome = 0)
        end
        # Start with nonzero old target x·z, so subtract-before-scatter matters.
        apply!(tab, Phase(2))
        @test any(!iszero, tab.xdotz_cache[1:tab.m])
        for g in (Fourier(2), Phase(2), SUM(2, 5, 1), CPhase(2, 5, 1), SWAP(2, 5))
            oldm = tab.m
            apply!(tab, CliffordOperator(g, d))
            @test tab.m == oldm
            _assert_duality(tab)
            for j in 1:tab.m
                expected = Int(mod(sum(big(stabilizer_matrix(tab)[q, j]) * stabilizer_matrix(tab)[tab.n + q, j]
                                       for q in 1:tab.n), d))
                @test tab.xdotz_cache[j] == expected
            end
            @test all(iszero, stabilizer_matrix(tab)[:, (tab.m + 1):end])
            @test all(iszero, destabilizer_matrix(tab)[:, (tab.m + 1):end])
            @test all(iszero, tab.xdotz_cache[(tab.m + 1):end])
            if sp && d > 2
                # Immediate deterministic query consumes the live cache.
                op = GeneralPauli(copy(stabilizer_matrix(tab)[1:(2tab.n), 1]), stabilizer_matrix(tab)[2tab.n + 1, 1])
                @test expect_int!(tab, op) == 0
            end
        end
    end
end

# x·z of stabilizer column j, the value `xdotz_cache[j]` must hold.
_stab_xdotz(tab, j) =
    mod(sum(stabilizer_matrix(tab)[q, j] * stabilizer_matrix(tab)[tab.n + q, j] for q in 1:tab.n; init = 0), tab.d)

function _assert_active_cache(tab)
    for j in 1:tab.m
        @test tab.xdotz_cache[j] == _stab_xdotz(tab, j)
    end
end

# A Pauli with exponents `xz`, Hermitian at d = 2 through the phase x·z mod 2.
function _hermitian_pauli(xz::AbstractVector{<:Integer}, d::Int)
    n = length(xz) ÷ 2
    phase = d == 2 ? mod(sum(xz[q] * xz[n + q] for q in 1:n; init = 0), 2) : 0
    return GeneralPauli(collect(Int, xz), phase)
end

@testset "Random states stay dual through apply! and every measure! branch" begin
    rng = Xoshiro(20260925)
    for d in (2, 3), n in (3, 6), m in unique((0, 1, n ÷ 2, n))
        tab = DestabilizerTableau(d, n)
        random_state!(rng, tab; m)
        _assert_duality(tab)
        @test tab.xdotz_cache == [j <= m ? _stab_xdotz(tab, j) : 0 for j in 1:n]
        apply!(tab, random_clifford(rng, d, 1:n))
        _assert_duality(tab)
        _assert_active_cache(tab)
        if m > 0
            # In span: a generator's own exponents give a deterministic outcome.
            measure!(tab, _hermitian_pauli(stabilizer_matrix(tab)[1:(2n), 1], d))
            @test tab.m == m
            _assert_duality(tab)
            _assert_active_cache(tab)
            # Noncommuting: a dual anticommutes with its own stabilizer.
            measure!(tab, _hermitian_pauli(destabilizer_matrix(tab)[:, 1], d);
                     outcome = rand(rng, 0:(d - 1)))
            @test tab.m == m
            _assert_duality(tab)
            _assert_active_cache(tab)
        end
        if m < n
            # Commuting and independent: remove a random vector's components
            # along the duals, so it commutes with every generator, and draw
            # again until it lies outside their span.
            op = _hermitian_pauli(zeros(Int, 2n), d)
            while true
                x = rand(rng, 0:(d - 1), 2n, 1)
                v = vec(x)
                for j in 1:m
                    c = mod(_symp(x, 1, stabilizer_matrix(tab), j, n), d)
                    v = mod.(v .- c .* destabilizer_matrix(tab)[:, j], d)
                end
                op = _hermitian_pauli(v, d)
                expect_int!(tab, op) == -1 && break
            end
            measure!(tab, op; outcome = rand(rng, 0:(d - 1)))
            @test tab.m == m + 1
            _assert_duality(tab)
            _assert_active_cache(tab)
        end
    end
end

@testset "Reset preserves duality" begin
    for d in (2, 3)
        rng = Xoshiro(2100 + d)
        for _ in 1:20
            n = rand(rng, 1:5)
            tab = DestabilizerTableau(d, n)
            random_state!(rng, tab; m=rand(rng, 0:n))
            sites = sort(randperm(rng, n)[1:rand(rng, 1:n)])
            bases = [rand(rng, (:X, :Y, :Z)) for _ in sites]
            reset!(tab, sites; basis=bases)
            _assert_duality(tab)
            _assert_active_cache(tab)
        end
    end
end

@testset "Stored one- to three-qudit operators keep duality and the cache" begin
    rng = Xoshiro(20261118)
    for d in (2, 3, 5), n in (4, 7), m in unique((1, n ÷ 2, n)), k in 1:3
        tab = DestabilizerTableau(d, n)
        random_state!(rng, tab; m)
        for _ in 1:4
            apply!(tab, random_clifford(rng, d, randperm(rng, n)[1:k]))
            _assert_duality(tab)
            _assert_active_cache(tab)
            @test all(iszero, tab.xdotz_cache[(tab.m + 1):end])
        end
    end
end

# The noncommuting update folds the pivot into each anticommuting generator on
# the pivot's sites only when that support is small (a local state), and over
# all rows otherwise; it patches `xdotz_cache` instead of recomputing it, and
# restricts the dual re-orthogonalization to the same sites. Monitored-circuit
# states reach the first path and random states the second, so both run here
# for every Pauli type, with duality and the cache checked after each step.
@testset "Monitored-circuit measurements keep duality and the cache" begin
    for d in (2, 3, 5), n in (12, 33), flavor in (:local, :random)
        rng = Xoshiro(20261119 + d + n)
        tab = DestabilizerTableau(d, n; state = :Z, storephase = isodd(n) || d != 5)
        if flavor === :random
            random_state!(rng, tab)
        end
        for layer in 1:6
            for q in (isodd(layer) ? 1 : 2):2:(n - 1)
                apply!(tab, random_clifford(rng, d, [q, q + 1]))
            end
            for q in 1:n
                rand(rng) < 0.3 || continue
                r = q + 1 <= n ? q + 1 : 1
                op = rand(rng, (SinglePauli(q, 0, 1), SinglePauli(q, 1, 0),
                                DoublePauli(q, 0, 1, r, 0, 1), DoublePauli(q, 1, 0, r, 1, 0),
                                TriplePauli(q, 1, 1, r, 0, 1, mod1(q + 2, n), 1, 0),
                                NPauli((q, r, mod1(q + 2, n), mod1(q + 3, n)),
                                       (1, 0, 1, 2), (0, 1, 1, 1)),
                                _hermitian_pauli([i in (q, n + r) ? 1 : 0 for i in 1:(2n)], d)))
                measure!(rng, tab, op; phase_policy = 1)
                @test tab.m == n
                _assert_duality(tab)
                _assert_active_cache(tab)
            end
        end
    end
end

# Channels rebuild generators through the rank-workspace copy, the column
# writers and the generator drop: duality and the live x·z cache must survive.
@testset "duality after channels" begin
    for d in (2, 3)
        n = 6
        rng = Xoshiro(20261120 + d)
        tab = DestabilizerTableau(d, n; state = :Z, storephase = true)
        for layer in 1:4
            for q in (isodd(layer) ? 1 : 2):2:(n - 1)
                apply!(tab, random_clifford(rng, d, [q, q + 1]))
            end
            for q in 1:n
                rand(rng) < 0.4 && measure!(rng, tab, SinglePauli(q, 0, 1); phase_policy = 1)
            end
        end
        _assert_duality(tab)
        _assert_active_cache(tab)

        reset!(tab, [2, 5])
        _assert_duality(tab)
        _assert_active_cache(tab)

        traceout!(tab, 3)
        _assert_duality(tab)
        _assert_active_cache(tab)

        dephase!(tab, SinglePauli(4, 0, 1))
        _assert_duality(tab)
        _assert_active_cache(tab)
    end
end

# Gates and measurements run generator-major; after each one the tableau is
# switched to qudit-major, checked through the layout-independent matrices,
# and switched back.
@testset "duality after every switch" begin
    for (d, n, T) in ((2, 7, UInt8), (3, 7, UInt8), (2, 7, Bits), (2, 70, UInt8), (2, 70, Bits))
        rng = Xoshiro(20261121 + d + n)
        # The tableau under test and a `UInt8` one take the same operations,
        # each with its own copy of the measurement RNG.
        tab = DestabilizerTableau(d, n; state = :Z, eltype = T)
        ref = DestabilizerTableau(d, n; state = :Z, eltype = UInt8)
        rt, rr = Xoshiro(d + n), Xoshiro(d + n)
        for step in 1:10
            if isodd(step)
                q = rand(rng, 1:(n - 1))
                C = random_clifford(rng, d, [q, q + 1])
                apply!(tab, C)
                apply!(ref, C)
            else
                q = rand(rng, 1:n)
                x = rand(rng, 0:(d - 1))
                z = x == 0 ? rand(rng, 1:(d - 1)) : rand(rng, 0:(d - 1))
                P = SinglePauli(q, x, z, d == 2 ? x * z : 0)
                @test measure!(rt, tab, P) == measure!(rr, ref, P)
            end
            @test stabilizer_matrix(tab) == stabilizer_matrix(ref)
            @test destabilizer_matrix(tab) == destabilizer_matrix(ref)
            @test tableau_eltype(tab) === T
            qudit_major!(tab)
            @test tableau_layout(tab) === :qudit
            S, D = stabilizer_matrix(tab), destabilizer_matrix(tab)
            m = tab.m
            dual = [mod(_symp(D, i, S, j, n), d) for i in 1:m, j in 1:m]
            @test dual == [Int(i == j) for i in 1:m, j in 1:m]
            @test tab.xdotz_cache[1:m] ==
                  [mod(sum(S[q, j] * S[n + q, j] for q in 1:n), d) for j in 1:m]
            generator_major!(tab)
            @test tableau_layout(tab) === :generator
        end
    end
end

# The packed measurement kernels and their destabilizer hooks, in both
# layouts: the append branch from a mixed start, the random branch with sparse
# and dense operators, and `dephase!`, each checked against duality and the
# cache, and against a `UInt8` tableau taking the same operations.
@testset "packed duality through measure! and dephase!" begin
    n = 70
    for layout in (:generator, :qudit)
        rng = Xoshiro(20261122)
        tab = DestabilizerTableau(2, n; state = :mixed, eltype = Bits, layout)
        ref = DestabilizerTableau(2, n; state = :mixed, eltype = UInt8, layout)
        rt, rr = Xoshiro(5), Xoshiro(5)
        ok = true
        for step in 1:200
            r = rand(rng)
            q, s = randperm(rng, n)[1:2]
            if r < 0.1
                C = random_clifford(rng, 2, [q, s])
                apply!(tab, C)
                apply!(ref, C)
            elseif r < 0.2
                P = DoublePauli(q, rand(rng, 0:1), 1, s, 1, 0)
                dephase!(tab, P)
                dephase!(ref, P)
            else
                P = r < 0.8 ? SinglePauli(q, rand(rng, 0:1), 1) :
                    _hermitian_pauli(rand(rng, 0:1, 2n), 2)
                ok &= measure!(rt, tab, P; phase_policy = 1) == measure!(rr, ref, P; phase_policy = 1)
            end
            S, D = stabilizer_matrix(tab), destabilizer_matrix(tab)
            m = tab.m
            ok &= S == stabilizer_matrix(ref) && D == destabilizer_matrix(ref) && m == ref.m
            ok &= [mod(_symp(D, i, S, j, n), 2) for i in 1:m, j in 1:m] ==
                  [Int(i == j) for i in 1:m, j in 1:m]
            ok &= tab.xdotz_cache == [j <= m ? mod(sum(S[k, j] * S[n + k, j] for k in 1:n), 2) : 0
                                      for j in 1:n]
            ok || (@error "packed duality" layout step; break)
        end
        @test ok
    end
end

# The duals of a raw-matrix `DestabilizerTableau` against an independent plain
# `Int` elimination: `A = Sᵀ J`, its pivot columns from forward elimination
# (the first `m` linearly independent columns), and `D` holding the inverse of
# `A` on those columns at the pivot rows. Each storage type and layout builds
# exactly these duals (the GF(2) path at d = 2, the transposed elimination
# mod d above), and they are dual to the generators.
function _ref_raw_duals(S::Matrix{Int}, d::Int, n::Int, m::Int)
    A = zeros(Int, m, 2n)
    for k in 1:m, q in 1:n
        A[k, q] = mod(S[n + q, k], d)
        A[k, n + q] = mod(-S[q, k], d)
    end
    W = copy(A)
    piv = Int[]
    r = 1
    for c in 1:2n
        r > m && break
        p = findfirst(i -> W[i, c] != 0, r:m)
        p === nothing && continue
        p += r - 1
        W[r, :], W[p, :] = W[p, :], W[r, :]
        W[r, :] = mod.(W[r, :] .* invmod(W[r, c], d), d)
        for rr in (r + 1):m
            W[rr, :] = mod.(W[rr, :] .- W[rr, c] .* W[r, :], d)
        end
        push!(piv, c)
        r += 1
    end
    D = zeros(Int, 2n, n)
    length(piv) == m || return D
    M = hcat(A[:, piv], _ref_identity(m))
    for c in 1:m
        p = findfirst(i -> M[i, c] != 0, c:m) + c - 1
        M[c, :], M[p, :] = M[p, :], M[c, :]
        M[c, :] = mod.(M[c, :] .* invmod(M[c, c], d), d)
        for rr in 1:m
            rr == c && continue
            M[rr, :] = mod.(M[rr, :] .- M[rr, c] .* M[c, :], d)
        end
    end
    for i in 1:m, j in 1:m
        D[piv[i], j] = M[i, m + j]
    end
    return D
end
_ref_identity(m) = [i == j ? 1 : 0 for i in 1:m, j in 1:m]

@testset "Raw-matrix duals match a reference elimination" begin
    for d in (2, 3, 5), n in (1, 5, 64, 65, 200)
        Es = d == 2 ? (UInt8, UInt16, Int, Bits) : (UInt8, UInt16, Int)
        for m in unique((1, n ÷ 2 + 1, n - 1, n))
            m >= 1 || continue
            r = StabilizerTableau(d, n; eltype = Int, layout = :generator)
            random_state!(Xoshiro(1000d + 10n + m), r; m)
            S = stabilizer_matrix(r)[:, 1:m]
            ref = _ref_raw_duals(S, d, n, m)
            for E in Es, layout in (:generator, :qudit)
                tab = DestabilizerTableau(d, S; eltype = E, layout)
                @test destabilizer_matrix(tab)[:, 1:m] == ref[:, 1:m]
                n <= 65 && _assert_duality(tab)
            end
        end
    end
    # Dependent generators (check = false) leave the duals zero.
    for d in (2, 3), E in (d == 2 ? (UInt8, Bits) : (UInt8,)), layout in (:generator, :qudit)
        S = zeros(Int, 13, 3); S[1, 1] = 1; S[1, 2] = 1; S[8, 3] = 1
        @test iszero(destabilizer_matrix(DestabilizerTableau(d, S; check = false, eltype = E, layout)))
    end
end

# The packed GHZ duals are written a word at a time; they equal the unpacked
# ones and keep every padding bit zero, through the constructor and `reset!`.
@testset "Packed GHZ duals by words" begin
    for n in (1, 2, 63, 64, 65, 127, 128, 130), layout in (:generator, :qudit)
        u = DestabilizerTableau(2, n; state = :ghz, eltype = UInt8, layout)
        t = DestabilizerTableau(2, n; state = :ghz, eltype = Bits, layout)
        @test destabilizer_matrix(t) == destabilizer_matrix(u)
        @test st_padding_zero(t)
        random_state!(Xoshiro(n), t)
        reset!(t; state = :ghz)
        @test destabilizer_matrix(t) == destabilizer_matrix(u)
        @test stabilizer_matrix(t) == stabilizer_matrix(u)
        @test st_padding_zero(t)
        _assert_duality(t)
    end
end
