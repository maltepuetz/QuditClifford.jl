using Test
include(joinpath(@__DIR__, "workloads.jl"))

@testset "Workloads" begin
    @testset "Purification is deterministic and reaches full rank" begin
        h1, l1 = purification_trajectory(d = 2, n = 8, seed = 42)
        h2, l2 = purification_trajectory(d = 2, n = 8, seed = 42)
        @test h1 == h2
        @test l1 == l2
        @test l1 <= 10_000              # purified before the cutoff
        @test h1[end] == 8              # full rank
        @test issorted(h1)              # rank never decreases

        h3, _ = purification_trajectory(d = 2, n = 8, seed = 43)
        @test h1 != h3                  # the seed actually matters
    end

    @testset "Ising is deterministic and stays pure" begin
        s1 = ising_trajectory!(L = 8, depth = 16, seed = 7)
        s2 = ising_trajectory!(L = 8, depth = 16, seed = 7)
        @test s1.stab == s2.stab
        @test s1.m == s2.m
        # Started from a pure product state; projective measurement keeps it pure.
        @test is_pure(s1; verify=true)

        s3 = ising_trajectory!(L = 8, depth = 16, seed = 8)
        @test s1.stab != s3.stab         # the seed actually matters

        # Exercise the default `depth = 8L` keyword path (every call above sets
        # depth explicitly) and pin its value: omitting depth must match passing
        # depth = 8*8 explicitly, for the same seed.
        s4 = ising_trajectory!(L = 8, seed = 7)
        s5 = ising_trajectory!(L = 8, depth = 8 * 8, seed = 7)
        @test s4.stab == s5.stab
        @test is_pure(s4; verify=true)
    end

    @testset "Micro group builds for both tableau types" begin
        for T in (StabilizerTableau, DestabilizerTableau)
            g = micro_group(d = 3, n = 8, T = T)
            @test length(keys(g)) == 8
        end
    end

    @testset "Snapshot/restore round-trips" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            tab = T(d, 16; state = :ghz)
            snap = snapshot(tab)
            stab0, m0 = copy(tab.stab), tab.m

            # A mutating op, then restore: the tableau must be bit-identical.
            measure!(tab, SinglePauli(1, 1, 0); outcome = 0, phase_policy = 2)
            canonicalize!(tab)
            @test tab.stab != stab0          # the ops really did change it
            restore!(snap)
            @test tab.stab == stab0
            @test tab.m == m0
            @test tab.iscanonical == false
            T === DestabilizerTableau && @test snap.destab !== nothing
        end
    end

    @testset "canonicalize! benchmark measures the real path" begin
        # If the restored tableau were already canonical, canonicalize! would
        # hit every `β == 0 && continue` guard and collapse to O(n*m) -- the
        # benchmark would silently measure the wrong thing, at ~1/100 the cost.
        for T in (StabilizerTableau, DestabilizerTableau)
            tab = T(3, 16; state = :ghz)
            snap = snapshot(tab)
            before = copy(tab.stab)

            canonicalize!(tab)
            @test tab.stab != before         # :ghz is genuinely non-canonical
            @test tab.iscanonical

            restore!(snap)
            @test tab.stab == before         # back to the non-canonical data
            @test !tab.iscanonical

            # ...so a second canonicalize! does the full job again, which is
            # the property the benchmark depends on sample after sample.
            canonicalize!(tab)
            @test tab.stab != before
        end

        # A :product tableau is already in RCEF -- the state the benchmark
        # must NOT use. Pinned so a future "simplification" to :product fails.
        tab = DestabilizerTableau(3, 16; state = :product)
        before = copy(tab.stab)
        canonicalize!(tab)
        @test tab.stab == before
    end

    @testset "Measurements stay quiet at d = 2" begin
        # phase_policy = 2 everywhere: the default policy @warns on every
        # non-Hermitian qubit Pauli, which a random circuit produces constantly.
        @test_logs ising_trajectory!(L = 6, depth = 8, seed = 3)
        @test_logs purification_trajectory(d = 2, n = 6, seed = 3)
    end

    @testset "Spine operators hit the branches they claim" begin
        # Each spine leaf is named for a measure! branch. If the operator stops
        # selecting that branch the leaf silently measures something else, and
        # nothing else in the suite would notice.
        for n in (8, 64, 256), d in (2, 3, 5)
            i, j = spread_sites(n)
            @test 1 <= i < j <= n
            @test length(unique(spread_eight(n))) == 8
            @test all(1 .<= spread_eight(n) .<= n)

            tab = DestabilizerTableau(d, n; state = :product)

            # noncommuting: X on a Z-basis site anticommutes with its stabilizer
            @test any(mod(QuditClifford.commutation_col(tab.stab, k, spread_x(n)), d) != 0
                      for k in 1:tab.m)
            # deterministic: Z*Z is a product of two generators, so in span
            @test expect_int!(tab, spread_z(n)) != -1
            # ...and spread_x is not, which is what expect!/out_of_span needs
            @test expect_int!(tab, spread_x(n)) == -1

            # Hermitian at d = 2, so phase_policy cannot matter for these two
            if d == 2
                @test_logs measure!(DestabilizerTableau(2, n; state = :product),
                                    spread_x(n); outcome = 0)
            end
        end
    end

    @testset "Mid-circuit state is scrambled, pure and deterministic" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5)
            a = scrambled_state(T, d, 16; seed = 11)
            b = scrambled_state(T, d, 16; seed = 11)
            @test a.stab == b.stab                     # deterministic
            @test a.stab != T(d, 16; state = :product).stab   # actually scrambled
            @test a.m == 16                            # stayed pure, so measure! cannot
            @test is_pure(a; verify=true)              # hit the m == n append error
            @test scrambled_state(T, d, 16; seed = 12).stab != a.stab
        end
    end

    @testset "Mid-circuit operators hit the branches they claim" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            tab = scrambled_state(T, d, 16; seed = 11)

            # in_span_operator is one of the generators, so expect! must find
            # it (-1 means <P> = 0, i.e. not in the stabilizer group).
            @test expect_int!(tab, in_span_operator(tab)) != -1

            # first_anticommuting must genuinely anticommute, or measure! takes
            # the commuting branch and throws once m == n.
            anti = first_anticommuting(tab, [SinglePauli(i, 1, 0) for i in 1:16])
            @test any(mod(QuditClifford.commutation_col(tab.stab, j, anti), d) != 0
                      for j in 1:tab.m)
            @test_throws ErrorException first_anticommuting(tab, SinglePauli[])
        end
    end

    @testset "entropy benchmark measures a non-canonical tableau" begin
        # entanglement_entropy never canonicalizes, so most of its cost is the
        # elimination in rank_fp_cols! -- and how much there is to eliminate
        # depends entirely on how close the generators already are to echelon
        # form. The leaf therefore has to hold a dirty tableau, and has to stay
        # dirty sample after sample.
        for T in (StabilizerTableau, DestabilizerTableau)
            tab = scrambled_state(T, 3, 16; seed = 1)
            sub = collect(1:8)
            @test !tab.iscanonical

            S = entanglement_entropy(tab, sub)
            @test !tab.iscanonical           # ...and every later sample sees the same state

            before = copy(tab.stab)
            canonicalize!(tab)
            @test tab.stab != before         # the data really was non-canonical
            @test entanglement_entropy(tab, sub) == S   # same answer either way
        end
    end

    @testset "Fallback-prime probe actually reaches the fallback tier" begin
        # The probe exists to cover the divide-twice fallback in src/modular.jl.
        # Choosing a prime the Barrett guard rejects is necessary but NOT
        # sufficient: a :ghz tableau holds only 0, 1 and d-1, so every
        # elimination multiplier is d-1 and it takes the multiply-free tier at
        # any prime -- so a probe built from a constructor would measure
        # nothing it claims to, however the prime is chosen. Assert the tier
        # directly rather than inferring it from the prime.
        d = FALLBACK_PRIME
        @test !QuditClifford._barrett_valid(d)

        # Replay the pivot scan of _canonicalize_tableau! to collect the
        # elimination multipliers. Only the scan is duplicated; the replay is
        # then checked against the real canonicalize! below, so it cannot drift
        # away from the implementation without this test failing.
        function elimination_multipliers(tab)
            A = copy(tab.stab)[1:(2 * tab.n), :]
            d, n, m = tab.d, tab.n, tab.m
            mults = Int[]
            r, c = 1, 1
            while r <= 2n && c <= m
                j = c
                while j <= m && A[r, j] == 0
                    j += 1
                end
                if j > m
                    r += 1
                    continue
                end
                j != c && (A[:, [c, j]] = A[:, [j, c]])
                α = invmod(A[r, c], d)
                for i in 1:(2n)
                    A[i, c] = mod(A[i, c] * α, d)
                end
                for jj in 1:m
                    jj == c && continue
                    β = A[r, jj]
                    β == 0 && continue
                    push!(mults, β)
                    for i in 1:(2n)
                        A[i, jj] = mod(A[i, jj] - β * A[i, c], d)
                    end
                end
                r += 1
                c += 1
            end
            return mults, A
        end

        for nn in (16, 64)
            tab = fallback_probe_state(nn; d = d)   # the exact probe fixture
            mults, A = elimination_multipliers(tab)

            # the replay is faithful: same XZ block as the real thing
            ref = deepcopy(tab)
            canonicalize!(ref)
            @test A[:, 1:ref.m] == ref.stab[1:(2 * ref.n), 1:ref.m]

            # and it reaches the general tier -- not just 0, 1, d-1
            general = count(β -> β ∉ (0, 1, d - 1), mults)
            @test !isempty(mults)
            @test general > 0

            # the contrast that makes the point: :ghz never does, at any prime
            ghz_mults, _ = elimination_multipliers(DestabilizerTableau(d, nn; state = :ghz))
            @test !isempty(ghz_mults)
            @test count(β -> β ∉ (0, 1, d - 1), ghz_mults) == 0
        end
    end

    @testset "Mid-circuit group builds and runs" begin
        for T in (StabilizerTableau, DestabilizerTableau)
            g = midcircuit_group(d = 3, n = 16, T = T)
            @test length(keys(g)) == 6
            # Actually execute every leaf: @benchmarkable bodies are quoted, so
            # a broken one is invisible until something runs it. This is also
            # what proves the two measure! leaves do not throw.
            run(g; samples = 1, evals = 1, seconds = 0.05)
        end
    end

    # A DestabilizerTableau is a StabilizerTableau plus its dual basis, one
    # 2n x n Int matrix. Anything much above 1.5x means per-tableau scratch has
    # crept back in.
    @testset "Destabilizer memory is 1.5x stabilizer" begin
        for n in (128, 512)
            rs = tableau_bytes(StabilizerTableau(2, n; state = :product))
            rd = tableau_bytes(DestabilizerTableau(2, n; state = :product))
            @test 1.35 < rd / rs < 1.65
        end
    end

    @testset "Clifford registration preserves an older baseline" begin
        suite = BenchmarkGroup()
        existing = BenchmarkGroup()
        suite["existing"] = existing
        legacy = Module(:CliffordAPINotYetPresent)
        @test register_clifford_group!(suite, (StabilizerTableau,), (3,), (8,); api = legacy) === suite
        @test Set(keys(suite)) == Set(["existing"])
        @test suite["existing"] === existing

        # In this checkout the new APIs exist, so normal registration adds the
        # group alongside the old one, rather than replacing the whole suite.
        @test register_clifford_group!(suite, (StabilizerTableau,), (3,), (8,)) === suite
        @test haskey(suite, "clifford")
        @test suite["existing"] === existing
    end

    @testset "Clifford group builds, runs, and its gates act" begin
        for T in (StabilizerTableau, DestabilizerTableau)
            g = clifford_group(d = 3, n = 16, T = T)
            @test length(keys(g)) == 3
            # @benchmarkable bodies are quoted, so a broken leaf is invisible
            # until something runs it.
            run(g; samples = 1, evals = 1, seconds = 0.05)
        end

        # A microbenchmark that times a no-op would look like a speedup. Check
        # the benchmarked gates genuinely act and preserve physical purity.
        # Duality and live caches are checked explicitly in test/destabilizers.jl.
        for T in (StabilizerTableau, DestabilizerTableau)
            tab = T(3, 16; state = :ghz)
            before = copy(tab.stab)
            apply!(tab, Fourier(4))
            @test tab.stab != before
            apply!(tab, SUM(4, 12, 1))
            @test is_pure(tab; verify = true)
        end
    end

    @testset "Stored Clifford benchmark fixtures act and all leaves run" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5), n in (2, 16)
            group = stored_clifford_group(; d, n, T)
            @test !isempty(keys(group))
            for (path, leaf) in BenchmarkTools.leaves(group)
                trial = run(leaf; samples = 1, evals = 1, seconds = 0.05)
                @test !isempty(trial.times)
            end
            for k in (1, 2, 8)
                k <= n || continue
                fixture = stored_clifford_fixture(d, k)
                stored = T(d, n; state = :product, basis = :Z)
                named = deepcopy(stored)
                before = copy(stored.stab)
                apply!(stored, fixture.U)
                for g in fixture.gates
                    apply!(named, g)
                end
                @test stored.stab != before
                @test stored.stab == named.stab
                @test is_pure(stored; verify = true)
                if T === DestabilizerTableau
                    @test stored.destab == named.destab
                    @test stored.xdotz_cache == named.xdotz_cache
                end
            end
        end
    end

    @testset "Registration preserves a P1 baseline without stored operators" begin
        # The actual workload functions remain qualified to QuditClifford; this
        # module controls feature detection and models the P1 API surface.
        p1 = Module(:CliffordP1Only)
        for name in (:apply!, :Fourier, :Phase, :SUM)
            Core.eval(p1, Expr(:const, Expr(:(=), name, getfield(QuditClifford, name))))
        end
        @test !isdefined(p1, :CliffordOperator)
        suite = BenchmarkGroup()
        register_clifford_group!(suite, (StabilizerTableau,), (3,), (8,); api = p1)
        @test Set(keys(suite["clifford"])) == Set(["StabilizerTableau/d=3/n=8"])
        run(suite["clifford"]; samples = 1, evals = 1, seconds = 0.05)
        head = BenchmarkGroup()
        register_clifford_group!(head, (StabilizerTableau,), (3,), (8,))
        @test haskey(head["clifford"], "stored/StabilizerTableau/d=3/n=8")
    end
end
