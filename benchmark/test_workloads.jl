using Test
include(joinpath(@__DIR__, "workloads.jl"))

# Barrett and fallback call counts, filled by the tier testset at the end.
const TIER_COUNTS = zeros(Int, 2)

@testset "Workloads" begin
    @testset "Entropy profile comparison and baseline compatibility" begin
        types = (StabilizerTableau, DestabilizerTableau)
        for T in types, m in (4, 8)
            tab = entropy_profile_fixture(T, 3, 8; m)
            same = entropy_profile_fixture(T, 3, 8; m)
            @test tab.stab == same.stab
            @test tab.m == m
            @test entropy_profile_loop(tab) == entanglement_profile(tab)
            @test tab.stab == same.stab
        end

        no_api = Module(:NoEntropyAPI)
        empty_suite = BenchmarkGroup()
        register_entropy_profile_group!(empty_suite, types, (8,); api=no_api)
        @test isempty(empty_suite)

        # A 0.3 baseline can build the fixture and time the old loop, but has
        # no profile API. Its existing comparison column must still run.
        baseline_api = Module(:BaselineEntropyAPI)
        Core.eval(baseline_api, :(random_state! = nothing))
        allocating_api = Module(:AllocatingEntropyAPI)
        Core.eval(allocating_api, :(random_state! = nothing))
        Core.eval(allocating_api, :(entanglement_profile = nothing))
        for api in (baseline_api, allocating_api, QuditClifford)
            suite = BenchmarkGroup()
            register_entropy_profile_group!(suite, types, (8,); api)
            @test length(suite["entropy_profile"]) == 4
            for (_, group) in suite["entropy_profile"]
                expected_keys = api === baseline_api ? ["loop"] :
                                api === allocating_api ? ["loop", "profile"] :
                                ["loop", "profile", "profile!"]
                @test Set(keys(group)) == Set(expected_keys)
                for (_, leaf) in group
                    trial = run(leaf; samples=1, evals=1, seconds=0.01)
                    @test !isempty(trial.times)
                end
            end
        end
    end

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

    @testset "Phase-free deterministic measurement starts noncanonical" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5)
            n = 16
            mk = tableau_maker(T, d, n; storephase=false)
            tab = mk(:ghz)
            i, j = spread_sites(n)
            op = DoublePauli(i, 0, 1, j, 0, d - 1)
            before = copy(tab.stab)
            snap = snapshot(tab)

            @test tab.m == n
            @test !tab.storephase
            @test !tab.iscanonical
            @test measure!(tab, op; outcome=d - 1) == 0
            @test tab.stab == before
            @test !tab.iscanonical

            canonicalize!(tab)
            @test tab.stab != before  # The data, not only the flag, is noncanonical.
            restore!(snap)
            @test tab.stab == before
            @test !tab.iscanonical

            trial = run(bench_measure(mk, :ghz, op); samples=2, evals=1, seconds=0.01)
            @test length(trial.times) == 2
        end
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

    @testset "Clifford registration skips a baseline without the Clifford API" begin
        suite = BenchmarkGroup()
        existing = BenchmarkGroup()
        suite["existing"] = existing
        no_clifford_api = Module(:NoCliffordAPI)
        @test register_clifford_group!(suite, (StabilizerTableau,), (3,), (8,); api = no_clifford_api) === suite
        @test Set(keys(suite)) == Set(["existing"])
        @test suite["existing"] === existing

        # In this checkout the Clifford APIs exist, so normal registration adds
        # the clifford group alongside the existing one, rather than replacing
        # the whole suite.
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

    @testset "Stored construct group is registered once per d and all its leaves run" begin
        for d in (2, 3, 5)
            group = stored_clifford_construct_group(d)
            @test !isempty(keys(group))
            for (path, leaf) in BenchmarkTools.leaves(group)
                trial = run(leaf; samples = 1, evals = 1, seconds = 0.05)
                @test !isempty(trial.times)
            end
            for k in (1, 2, 8)
                fixture = stored_clifford_fixture(d, k)
                @test CliffordOperator(d, fixture.targets, fixture.F, fixture.a) == fixture.U
                @test CliffordOperator(d, fixture.targets, fixture.F, fixture.a;
                                       check = false) == fixture.U
            end
        end

        # register_clifford_group! must add exactly one construct cell per d,
        # regardless of how many (T, n) cells share that d.
        suite = BenchmarkGroup()
        register_clifford_group!(suite, (StabilizerTableau, DestabilizerTableau), (3,), (8, 16))
        construct_keys = [k for k in keys(suite["clifford"]) if startswith(k, "stored/construct/")]
        @test Set(construct_keys) == Set(["stored/construct/d=3"])
    end

    @testset "scrambled_tableau is dense, physical and deterministic" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            n = 16
            mk = tableau_maker(T, d, n)

            a = scrambled_tableau(mk, d, n; seed = 11)
            b = scrambled_tableau(mk, d, n; seed = 11)
            c = scrambled_tableau(mk, d, n; seed = 12)
            @test a.stab == b.stab                            # deterministic
            @test a.stab != c.stab                            # the seed actually matters
            T === DestabilizerTableau && @test a.destab == b.destab

            @test is_pure(a; verify = true)
            @test a.stab != T(d, n; state = :product).stab    # actually scrambled

            # Dense local coordinates: > 25% of the k = 8 target rows are
            # nonzero across all n = 16 active columns -- the property
            # stored_clifford_group relies on scrambled_tableau for. A
            # :product or :mixed leaf is instead mostly zero or one-hot (see
            # scrambled_tableau's docstring).
            k = 8
            rows = vcat(1:k, n .+ (1:k))
            density = count(!iszero, a.stab[rows, 1:a.m]) / (length(rows) * a.m)
            @test density > 0.25
        end
    end

    @testset "Every stored, algebra and safe leaf runs" begin
        n = 16
        ks = (1, 2, 8)
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            group = stored_clifford_group(; d, n, T, ks, product = true)
            @test !isempty(keys(group))
            for (path, leaf) in BenchmarkTools.leaves(group)
                trial = run(leaf; samples = 1, evals = 1, seconds = 0.05)
                @test !isempty(trial.times)
            end
            # k = 1, 2, 8 all <= n = 16, and product = true adds a third leaf
            # per k on top of the base and m=0 leaves.
            @test length(keys(group)) == 3 * length(ks)
        end

        for d in (2, 3)
            group = stored_clifford_algebra_group(d; ks)
            @test !isempty(keys(group))
            for (path, leaf) in BenchmarkTools.leaves(group)
                trial = run(leaf; samples = 1, evals = 1, seconds = 0.05)
                @test !isempty(trial.times)
            end
            @test length(keys(group)) == 3 * length(ks)
        end

        let group = stored_clifford_safe_group((StabilizerTableau, DestabilizerTableau))
            @test !isempty(keys(group))
            for (path, leaf) in BenchmarkTools.leaves(group)
                trial = run(leaf; samples = 1, evals = 1, seconds = 0.05)
                @test !isempty(trial.times)
            end
            @test Set(keys(group)) ==
                  Set(["apply!/StabilizerTableau/k=8", "apply!/DestabilizerTableau/k=8",
                       "conjugate/k=8", "inv/k=8", "compose/k=8"])
        end
    end

    @testset "Benchmarked stored-operator calls compute the right thing" begin
        n = 16
        for d in (2, 3), k in (1, 2, 8)
            fixture = stored_clifford_fixture(d, k)
            U = fixture.U
            S = 2k

            # inv(U) ∘ U is the identity operator on U's targets: ∘ applies
            # its right operand first, so this is inv(U) after U, which must
            # cancel exactly.
            V = inv(U) ∘ U
            identity_F = [Int(i == j) for i in 1:S, j in 1:S]
            @test V.F == identity_F
            @test all(iszero, V.a)
            @test V.targets == U.targets

            # conjugate(U, op) equals sequential named conjugation through
            # fixture.gates, in the SAME order stored_clifford_fixture built U
            # from. U's whole point is that its conjugation action equals
            # that composed action for ANY op, not only the basis vectors
            # used to build F -- conjugation is linear in the exponent
            # vector, so the columns determine the map everywhere.
            rng = Random.MersenneTwister(1000 + d * 100 + k)
            xz = [rand(rng, 0:(d - 1)) for _ in 1:S]
            phase = rand(rng, 0:(QuditClifford.phase_modulus(d) - 1))
            op = GeneralPauli(xz, phase)
            lhs = QuditClifford.conjugate(U, op)
            ref = deepcopy(op)
            for g in fixture.gates
                ref = QuditClifford.conjugate(g, ref; d)
            end
            @test lhs.xz == ref.xz
            @test mod(lhs.phase, QuditClifford.phase_modulus(d)) ==
                  mod(ref.phase, QuditClifford.phase_modulus(d))

            # U ∘ U applied to a tableau equals applying U twice.
            UU = U ∘ U
            tab1 = DestabilizerTableau(d, n; state = :product)
            apply!(tab1, UU)
            tab2 = DestabilizerTableau(d, n; state = :product)
            apply!(tab2, U)
            apply!(tab2, U)
            @test tab1.stab == tab2.stab
            @test tab1.destab == tab2.destab

            # Stored apply! on the scrambled state equals applying
            # fixture.gates in order -- what stored_clifford_group's
            # "apply!/stored/k=$k" leaf (via scrambled_tableau) measures.
            mk = tableau_maker(DestabilizerTableau, d, n)
            scr = scrambled_tableau(mk, d, n; seed = 11)
            stored = deepcopy(scr)
            apply!(stored, U)
            named = deepcopy(scr)
            for g in fixture.gates
                apply!(named, g)
            end
            @test stored.stab == named.stab
            @test stored.destab == named.destab
        end
    end

    @testset "Overflow-safe group reaches the safe tier and computes correctly" begin
        d = 1_000_000_007
        n = 8
        k = 8
        @test QuditClifford.max_safe_dimension(n) >= d   # the tableau stays safe
        @test !QuditClifford.clifford_fast_dots(2k, d)    # but the Clifford dot does not

        fixture = stored_clifford_fixture(d, k)
        U = fixture.U
        @test !U.fast

        for T in (StabilizerTableau, DestabilizerTableau)
            mk = tableau_maker(T, d, n; inversemod = QuditClifford.JustInTimeInvMod())
            scr = scrambled_tableau(mk, d, n; seed = 20260910)
            @test is_pure(scr; verify = true)

            stored = deepcopy(scr)
            apply!(stored, U)
            named = deepcopy(scr)
            for g in fixture.gates
                apply!(named, g)
            end
            @test stored.stab == named.stab
            T === DestabilizerTableau && @test stored.destab == named.destab
        end
    end

    @testset "Random registration skips a baseline without the sampling API" begin
        suite = BenchmarkGroup()
        existing = BenchmarkGroup()
        suite["existing"] = existing
        no_random_api = Module(:NoRandomAPI)
        @test register_random_group!(suite, (StabilizerTableau,), (3,), (8,);
                                     circuits = true, api = no_random_api) === suite
        @test Set(keys(suite)) == Set(["existing"])
        @test suite["existing"] === existing

        types = (StabilizerTableau, DestabilizerTableau)
        head = BenchmarkGroup()
        @test register_random_group!(head, types, (2, 3), (8, 16);
                                     ks = (1, 2, 8), circuits = true) === head
        expected = Set(vcat(["refill/d=$d/k=$k" for d in (2, 3) for k in (1, 2, 8)],
                            ["sample/d=$d/k=8" for d in (2, 3)],
                            ["state/$(nameof(T))/d=$d/n=8/m=$m"
                             for T in types for d in (2, 3) for m in (1, 8)]))
        @test Set(keys(head["random"])) == expected
        @test Set(keys(head["circuit"])) ==
              Set(vcat(["monitored/d=2/n=$n" for n in (16, 32, 64)],
                       ["monitored/$(nameof(T))/d=3/n=32/layers=64" for T in types]))
        for T in types
            leaf = head["circuit"]["monitored/$(nameof(T))/d=3/n=32/layers=64"]
            @test !isempty(run(leaf; samples = 1, evals = 1, seconds = 0.05).times)
        end
        # @benchmarkable bodies are quoted, so a broken leaf is invisible until run.
        for (path, leaf) in BenchmarkTools.leaves(head["random"])
            trial = run(leaf; samples = 1, evals = 1, seconds = 0.05)
            @test !isempty(trial.times)
        end
        # Same reasoning for the monitored circuit leaves: run the cheapest
        # one (n = 16) rather than a whole trajectory at every n.
        monitored_trial = run(head["circuit"]["monitored/d=2/n=16"];
                              samples = 1, evals = 1, seconds = 0.05)
        @test !isempty(monitored_trial.times)
    end

    @testset "Random workloads compute what their leaves claim" begin
        for d in (2, 3, 5), k in (1, 2, 8)
            fixture = random_operator_fixture(d, k)
            @test fixture.U == random_operator_fixture(d, k).U      # repeatable
            for _ in 1:3
                random_clifford!(fixture.rng, fixture.U)
                # The checked constructor accepts only symplectic F and, at
                # d = 2, Hermitian generator images.
                U = fixture.U
                @test CliffordOperator(d, U.targets, U.F, U.a) == U
            end
        end
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3), n in (8, 16)
            for m in (1, n)
                (; rng, tab) = random_state_fixture(T, d, n)
                random_state!(rng, tab; m)
                @test tab.m == m
                @test is_pure(tab; verify = true) == (m == n)
            end
        end
    end

    @testset "Monitored trajectory is deterministic, pure and allocation-free" begin
        warm_allocations(tab, bonds, rng) =
            (monitored_layers!(tab, bonds, rng, 4); @allocated monitored_layers!(tab, bonds, rng, 4))
        for T in (StabilizerTableau, DestabilizerTableau)
            a = monitored_trajectory(d = 2, n = 8, layers = 16, seed = 5, T = T)
            b = monitored_trajectory(d = 2, n = 8, layers = 16, seed = 5, T = T)
            @test a.stab == b.stab
            @test is_pure(a; verify = true)
            # layers defaults to 4n.
            @test monitored_trajectory(d = 2, n = 4, seed = 5, T = T).stab ==
                  monitored_trajectory(d = 2, n = 4, layers = 16, seed = 5, T = T).stab
            tab, bonds, rng = monitored_setup(d = 2, n = 8, seed = 5, T = T)
            @test warm_allocations(tab, bonds, rng) == 0
        end
        # A periodic brickwork chain needs an even number of at least two sites.
        for n in (0, 1, 3)
            @test_throws ArgumentError monitored_setup(d = 2, n = n, seed = 1)
        end
        # Z measurements and parity-respecting qubit Cliffords never warn.
        @test_logs monitored_trajectory(d = 2, n = 6, layers = 8, seed = 3)
    end

    @testset "Registration preserves a baseline without CliffordOperator" begin
        # The actual workload functions remain qualified to QuditClifford; this
        # module controls feature detection and models an API surface without
        # CliffordOperator.
        named_only = Module(:NamedGatesOnly)
        for name in (:apply!, :Fourier, :Phase, :SUM)
            Core.eval(named_only, Expr(:const, Expr(:(=), name, getfield(QuditClifford, name))))
        end
        @test !isdefined(named_only, :CliffordOperator)
        suite = BenchmarkGroup()
        register_clifford_group!(suite, (StabilizerTableau,), (3,), (8,); api = named_only)
        @test Set(keys(suite["clifford"])) == Set(["StabilizerTableau/d=3/n=8"])
        # No stored/... key of any kind -- construct, algebra and safe
        # included -- appears without CliffordOperator.
        @test !any(startswith(k, "stored/") for k in keys(suite["clifford"]))
        run(suite["clifford"]; samples = 1, evals = 1, seconds = 0.05)

        head = BenchmarkGroup()
        register_clifford_group!(head, (StabilizerTableau,), (3,), (8,); ks = (1, 2, 8))
        @test haskey(head["clifford"], "stored/StabilizerTableau/d=3/n=8")
        # With CliffordOperator, registration also adds the per-d algebra group
        # and the fixed-d safe group beside the stored/construct cell.
        @test haskey(head["clifford"], "stored/algebra/d=3")
        @test haskey(head["clifford"], "stored/safe/d=1000000007")
    end

    @testset "Read-only leaves evaluate many times per sample" begin
        mk = tableau_maker(DestabilizerTableau, 3, 64)
        @test bench_expect(mk, spread_z(64)).params.evals == READONLY_EVALS
        @test bench_reset(mk).params.evals == READONLY_EVALS
        for (_, leaf) in BenchmarkTools.leaves(stored_clifford_construct_group(3))
            @test leaf.params.evals == READONLY_EVALS
        end
        for (_, leaf) in BenchmarkTools.leaves(stored_clifford_algebra_group(3))
            @test leaf.params.evals == READONLY_EVALS
        end
        for (_, leaf) in BenchmarkTools.leaves(random_refill_group(3))
            @test leaf.params.evals == READONLY_EVALS
        end
        # Repeating expect! leaves its tableau where it was, which is what
        # makes evals > 1 measure the same call every time.
        tab = mk(:product)
        canonicalize!(tab)
        before = copy(tab.stab)
        for _ in 1:5
            expect!(tab, spread_z(64))
        end
        @test tab.stab == before
    end

    @testset "Random leaves reseed their RNG in setup" begin
        # Every sample of a random/* leaf draws the same operator or state.
        (; rng, U) = random_operator_fixture(3, 2)
        draws = map(1:2) do _
            Random.seed!(rng, 20260925 + 2)
            copy(QuditClifford.random_clifford!(rng, U).F)
        end
        @test draws[1] == draws[2]
    end

    @testset "Dense fixtures are volume-law, identical across types and noncanonical" begin
        for d in (2, 3, 5), n in (8, 64)
            D = dense_state(DestabilizerTableau, d, n)
            S = dense_state(StabilizerTableau, d, n)
            @test D.m == S.m == n
            @test S == D
            @test is_pure(D; verify = true)
            @test dense_state(DestabilizerTableau, d, n).stab == D.stab   # seeded
            # most entries nonzero, unlike a constructor state's one-hot rows
            @test count(!iszero, D.stab[1:(2n), :]) > 0.4 * 2n * n
            for tab in (S, D)
                @test !tab.iscanonical
                before = copy(tab.stab)
                c = copy(tab)
                canonicalize!(c)
                @test c.stab != before        # canonicalize! has real work to do
            end
            mixed = dense_state(StabilizerTableau, d, n; m = n ÷ 2)
            @test mixed.m == n ÷ 2
            @test mixed == dense_state(DestabilizerTableau, d, n; m = n ÷ 2)
        end
    end

    @testset "Dense measure! leaves hit the branches they claim" begin
        for T in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5), n in (8, 64)
            pure = dense_state(T, d, n)
            D = dense_state(DestabilizerTableau, d, n)

            # noncommuting: spread_x anticommutes with many generators, so the
            # update rewrites a large share of the columns
            op = first_anticommuting(pure, [spread_x(n)])
            @test op == spread_x(n)
            anti = count(j -> mod(QuditClifford.commutation_col(pure.stab, j, op), d) != 0,
                         1:pure.m)
            @test anti >= n ÷ 4
            t = copy(pure)
            measure!(t, op; outcome = 0, phase_policy = 2)
            @test t.m == n && t != pure

            # deterministic: the span product decomposes into exactly cld(n, 2)
            # generators, read off the dual basis
            inspan = span_product(pure)
            @test inspan.xz == span_product(D).xz
            coeffs = count(j -> mod(QuditClifford.commutation_col(D.destab, j, inspan), d) != 0,
                           1:n)
            @test coeffs == cld(n, 2)
            @test expect_int!(copy(pure), inspan) != -1
            t = copy(pure)
            measure!(t, inspan; outcome = 0, phase_policy = 2)
            @test t.m == n && t == pure

            # append: commutes with every generator of the m = n/2 state, lies
            # outside its span, and raises m by one
            mixed = dense_state(T, d, n; m = n ÷ 2)
            w = commuting_out_of_span(dense_state(DestabilizerTableau, d, n; m = n ÷ 2))
            @test any(!iszero, w.xz)
            @test all(j -> mod(QuditClifford.commutation_col(mixed.stab, j, w), d) == 0,
                      1:mixed.m)
            @test expect_int!(copy(mixed), w) == -1
            t = copy(mixed)
            measure!(t, w; outcome = 0, phase_policy = 2)
            @test t.m == n ÷ 2 + 1

            # every leaf runs, and a restored sample starts from the same state
            g = dense_group(; d, n, T)
            @test Set(keys(g)) == Set(["measure!/noncommuting", "measure!/deterministic",
                                       "measure!/append", "expect!", "canonicalize!"])
            run(g; samples = 2, evals = 1, seconds = 0.05)
        end
    end

    @testset "Channel and information leaves" begin
        for T in (StabilizerTableau, DestabilizerTableau), n in (8, 64)
            pure = dense_state(T, 3, n)
            s1, _ = spread_sites(n)
            quarter = collect(s1:(s1 + n ÷ 4 - 1))
            # each channel lowers the rank of the pure dense state
            for f! in (t -> dephase!(t, spread_x(n)), t -> traceout!(t, s1),
                       t -> traceout!(t, quarter), t -> reset!(t, s1))
                t = copy(pure)
                f!(t)
                @test t.m < n
            end
            run(channel_group(; d = 3, n, T); samples = 2, evals = 1, seconds = 0.05)

            # the information measures leave the state untouched, so their
            # leaves can share one tableau across samples
            A, B, C = quarters(n)
            @test isempty(intersect(A, B)) && isempty(intersect(B, C)) && maximum(C) <= n
            for m in (n, n ÷ 2)
                tab = dense_state(T, 3, n; m)
                before, canon = copy(tab.stab), tab.iscanonical
                @test mutual_information(tab, A, B) >= 0
                tripartite_information(tab, A, B, C)
                @test conditional_mutual_information(tab, A, B, C) >= 0
                @test entanglement_negativity(tab, A, B) >= 0
                @test tab.stab == before && tab.iscanonical == canon && tab.m == m
            end
            g = information_group(; d = 3, n, T)
            @test length(g) == 8
            run(g; samples = 1, evals = 1, seconds = 0.05)
        end
    end

    @testset "Gate and tableau-utility leaves" begin
        for T in (StabilizerTableau, DestabilizerTableau), n in (8, 64)
            g = gate_group(; d = 5, n, T)
            @test length(g) == 7
            run(g; samples = 2, evals = 1, seconds = 0.05)

            u = tableau_group(; d = 3, n, T)
            @test haskey(u, "measure!/rng")
            # the read-only leaves still see the fixture after the measuring
            # leaf has run: == takes the entrywise fast path, which stays at
            # zero allocations
            run(u["measure!/rng"]; samples = 2, evals = 1, seconds = 0.05)
            run(u; samples = 2, evals = 1, seconds = 0.05)
            @test allocs(run(u["==/same-basis"]; samples = 2, evals = 1)) == 0

            tab = scrambled_tableau(tableau_maker(T, 3, n), 3, n; seed = DENSE_SEED)
            other = copy(tab)
            canonicalize!(other)
            @test !tab.iscanonical && other.stab != tab.stab   # the slow == path
            @test tab == other && hash(tab) == hash(other)
            nraw = max(2, n ÷ 4)
            raw = scrambled_tableau(tableau_maker(T, 3, nraw), 3, nraw; seed = DENSE_SEED)
            M = copy(raw.stab)
            @test T(3, M; m = nraw) == raw
            @test T(3, M; m = nraw, check = false) == raw
            @test M == raw.stab                                # input not modified

            # measure!/rng: restoring the snapshot and reseeding gives the same
            # drawn outcome every sample
            op = first_anticommuting(tab, [spread_x(n)])
            outcomes = map(1:2) do _
                t = copy(tab)
                rng = Random.Xoshiro(DENSE_SEED)
                measure!(rng, t, op; phase_policy = 2)
                t.stab[end, :]
            end
            @test outcomes[1] == outcomes[2]
        end
    end

    @testset "Breadth registration" begin
        types = (StabilizerTableau, DestabilizerTableau)
        no_api = BenchmarkGroup()
        register_breadth_groups!(no_api, types; n = 8, dense_ds = (2, 3), modulus_n = 8,
                                 api = Module(:NoRandomState))
        @test isempty(no_api)
        suite = BenchmarkGroup()
        register_breadth_groups!(suite, types; n = 8, dense_ds = (2, 3), modulus_n = 8)
        @test Set(keys(suite)) ==
              Set(["dense", "modulus", "channels", "information", "gates", "tableau"])
        @test length(suite["dense"]) == 4
        @test length(suite["modulus"]) == 2 * length(MODULUS_PRIMES)
    end

    @testset "Purification leaves purify, except the capped one" begin
        for n in (16, 32)
            _, layer = purification_trajectory(d = 3, n = n, seed = 1234)
            @test layer <= 10_000
        end
        hist, layer = purification_trajectory(d = 3, n = 64, seed = 1234)
        @test layer == 10_001
        @test hist[end] < 64
    end

    # Count which tier of src/modular.jl each leaf reaches. Every call with a
    # multiplier other than 0, 1 and d-1 asks `_barrett_valid` once, so a
    # counting copy of it records Barrett (true) and fallback (false) calls.
    # The copy recomputes the predicate from the same table, so results are
    # unchanged; if those internals move, this errors instead of passing.
    @testset "Dense and modulus leaves reach the arithmetic tier they claim" begin
        @eval QuditClifford @inline function _barrett_valid(d::Int)
            ok = d < 1 ? false :
                 d <= _BARRETT_TABLE_MAX ? (@inbounds _BARRETT_VALID[d]) :
                 (ispow2(d) && ((1 << _BARRETT_K) % d == 0))
            Main.TIER_COUNTS[ok ? 1 : 2] += 1
            return ok
        end
        function tiers(leaf)
            fill!(TIER_COUNTS, 0)
            run(leaf; samples = 1, evals = 1, seconds = 1.0)
            return (barrett = TIER_COUNTS[1], fallback = TIER_COUNTS[2])
        end

        # the counter sees the existing fallback probe
        @test tiers(bench_canonicalize_state(fallback_probe_state(16))).fallback > 0

        @test QuditClifford._barrett_valid(5)
        for T in (StabilizerTableau, DestabilizerTableau)
            for (key, leaf) in dense_group(; d = 5, n = 16, T)
                c = tiers(leaf)
                @test c.barrett > 0
                @test c.fallback == 0
            end
            for d in MODULUS_PRIMES
                barrett = QuditClifford._barrett_valid(d)
                @test barrett == (d == 443)
                for (key, leaf) in modulus_group(; d, n = 16, T)
                    c = tiers(leaf)
                    @test (barrett ? c.barrett : c.fallback) > 0
                    @test (barrett ? c.fallback : c.barrett) == 0
                end
            end
        end
    end
end
