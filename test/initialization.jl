using Test
using QuditClifford

@testset "Initialization presets" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                # Mixed state
                tab_mixed = TT(2, 3; state=:mixed, storephase=true)
                @test tab_mixed.m == 0
                @test all(tab_mixed.stab .== 0)

                # Accept transposed tableau inputs (m × (2n+1))
                tab_row = reshape(Int[0, 1, 0], 1, 3)
                tab_row = TT(2, tab_row; m=1, storephase=true)
                @test tab_row.stab[2, 1] == 1

                # Product states in Z and X bases
                tab_z = TT(2, 2; state=:product, basis=:Z, storephase=true)
                @test tab_z.m == 2
                @test tab_z.stab[3, 1] == 1
                @test tab_z.stab[4, 2] == 1

                tab_x = TT(2, 2; state=:product, basis=:X, storephase=true)
                @test tab_x.stab[1, 1] == 1
                @test tab_x.stab[2, 2] == 1
                tab_x_alias = TT(2, 1; state=:X)
                @test tab_x_alias.stab[1, 1] == 1

                # Y basis (qubits only) requires phase row
                tab_y = TT(2, 1; state=:product, basis=:Y, storephase=true)
                @test tab_y.stab[1, 1] == 1
                @test tab_y.stab[2, 1] == 1
                @test tab_y.stab[3, 1] == 1

                # GHZ state
                tab_ghz = TT(2, 3; state=:ghz, storephase=true)
                @test tab_ghz.m == 3
                @test tab_ghz.stab[1, 1] == 1
                @test tab_ghz.stab[2, 1] == 1
                @test tab_ghz.stab[3, 1] == 1
                @test tab_ghz.stab[4, 2] == 1
                @test tab_ghz.stab[5, 2] == 1
                @test tab_ghz.stab[5, 3] == 1
                @test tab_ghz.stab[6, 3] == 1

                # reset! to presets
                tab_reset = TT(2, 2; state=:mixed, storephase=true)
                reset!(tab_reset; state=:product, basis=:Z)
                @test tab_reset.m == 2
                @test tab_reset.stab[3, 1] == 1
                @test tab_reset.stab[4, 2] == 1
                reset!(tab_reset, :ghz)
                @test tab_reset.stab[1, 1] == 1
                @test tab_reset.stab[2, 1] == 1
                @test tab_reset.stab[3, 2] == 1

                tab_nophase = TT(2, 1; state=:mixed, storephase=false)
                reset!(tab_nophase; state=:product, basis=:Y)

                # reset! should not allocate (after warm-up)
                reset!(tab_reset; state=:mixed)
                alloc_mixed = @allocated reset!(tab_reset; state=:mixed)
                @test alloc_mixed == 0

                basis_vec = [:Z, :X]
                reset!(tab_reset; state=:product, basis=basis_vec)
                alloc_prod = @allocated reset!(tab_reset; state=:product, basis=basis_vec)
                @test alloc_prod == 0

                reset!(tab_reset, :ghz)
                alloc_ghz = @allocated reset!(tab_reset, :ghz)
                @test alloc_ghz == 0
            end

            @testset "Qudit (d=3)" begin
                tab_mixed = TT(3, 2; state=:mixed, storephase=true)
                @test tab_mixed.m == 0
                @test all(tab_mixed.stab .== 0)

                tab_z = TT(3, 2; state=:product, basis=:Z, storephase=true)
                @test tab_z.m == 2
                @test tab_z.stab[3, 1] == 1
                @test tab_z.stab[4, 2] == 1

                tab_x = TT(3, 2; state=:product, basis=:X, storephase=true)
                @test tab_x.stab[1, 1] == 1
                @test tab_x.stab[2, 2] == 1
                tab_x_alias = TT(3, 1; state=:X)
                @test tab_x_alias.stab[1, 1] == 1

                tab_ghz = TT(3, 3; state=:ghz, storephase=true)
                @test tab_ghz.m == 3
                @test tab_ghz.stab[1, 1] == 1
                @test tab_ghz.stab[2, 1] == 1
                @test tab_ghz.stab[3, 1] == 1
                @test tab_ghz.stab[4, 2] == 1
                @test tab_ghz.stab[5, 2] == 2 # d-1 for qudits
                @test tab_ghz.stab[5, 3] == 1
                @test tab_ghz.stab[6, 3] == 2 # d-1 for qudits

                tab_reset = TT(3, 2; state=:mixed, storephase=true)
                reset!(tab_reset; state=:product, basis=:Z)
                @test tab_reset.m == 2
                reset!(tab_reset, :ghz)
                @test tab_reset.stab[1, 1] == 1
                @test tab_reset.stab[2, 1] == 1
                @test tab_reset.stab[3, 2] == 1
            end
        end
    end
end

@testset "Matrix construction and validation" begin
    for (label, TT) in [
        ("StabilizerTableau", StabilizerTableau),
        ("DestabilizerTableau", DestabilizerTableau),
    ]
        @testset "$label" begin
            # A compact active-generator matrix is padded to the inferred capacity.
            partial = reshape(Int[0, 0, 1, 0, 0], 5, 1) # Z₁ on a two-qudit system
            tab = TT(3, partial)
            @test tab.n == 2
            @test tab.m == 1
            @test size(tab.stab) == (5, 2)
            @test tab.stab[:, 1] == partial[:, 1]
            @test all(iszero, tab.stab[:, 2])

            full = zeros(Int, 5, 2)
            full[3, 1] = 1
            full[4, 2] = 1
            full_tab = TT(3, full)
            @test full_tab.n == full_tab.m == 2
            @test full_tab.stab == full

            # Positional state construction is an exact alias of the keyword API.
            positional = TT(2, 1, :X)
            keyword = TT(2, 1; state=:X)
            @test positional.stab == keyword.stab
            @test positional.m == keyword.m == 1

            @test_throws ArgumentError TT(2, zeros(Int, 4, 3))
            @test_throws ArgumentError TT(2, zeros(Int, 4, 2); storephase=true)
            @test_throws ArgumentError TT(2, zeros(Int, 5, 2); m=-1)
            @test_throws ArgumentError TT(2, zeros(Int, 5, 2); m=3)
            @test_throws ArgumentError TT(2, zeros(Int, 6, 2); m=1, storephase=false)
            @test_throws ArgumentError TT(4, 1; state=:mixed)
        end
    end
end

@testset "State aliases, heterogeneous bases, and validation" begin
    for (label, TT) in [
        ("StabilizerTableau", StabilizerTableau),
        ("DestabilizerTableau", DestabilizerTableau),
    ]
        @testset "$label" begin
            tab = TT(2, 3; state=:product, basis=(:x, :Y, :z))
            @test tab.stab[:, 1] == Int[1, 0, 0, 0, 0, 0, 0]
            @test tab.stab[:, 2] == Int[0, 1, 0, 0, 1, 0, 1]
            @test tab.stab[:, 3] == Int[0, 0, 0, 0, 0, 1, 0]

            mixed_alias = TT(2, 1; state=:maximally_mixed, basis=:z)
            @test mixed_alias.m == 0
            @test all(iszero, mixed_alias.stab)

            @test_throws ArgumentError TT(2, 1; state=:X, basis=:Y)
            @test_throws ArgumentError TT(2, 1; state=:X, basis=[:X])
            @test_throws ArgumentError TT(2, 1; state=:mixed, basis=[:Z])
            @test_throws ArgumentError TT(2, 1; state=:ghz, basis=:X)
            @test_throws ArgumentError TT(2, 1; state=:unknown)
            @test_throws ArgumentError TT(2, 1; state=:product, basis=:unknown)
            @test_throws ArgumentError TT(2, 2; state=:product, basis=(:X,))
            @test_throws ArgumentError TT(2, 2; state=:product, basis=(:X, 1))
            @test_throws ArgumentError TT(2, 1; state=:product, basis=1)
        end
    end

    # The preset builder has its own defensive contract in addition to the
    # public normalization layer.
    raw = zeros(Int, 3, 1)
    @test_throws ArgumentError QuditClifford._preset_tableau!(
        raw,
        2,
        1,
        :unknown,
        nothing,
        true,
    )
end

@testset "Narrow and unsigned inverse tables behave as Int tables" begin
    # The package is Int arithmetic throughout: binom2_mod_oddprime takes an
    # Int, and measure! forms mod(-commutator * inv, d). Handed an unsigned
    # inverse, that negation and multiply would wrap in unsigned arithmetic
    # BEFORE the mod, leaving both tableau types with non-commuting generators
    # and no error at all -- a corrupt state, not an exception -- and an Int32
    # table would hit a MethodError in binom2_mod_oddprime. PrecomputedInvMod
    # therefore converts to Vector{Int} on construction, so neither can arise;
    # these assert that every accepted table type behaves exactly as the Int
    # control.
    tables = (Int[1, 2], Int32[1, 2], UInt64[1, 2])
    mk(TT, tbl; kw...) = TT(3, 2; inversemod=QuditClifford.PrecomputedInvMod(tbl), kw...)

    for TT in (StabilizerTableau, DestabilizerTableau)
        # 1+2. canonicalization of a GHZ tableau agrees with the Int control
        ref = (t = mk(TT, tables[1]; state=:ghz); canonicalize!(t); copy(t.stab))
        for tbl in tables
            t = mk(TT, tbl; state=:ghz)
            canonicalize!(t)
            @test t.stab == ref
        end

        # 3. expectation / span reconstruction agrees
        op = DoublePauli(1, 0, 1, 2, 0, 1)
        want = expect_int!(mk(TT, tables[1]; state=:ghz), op)
        for tbl in tables
            @test expect_int!(mk(TT, tbl; state=:ghz), op) == want
        end

        # 4. a non-commuting measurement leaves the generators mutually
        #    commuting -- this is the assertion the corrupt state failed.
        ctrl = (t = mk(TT, tables[1]; state=:product, basis=:X);
                measure!(t, op; outcome=0); t)
        for tbl in tables
            tab = mk(TT, tbl; state=:product, basis=:X)
            @test measure!(tab, op; outcome=0) == 0
            for i in 1:tab.m, j in 1:tab.m
                @test mod(QuditClifford.commutation_col(tab.stab, i, tab.stab[:, j]), tab.d) == 0
            end
            @test tab.stab == ctrl.stab
            @test tab.m == ctrl.m
            @test all(tab.stab[:, (tab.m+1):end] .== 0)      # capacity stays zeroed

            # 5. destabilizer duality and the x.z cache survive too
            if TT === DestabilizerTableau
                @test tab.destab == ctrl.destab
                for i in 1:tab.m, j in 1:tab.m
                    v = mod(sum(tab.destab[q, i] * tab.stab[tab.n + q, j] -
                                tab.destab[tab.n + q, i] * tab.stab[q, j] for q in 1:tab.n), tab.d)
                    @test v == (i == j ? 1 : 0)
                end
                for j in 1:tab.m
                    @test tab.xdotz_cache[j] == QuditClifford.dot_xz_col(tab.stab, tab.n, j, tab.d)
                end
            end
        end
    end
end

@testset "Large-dimension overflow warning" begin
    # Every phase and symplectic dot product accumulates n terms of size up to
    # (d-1)^2, and the odd-d phase update sums two such terms, so a tableau is
    # only safe while max(n, 2)*(d-1)^2 fits in an Int. Past that those sums
    # wrap and every result is silently wrong, so the constructors warn.
    safe_d(n) = isqrt(typemax(Int) ÷ max(n, 2)) + 1
    n = 4
    over, under = Sys.WORD_SIZE == 64 ? (3037000507, 1518500213) : (65537, 8191)
    @test over > safe_d(n)
    @test under <= safe_d(n)

    jit = QuditClifford.JustInTimeInvMod()
    @test_logs (:warn, r"overflow") StabilizerTableau(over, n; state=:mixed, inversemod=jit)
    @test_logs (:warn, r"overflow") DestabilizerTableau(over, n; state=:mixed, inversemod=jit)

    # Inside the bound, silence -- from both builders, preset and raw paths.
    @test_logs StabilizerTableau(under, n; state=:mixed, inversemod=jit)
    @test_logs DestabilizerTableau(under, n; state=:mixed, inversemod=jit)
    @test_logs StabilizerTableau(3, 4; state=:ghz)
    @test_logs DestabilizerTableau(2, 8; state=:product, basis=:Z)
    # Z₁, Z₂ rather than an all-zero matrix: zero columns are rank 0, which the
    # raw-matrix constructor rejects. This line is here to prove the raw path is
    # silent, not to probe the generator contract.
    @test_logs StabilizerTableau(3, [0 0; 0 0; 1 0; 0 1; 0 0]; m=2, storephase=true)

    # The bound is on n*(d-1)^2, not on d alone: the same d is safe at n = 2
    # and not at n = 256. A check on d by itself could not express this.
    d_mid = Sys.WORD_SIZE == 64 ? 1000000007 : 32749
    @test d_mid <= safe_d(2)
    @test d_mid > safe_d(256)
    @test_logs StabilizerTableau(d_mid, 2; state=:mixed, inversemod=jit)
    @test_logs (:warn, r"overflow") StabilizerTableau(d_mid, 256; state=:mixed, inversemod=jit)

    # n = 0 has no dot products at all and must not warn or divide by zero.
    @test_logs StabilizerTableau(over, 0; state=:mixed, inversemod=jit)

    # The boundary value itself is safe and one past it is not. Exercised on
    # the helper directly: the constructors also demand a prime d, and
    # max_safe_dimension(n) is not prime, so no tableau can sit exactly there.
    @test_logs QuditClifford._warn_if_dimension_unsafe(safe_d(4), 4)
    @test_logs (:warn, r"overflow") QuditClifford._warn_if_dimension_unsafe(safe_d(4) + 1, 4)

    # The bound is exact, not approximate: max_safe_dimension(n) is the largest
    # d whose worst-case accumulator still fits, and one more does not. Computed
    # in Int128 so the check cannot itself overflow. This is what pins the
    # max(n, 2) factor -- without it n = 1 would be off by 2x.
    for nn in (1, 2, 4, 64, 256, 1024)
        dm = Int128(QuditClifford.max_safe_dimension(nn))
        terms = Int128(max(nn, 2))
        @test terms * (dm - 1)^2 <= typemax(Int)
        @test terms * dm^2 > typemax(Int)
    end

    # maxlog=1 budgets by the log message's id. With the default id -- the call
    # site -- one unsafe tableau would spend the budget for every dimension there
    # will ever be, silencing a later, strictly worse (d, n). The id therefore
    # carries (d, n), and the policy that follows is identity, not severity:
    # EVERY distinct pair warns once, including one that is unsafe by less than
    # a pair already reported. n = 2 last pins that half -- its accumulator is
    # the smallest of the three and it still speaks, which is what separates
    # this policy from "only a worse pair warns". Asserted in one logger so the
    # suppression is real rather than reset between blocks.
    @testset "suppression is per (d, n), not per call site" begin
        logger = Test.TestLogger(; respect_maxlog=true)
        Base.CoreLogging.with_logger(logger) do
            StabilizerTableau(over, 4; state=:mixed, inversemod=jit)
            StabilizerTableau(over, 4; state=:mixed, inversemod=jit)  # repeat: silent
            StabilizerTableau(over, 8; state=:mixed, inversemod=jit)  # new pair: warns
            StabilizerTableau(over, 2; state=:mixed, inversemod=jit)  # less unsafe: warns
        end
        warns = [r for r in logger.logs if r.level == Base.CoreLogging.Warn]
        @test length(warns) == 3
        @test [r.kwargs[:n] for r in warns] == [4, 8, 2]
        @test all(r -> r.kwargs[:d] == over, warns)
    end
end

@testset "Modular inversion strategies" begin
    @test_throws ArgumentError QuditClifford.PrecomputedInvMod(4)

    # Integer tables are converted to Vector{Int}; non-integer ones are
    # rejected, so the error names the actual mistake. An accepted Float64 table
    # would behave differently per dimension: fine at d = 2, which never takes
    # the odd-d phase branch, but a MethodError from inside binom2_mod_oddprime
    # at every odd prime.
    @test_throws ArgumentError QuditClifford.PrecomputedInvMod([1.0])
    @test_throws ArgumentError QuditClifford.PrecomputedInvMod([1 // 1])
    # A table given in any integer type is converted to Vector{Int} on
    # construction, so nothing downstream ever sees another type. Pinned by
    # storage type, not just by value.
    for tbl in (Int[1, 2], Int32[1, 2], UInt64[1, 2])
        p = QuditClifford.PrecomputedInvMod(tbl)
        @test p.lookuptable isa Vector{Int}
        @test p(2, 3) === 2
    end
    @test QuditClifford.PrecomputedInvMod(Int32(5)).lookuptable isa Vector{Int}

    precomputed = QuditClifford.PrecomputedInvMod(Int[1, 2])
    just_in_time = QuditClifford.JustInTimeInvMod()
    @test precomputed(1, 3) == 1
    @test precomputed(2, 3) == 2
    @test just_in_time(2, 3) == 2

    tab = StabilizerTableau(3, 2; state=:ghz, inversemod=just_in_time)
    canonicalize!(tab)
    @test tab.iscanonical
    @test is_pure(tab; verify=true)
end

@testset "Tableau metadata and display" begin
    mixed = StabilizerTableau(2, 2; state=:mixed)
    @test QuditClifford.ngens(mixed) == 0
    @test QuditClifford.is_mixed(mixed)
    reset!(mixed; state=:product, basis=:Z)
    @test QuditClifford.ngens(mixed) == 2
    @test !QuditClifford.is_mixed(mixed)

    stab_mixed_text = sprint(show, StabilizerTableau(2, 2; state=:mixed))
    @test occursin("Stabilizer Tableau:", stab_mixed_text)
    @test occursin("no generators", stab_mixed_text)

    stab_one_text = sprint(show, StabilizerTableau(2, 1; state=:product, basis=:Z))
    @test occursin("    Tableau:", stab_one_text)
    @test occursin("X", stab_one_text)
    @test occursin("Z", stab_one_text)

    stab_two_text = sprint(show, StabilizerTableau(2, 2; state=:product, basis=:Z))
    @test occursin(" X     Z ", stab_two_text)

    partial = reshape(Int[10, 0, 0, 0, 0, 0, 0], 7, 1)
    stab_partial_text = sprint(show, StabilizerTableau(11, partial))
    @test occursin("Mixed tableau", stab_partial_text)
    @test occursin("10", stab_partial_text)

    stab_large_text = sprint(show, StabilizerTableau(2, 30; state=:mixed))
    @test occursin("too large to display", stab_large_text)

    destab_mixed_text = sprint(show, DestabilizerTableau(2, 2; state=:mixed))
    @test occursin("Destabilizer Tableau:", destab_mixed_text)
    @test occursin("no generators", destab_mixed_text)

    destab_one_text = sprint(show, DestabilizerTableau(2, 1; state=:product, basis=:Z))
    @test occursin("Stabilizers:", destab_one_text)
    @test occursin("Destabilizers:", destab_one_text)

    destab_two_text = sprint(show, DestabilizerTableau(2, 2; state=:product, basis=:Z))
    @test occursin(" X     Z ", destab_two_text)
    @test occursin("Destabilizers:", destab_two_text)

    destab_partial_text = sprint(show, DestabilizerTableau(11, partial))
    @test occursin("Mixed tableau", destab_partial_text)
    @test occursin("10", destab_partial_text)

    destab_large_text = sprint(show, DestabilizerTableau(2, 30; state=:mixed))
    @test occursin("too large to display", destab_large_text)
end

# Preset construction writes values that are already reduced, so it needs no
# separate mod pass over the (2n + storephase) × n matrix. Nothing else
# guarantees that: the GHZ generators are the natural place to write a literal
# -1, and only an assertion like this catches it once the pass is gone.
@testset "Preset tableaux are built already reduced" begin
    for TT in (StabilizerTableau, DestabilizerTableau),
        d in (2, 3, 5, 7), n in (1, 3, 4), storephase in (true, false)

        specs = Any[(:mixed, :Z), (:ghz, :Z),
                    (:product, :X), (:product, :Y), (:product, :Z)]
        n >= 3 && push!(specs, (:product, [:X, :Y, :Z][mod1.(1:n, 3)]))

        for (state, basis) in specs
            fresh = TT(d, n; state=state, basis=basis, storephase=storephase)
            after_reset = reset!(TT(d, n; state=:mixed, storephase=storephase);
                                 state=state, basis=basis)

            for tab in (fresh, after_reset)
                @test all(v -> 0 <= v < d, @view tab.stab[1:(2n), :])
                if storephase
                    dphase = QuditClifford.phase_modulus(d)
                    @test all(v -> 0 <= v < dphase, @view tab.stab[2n + 1, :])
                end
            end
        end
    end
end
