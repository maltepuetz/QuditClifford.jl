using QuditClifford
using Test
using Random

const TCE_TABLEAU_TYPES = [("StabilizerTableau", StabilizerTableau),
                           ("DestabilizerTableau", DestabilizerTableau)]

# The active generator columns, phase row included when stored.
tce_active(tab) = tab.stab[:, 1:tab.m]

# Every field of a tableau as a tuple of independent copies, for checking that
# an operation left its argument untouched. The inversion strategy is never
# mutated and has no `==`, so it is recorded by identity.
tce_snapshot(tab) = Tuple(f === :inversemod ? objectid(tab.inversemod) :
                          deepcopy(getfield(tab, f)) for f in fieldnames(typeof(tab)))

function tce_no_shared_arrays(a, b)
    for f in fieldnames(typeof(a))
        x, y = getfield(a, f), getfield(b, f)
        x isa Array && x === y && return false
    end
    return true
end

# Duality ⟨D_j, S_k⟩ = δ_jk and the x·z cache of a DestabilizerTableau.
function tce_dual_ok(tab::DestabilizerTableau)
    n, d, m = tab.n, tab.d, tab.m
    for j in 1:m, k in 1:m
        s = 0
        for q in 1:n
            s += tab.destab[q, j] * tab.stab[n + q, k] - tab.destab[n + q, j] * tab.stab[q, k]
        end
        mod(s, d) == (j == k ? 1 : 0) || return false
    end
    for j in 1:m
        x = sum(tab.stab[q, j] * tab.stab[n + q, j] for q in 1:n; init = 0)
        mod(x, d) == tab.xdotz_cache[j] || return false
    end
    return true
end
tce_dual_ok(::StabilizerTableau) = true

# A random state reached through measurements and Cliffords, so the generator
# basis is neither preset nor canonical.
function tce_scrambled(TT, rng, d, n; m = n, storephase = true)
    tab = TT(d, n; storephase = storephase)
    random_state!(rng, tab; m = m)
    for _ in 1:3
        apply!(tab, random_clifford(rng, d, randperm(rng, n)[1:min(2, n)]))
    end
    return tab
end

@testset "Tableau copy and equality" begin
    for (label, TT) in TCE_TABLEAU_TYPES
        @testset "$label" begin
            for d in (2, 3)
                @testset "d=$d" begin
                    rng = Xoshiro(17 + d)
                    n = 4

                    @testset "copy is an independent equal tableau" begin
                        for storephase in (true, false), m in (0, 2, n)
                            tab = tce_scrambled(TT, rng, d, n; m = m, storephase = storephase)
                            before = tce_snapshot(tab)
                            c = copy(tab)
                            @test typeof(c) === typeof(tab)
                            @test tce_snapshot(tab) == before
                            @test tce_no_shared_arrays(c, tab)
                            @test (c.d, c.n, c.m, c.storephase, c.iscanonical) ==
                                  (tab.d, tab.n, tab.m, tab.storephase, tab.iscanonical)
                            @test c.stab == tab.stab
                            @test c.xdotz_cache == tab.xdotz_cache
                            TT === DestabilizerTableau && @test c.destab == tab.destab
                            @test tce_dual_ok(c)
                            @test c == tab

                            # Driving the copy leaves the original alone.
                            measure!(c, SinglePauli(1, 1, 0); phase_policy = 2)
                            measure!(c, SinglePauli(2, 0, 1); phase_policy = 2)
                            apply!(c, Fourier(3))
                            @test tce_snapshot(tab) == before
                            @test tce_dual_ok(c)
                        end
                    end

                    @testset "copy keeps the inversion strategy" begin
                        tab = TT(d, 3, :ghz; inversemod = QuditClifford.JustInTimeInvMod())
                        @test copy(tab).inversemod isa QuditClifford.JustInTimeInvMod
                        canonicalize!(tab)
                        c = copy(tab)
                        @test c.iscanonical
                        @test c.pivcol_of_row == tab.pivcol_of_row
                    end

                    @testset "copy! overwrites in place" begin
                        for storephase in (true, false)
                            src = tce_scrambled(TT, rng, d, n; m = 3, storephase = storephase)
                            dst = TT(d, n, :ghz; storephase = storephase)
                            dst_stab = dst.stab
                            before = tce_snapshot(src)
                            @test copy!(dst, src) === dst
                            @test dst.stab === dst_stab
                            @test tce_snapshot(src) == before
                            @test dst == src
                            @test (dst.m, dst.iscanonical) == (src.m, src.iscanonical)
                            @test tce_active(dst) == tce_active(src)
                            @test tce_dual_ok(dst)

                            measure!(dst, SinglePauli(1, 0, 1); phase_policy = 2)
                            @test tce_snapshot(src) == before

                            copy!(dst, src)
                            @test @allocated(copy!(dst, src)) == 0
                        end
                    end

                    @testset "copy! rejects a mismatched destination" begin
                        src = TT(d, 3, :ghz)
                        dst_d = d == 2 ? 3 : 2
                        for dst in (TT(dst_d, 3), TT(d, 4), TT(d, 3; storephase = false))
                            before = tce_snapshot(dst)
                            @test_throws ArgumentError copy!(dst, src)
                            @test tce_snapshot(dst) == before
                        end
                    end

                    @testset "== compares states, not bases" begin
                        for storephase in (true, false), m in (0, 1, 3, n)
                            tab = tce_scrambled(TT, rng, d, n; m = m, storephase = storephase)
                            # A column permutation is the same state in another basis.
                            perm = randperm(rng, m)
                            M = tce_active(tab)[:, perm]
                            permuted = TT(d, M; m = m, storephase = storephase)
                            canonical = copy(tab)
                            canonicalize!(canonical)

                            for other in (permuted, canonical)
                                @test tab == other
                                @test other == tab
                                @test isequal(tab, other)
                                @test hash(tab) == hash(other)
                            end
                            m > 1 && @test tce_active(canonical) != tce_active(tab)
                        end
                    end

                    @testset "== does not modify its arguments" begin
                        a = tce_scrambled(TT, rng, d, n)
                        b = copy(a)
                        canonicalize!(b)
                        a_before, b_before = tce_snapshot(a), tce_snapshot(b)
                        @test a == b
                        hash(a)
                        @test tce_snapshot(a) == a_before
                        @test tce_snapshot(b) == b_before
                        @test !a.iscanonical
                    end

                    @testset "== distinguishes different states" begin
                        ghz = TT(d, n, :ghz)
                        @test ghz != TT(d, n, :product)
                        @test ghz != TT(d, n, :mixed)
                        @test ghz != TT(d, n + 1, :ghz)
                        @test ghz != TT(d == 2 ? 3 : 2, n, :ghz)
                        @test ghz != TT(d, n, :ghz; storephase = false)
                        @test TT(d, n, :mixed) != TT(d, n + 1, :mixed)

                        # Same stabilizer group up to phases: shift one phase by
                        # a Hermiticity-preserving amount.
                        tab = tce_scrambled(TT, rng, d, n)
                        M = tce_active(tab)
                        M[end, 2] = mod(M[end, 2] + (d == 2 ? 2 : 1), d == 2 ? 4 : d)
                        shifted = TT(d, M)
                        @test tab != shifted
                        @test shifted != tab

                        # Without a phase row the same group is equal.
                        plain_a = TT(d, tce_active(tab)[1:2n, :]; storephase = false)
                        plain_b = TT(d, M[1:2n, :]; storephase = false)
                        @test plain_a == plain_b
                        @test hash(plain_a) == hash(plain_b)

                        # Distinct mixed states with the same number of generators.
                        z1 = TT(d, reshape([0, 0, 1, 0, 0], 5, 1))
                        z2 = TT(d, reshape([0, 0, 0, 1, 0], 5, 1))
                        @test (z1.n, z1.m, z2.n, z2.m) == (2, 1, 2, 1)
                        @test z1 != z2
                    end

                    @testset "== across tableau types" begin
                        other_TT = TT === StabilizerTableau ? DestabilizerTableau : StabilizerTableau
                        tab = tce_scrambled(TT, rng, d, n; m = 3)
                        twin = other_TT(d, tce_active(tab); m = 3)
                        canonicalize!(twin)
                        @test tce_active(twin) != tce_active(tab)
                        @test tab == twin
                        @test hash(tab) == hash(twin)
                        @test TT(d, n, :ghz) == other_TT(d, n, :ghz)
                    end

                    @testset "== agrees with density-matrix equality" begin
                        # Small enough that random draws often coincide.
                        pool = [tce_scrambled(TT, rng, d, 2; m = rand(rng, 0:2)) for _ in 1:40]
                        rhos = [density_matrix(t) for t in pool]
                        hits = 0
                        for i in eachindex(pool), j in eachindex(pool)
                            same = isapprox(rhos[i], rhos[j]; atol = 1e-10)
                            hits += same && i != j
                            @test (pool[i] == pool[j]) == same
                            same && @test hash(pool[i]) == hash(pool[j])
                        end
                        @test hits > 0
                    end

                    @testset "Equal states collapse in a Set" begin
                        a = tce_scrambled(TT, rng, d, n)
                        b = copy(a); canonicalize!(b)
                        c = TT(d, n, :product)
                        @test length(Set([a, b, c, copy(c)])) == 2
                    end
                end
            end
        end
    end
end
