using QuditClifford
using Test
using Random

@testset "Purity checks" begin
    # Z1, Z2 -- commuting, independent, m == n.
    pure_mat = zeros(Int, 5, 2); pure_mat[3, 1] = 1; pure_mat[4, 2] = 1
    # X1, Z1 -- anticommuting.
    noncomm_mat = zeros(Int, 5, 2); noncomm_mat[1, 1] = 1; noncomm_mat[3, 2] = 1
    # Z1, Z1 -- commuting but rank 1.
    dep_mat = zeros(Int, 5, 2); dep_mat[3, 1] = 1; dep_mat[3, 2] = 1

    for (label, TT) in [("StabilizerTableau", StabilizerTableau),
                        ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            for d in (2, 3)
                @testset "d=$d" begin
                    # Each tableau is rebuilt per assertion: `is_commuting` and
                    # `is_independent` both consume `workspace`, so sharing one
                    # would make the tests depend on the order they run in.
                    mk(mat; kw...) = TT(d, mat; m=2, storephase=true, kw...)

                    @test QuditClifford.is_commuting(mk(pure_mat))
                    @test QuditClifford.is_independent(mk(pure_mat))
                    @test is_pure(mk(pure_mat))
                    @test is_pure(mk(pure_mat); verify=true)

                    # The invalid pair only exist behind check=false now; the
                    # constructor rejecting them is asserted separately below.
                    @test !QuditClifford.is_commuting(mk(noncomm_mat; check=false))
                    @test !is_pure(mk(noncomm_mat; check=false); verify=true)

                    @test QuditClifford.is_commuting(mk(dep_mat; check=false))
                    @test !QuditClifford.is_independent(mk(dep_mat; check=false))
                    @test !is_pure(mk(dep_mat; check=false); verify=true)

                    # m < n is mixed, whichever way it is asked.
                    mixed = TT(d, 2; state=:mixed)
                    @test !is_pure(mixed)
                    @test !is_pure(mixed; verify=true)
                end
            end
        end
    end
end

# The symplectic Gram kernel behind `is_commuting`. Tested directly on raw
# matrices rather than through a tableau so it stays reachable once the
# constructors start rejecting non-commuting input.
#
# `_first_noncommuting_pair` computes P = X'Z once and reads the symplectic
# form off it as P[j,i] - P[i,j], which relies on the form being
# antisymmetric. A reference written the other way round -- summing
# x_j*z_i - z_j*x_i directly, per pair -- is what pins that identity.
@testset "First non-commuting pair" begin
    # Independent reference: the definition, per pair, no Gram matrix.
    function reference_pair(stab, n, m, d)
        for j in 1:m, i in (j+1):m
            s = 0
            for q in 1:n
                s += stab[q, j] * stab[n+q, i] - stab[n+q, j] * stab[q, i]
            end
            mod(s, d) != 0 && return (j, i)
        end
        return (0, 0)
    end

    kernel(stab, n, m, d) = QuditClifford._first_noncommuting_pair(
        stab, n, m, d, zeros(Int, max(m, 1), max(m, 1)))

    @testset "agrees with the definition on random tableaux" begin
        rng = MersenneTwister(20260915)
        for d in (2, 3, 5, 7), n in (1, 2, 3, 5, 9), m in 0:n
            for _ in 1:12
                stab = rand(rng, 0:(d-1), 2n, max(m, 1))
                @test kernel(stab, n, m, d) == reference_pair(stab, n, m, d)
            end
        end
    end

    # Random tableaux are almost never commuting, so on their own they would
    # never exercise the (0,0) return. These do.
    @testset "commuting tableaux report no pair" begin
        for d in (2, 3, 5), n in (1, 2, 4, 7)
            # All-Z generators pairwise commute: every x entry is zero.
            zonly = zeros(Int, 2n, n)
            for j in 1:n
                zonly[n+j, j] = 1
            end
            @test kernel(zonly, n, n, d) == (0, 0)

            # A genuine state's generators, reached through the package.
            tab = StabilizerTableau(d, n; state = :ghz)
            @test kernel(tab.stab, n, tab.m, d) == (0, 0)
        end
    end

    @testset "reports the first pair in column order" begin
        # X1 and Z1 anticommute; everything else here commutes, so (2,4) is
        # the answer only if the scan really is ordered by j and then i.
        # Column 1 is deliberately Z2 rather than Z1 -- Z1 would anticommute
        # with the X1 in column 2 and make (1,2) the first pair instead.
        d, n = 3, 2
        stab = zeros(Int, 2n, 4)
        stab[n+2, 1] = 1   # Z2
        stab[1, 2] = 1     # X1
        stab[n+2, 3] = 1   # Z2
        stab[n+1, 4] = 1   # Z1
        @test reference_pair(stab, n, 4, d) == (2, 4)   # the fixture is what I claim
        @test kernel(stab, n, 4, d) == (2, 4)
    end

    @testset "m <= 1 has no pairs to check" begin
        for d in (2, 3), n in (1, 3)
            stab = rand(MersenneTwister(1), 0:(d-1), 2n, 1)
            @test kernel(stab, n, 0, d) == (0, 0)
            @test kernel(stab, n, 1, d) == (0, 0)
        end
    end
end

# The generator contract -- pairwise commuting, and independent -- is stated in
# both constructor docstrings but was never enforced, so an invalid matrix
# produced a tableau that every later operation silently believed. It is
# checked once at construction now, which is what lets `is_pure` trust `m == n`
# instead of re-deriving the contract on every call.
@testset "Raw-matrix construction validates the generator contract" begin
    # X1 and Z1 anticommute.
    noncomm = zeros(Int, 5, 2); noncomm[1, 1] = 1; noncomm[3, 2] = 1
    # Z1 twice: commuting, but rank 1 with m = 2.
    dependent = zeros(Int, 5, 2); dependent[3, 1] = 1; dependent[3, 2] = 1
    # Z1, Z2: the valid control, so a throw means the input and not the check.
    valid = zeros(Int, 5, 2); valid[3, 1] = 1; valid[4, 2] = 1

    for (label, TT) in [("StabilizerTableau", StabilizerTableau),
                        ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            for d in (2, 3)
                @testset "d=$d" begin
                    @test TT(d, valid; m=2, storephase=true) isa TT

                    @test_throws ArgumentError TT(d, noncomm; m=2, storephase=true)
                    @test_throws ArgumentError TT(d, dependent; m=2, storephase=true)

                    # The two failures must be distinguishable, or the message
                    # cannot tell a user which half of the contract they broke.
                    msg_nc = try; TT(d, noncomm; m=2, storephase=true); "" catch e; e.msg end
                    msg_dep = try; TT(d, dependent; m=2, storephase=true); "" catch e; e.msg end
                    @test occursin("commut", msg_nc)
                    @test occursin("independent", msg_dep)
                    @test msg_nc != msg_dep

                    # check=false is the documented escape hatch.
                    @test TT(d, noncomm; m=2, storephase=true, check=false) isa TT
                    @test TT(d, dependent; m=2, storephase=true, check=false) isa TT

                    # Presets satisfy the contract by construction, so the
                    # (d, n) path must never pay for it or trip over it.
                    for st in (:mixed, :product, :ghz)
                        @test TT(d, 4; state=st) isa TT
                    end
                end
            end
        end
    end
end

@testset "is_pure trusts the invariant; verify=true re-derives it" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau),
                        ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            for d in (2, 3)
                # On any tableau the package itself produced, the cheap form
                # and the verifying form must agree -- that agreement is the
                # whole licence for the cheap form.
                for n in (1, 2, 4, 6), st in (:mixed, :product, :ghz)
                    tab = TT(d, n; state=st)
                    @test is_pure(tab) == is_pure(tab; verify=true)
                end

                # ... and it must keep agreeing after measurements, which is
                # where the invariant would break if measure! ever stopped
                # preserving it.
                rng = MersenneTwister(hash((label, d)))
                tab = TT(d, 6; state=:product)
                for _ in 1:60
                    measure!(tab, SinglePauli(rand(rng, 1:6), rand(rng, 0:d-1),
                                              rand(rng, 0:d-1)); phase_policy=2)
                    @test is_pure(tab) == is_pure(tab; verify=true)
                end

                # Only a tableau built behind check=false can separate them.
                noncomm = zeros(Int, 5, 2); noncomm[1, 1] = 1; noncomm[3, 2] = 1
                bad = TT(d, noncomm; m=2, storephase=true, check=false)
                @test is_pure(bad)                    # m == n, taken on trust
                @test !is_pure(bad; verify=true)      # the contract is broken
            end
        end
    end
end
