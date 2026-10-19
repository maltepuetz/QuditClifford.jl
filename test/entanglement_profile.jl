using QuditClifford
using Test
using Random

# Warm each destination/tableau specialization before measuring allocations.
function _profile_allocations(dest, tab)
    entanglement_profile!(dest, tab)
    return @allocated entanglement_profile!(dest, tab)
end

struct _ZeroBasedProfileVector <: AbstractVector{Int}
    data::Vector{Int}
end
Base.size(v::_ZeroBasedProfileVector) = size(v.data)
Base.axes(v::_ZeroBasedProfileVector) = (0:(length(v.data) - 1),)
Base.getindex(v::_ZeroBasedProfileVector, i::Int) = v.data[i + 1]
Base.setindex!(v::_ZeroBasedProfileVector, x, i::Int) = (v.data[i + 1] = x)

@testset "Entanglement profile" begin
    @testset "Caller-owned output is overwritten and reused without allocation" begin
        rng = MersenneTwister(20261020)
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5, 7),
            storephase in (false, true), n in (0, 1, 8)
            tab = TT(d, n; storephase)
            dest = fill(-1, n + 1)
            backing = fill(-1, 2n + 3)
            strided = view(backing, 2:2:(2n + 2))
            for m in unique((0, n ÷ 2, n)), output in (dest, strided)
                random_state!(rng, tab; m)
                expected = [entanglement_entropy(tab, 1:l) for l in 0:n]
                fill!(output, -1)
                @test (@inferred entanglement_profile!(output, tab)) === output
                @test output == expected
                @test _profile_allocations(output, tab) == 0
                @test all(==(-1), view(backing, 1:2:length(backing)))
            end
            # Reuse a buffer filled by the general path on the m=0 fast path.
            reset!(tab; state=:mixed)
            @test entanglement_profile!(dest, tab) === dest
            @test dest == collect(0:n)
        end
    end

    @testset "Destination length is checked before any mutation" begin
        for TT in (StabilizerTableau, DestabilizerTableau), state in (:mixed, :ghz)
            tab = TT(3, 4; state)
            before = deepcopy(tab)
            for len in (0, 4, 6)
                dest = fill(-1, len)
                @test_throws DimensionMismatch entanglement_profile!(dest, tab)
                @test dest == fill(-1, len)
                @test tab.stab == before.stab
                @test tab.workspace == before.workspace
                TT === DestabilizerTableau && @test tab.destab == before.destab
            end
            offset = _ZeroBasedProfileVector(fill(-1, 5))
            @test_throws ArgumentError entanglement_profile!(offset, tab)
            @test offset.data == fill(-1, 5)
            @test tab.stab == before.stab
            @test tab.workspace == before.workspace
        end
    end

    @testset "Known profiles and empty boundaries" begin
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5, 7),
            storephase in (false, true)
            for n in (0, 1, 2, 7)
                product = TT(d, n; state=:product, storephase)
                @test entanglement_profile(product) == zeros(Int, n + 1)
                mixed = TT(d, n; storephase)
                @test entanglement_profile(mixed) == collect(0:n)
                if n > 0
                    ghz = TT(d, n; state=:ghz, storephase)
                    @test entanglement_profile(ghz) == [0; ones(Int, n - 1); 0]
                end
            end

            # Two independent local Paulis end at site 3. Both pivots must
            # count, and a generator ending earlier must survive elimination.
            pairs = TT(d, 5; state=:product, storephase)
            for (a, b) in ((1, 3), (2, 5))
                apply!(pairs, Fourier(a))
                apply!(pairs, SUM(a, b))
            end
            @test entanglement_profile(pairs) == [0, 1, 2, 1, 1, 0]

            # Purity equates a prefix with its complementary suffix, not
            # with the equally long prefix measured from the other end.
            asymmetric = TT(d, 3; state=:product, storephase)
            apply!(asymmetric, Fourier(1))
            apply!(asymmetric, SUM(1, 2))
            @test entanglement_profile(asymmetric) == [0, 1, 0, 0]

            # Mixed state with one fixed site in the middle: total entropy
            # is n-m, and need not return to zero at the final cut.
            partial = TT(d, 3; storephase)
            measure!(partial, SinglePauli(2, 0, 1); outcome=0)
            @test entanglement_profile(partial) == [0, 1, 1, 2]
        end
    end

    @testset "Every cut agrees with subsystem entropy" begin
        rng = MersenneTwister(20261019)
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5, 7),
            storephase in (false, true), n in (1, 2, 5, 9),
            m in unique((0, 1, n ÷ 2, n)), trial in 1:3
            tab = TT(d, n; storephase)
            random_state!(rng, tab; m)
            expected = [entanglement_entropy(tab, 1:l) for l in 0:n]
            profile = @inferred entanglement_profile(tab)
            @test profile isa Vector{Int}
            @test profile == expected
            dest = fill(-1, n + 1)
            @test entanglement_profile!(dest, tab) === dest
            @test dest == expected
            @test first(profile) == 0
            @test last(profile) == n - m
            if m == n
                @test profile == [entanglement_entropy(tab, (l + 1):n) for l in 0:n]
            end
        end
    end

    @testset "State, canonical caches and subsequent operations are preserved" begin
        rng = MersenneTwister(7123)
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5),
            storephase in (false, true), m in (0, 3, 6), canonical in (false, true)
            tab = TT(d, 6; storephase)
            random_state!(rng, tab; m)
            canonical && canonicalize!(tab)
            reference = deepcopy(tab)
            expected = [entanglement_entropy(reference, 1:l) for l in 0:6]

            # Scratch may contain residues from any previous operation,
            # including columns beyond the active generator count.
            fill!(tab.workspace, d - 1)
            fill!(tab.c_workspace, d - 1)
            profile = entanglement_profile(tab)
            @test profile == expected
            dest = fill(-1, 7)
            @test entanglement_profile!(dest, tab) === dest
            @test dest == expected
            @test tab.stab == reference.stab
            @test tab.m == reference.m
            @test tab.iscanonical == reference.iscanonical
            @test tab.pivcol_of_row == reference.pivcol_of_row
            @test tab.xdotz_cache == reference.xdotz_cache
            TT === DestabilizerTableau && @test tab.destab == reference.destab

            # Interleave callers of the shared workspaces, then measure on
            # both copies to catch stale scratch or damaged dual/cached data.
            @test entanglement_entropy(tab, [2, 4, 6]) ==
                  entanglement_entropy(reference, [2, 4, 6])
            op = SinglePauli(2, 1, 0)
            @test expect!(tab, op) == expect!(reference, op)
            @test entanglement_profile(tab) == profile
            measure!(tab, op; outcome=0)
            measure!(reference, op; outcome=0)
            @test tab.stab == reference.stab
            @test tab.m == reference.m
            TT === DestabilizerTableau && @test tab.destab == reference.destab
            @test entanglement_profile(tab) ==
                  [entanglement_entropy(reference, 1:l) for l in 0:6]
            @test profile == expected  # returned storage is owned by the caller
        end
    end

    @testset "Modular arithmetic tiers and inverse strategies" begin
        rng = MersenneTwister(443131)
        for TT in (StabilizerTableau, DestabilizerTableau), d in (5, 131, 443),
            inversemod in (QuditClifford.PrecomputedInvMod(d), QuditClifford.JustInTimeInvMod()),
            m in (3, 7)
            tab = TT(d, 7; inversemod)
            random_state!(rng, tab; m)
            @test entanglement_profile(tab) ==
                  [entanglement_entropy(tab, 1:l) for l in 0:7]
        end
    end
end
