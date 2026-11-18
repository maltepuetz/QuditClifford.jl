using QuditClifford
using Test
using Random

# The rank-based quantities eliminate in a UInt8 workspace for d <= 255 and on
# bit-packed words for d = 2, and fall back to the Int workspace above 255.
# Everything here compares them with a plain reference that reduces with `mod`
# after every operation and never touches the tableau's workspaces.

const RW = QuditClifford

# Column echelon reduction of an Int copy, pivoting on rows 1:pivot_rows.
# Returns the rank and the reduced matrix: columns rank+1:end are zero on the
# pivot rows and span the combinations of the inputs that vanish there.
function rw_colreduce(A0::AbstractMatrix, d::Int, pivot_rows::Int=size(A0, 1))
    A = mod.(Matrix{Int}(A0), d)
    nr, nc = size(A)
    r = c = 1
    while r <= pivot_rows && c <= nc
        j = findfirst(jj -> A[r, jj] != 0, c:nc)
        if j === nothing
            r += 1
            continue
        end
        j += c - 1
        A[:, [c, j]] = A[:, [j, c]]
        A[:, c] = mod.(A[:, c] .* invmod(A[r, c], d), d)
        for jj in (c + 1):nc
            f = A[r, jj]
            f == 0 && continue
            A[:, jj] = mod.(A[:, jj] .- f .* A[:, c], d)
        end
        r += 1
        c += 1
    end
    return c - 1, A
end

rw_rank(A, d) = rw_colreduce(A, d)[1]

# Rows of the X and Z exponents of `sites` in the active generator columns.
rw_restrict(tab, sites) = tab.stab[vcat(collect(sites), tab.n .+ collect(sites)), 1:tab.m]

# S(A) = |A| - m + rank of the generators restricted to the complement of A.
function rw_entropy(tab, A)
    B = setdiff(1:tab.n, A)
    rB = isempty(B) || tab.m == 0 ? 0 : rw_rank(rw_restrict(tab, B), tab.d)
    return length(A) - tab.m + rB
end

# ℰ(A:B) = rank(K)/2 with K the symplectic products on A of a basis of the
# stabilizers supported on A ∪ B.
function rw_negativity(tab, A, B)
    (isempty(A) || isempty(B) || tab.m == 0) && return 0
    n, m, d = tab.n, tab.m, tab.d
    C = setdiff(1:n, A, B)
    G = tab.stab[1:(2n), 1:m]
    rows_C = vcat(C, n .+ C)
    piv, R = rw_colreduce(vcat(G[rows_C, :], G), d, length(rows_C))
    H = R[(length(rows_C) + 1):end, (piv + 1):m]
    k = size(H, 2)
    k <= 1 && return 0
    X = H[A, :]
    Z = H[n .+ A, :]
    K = mod.(transpose(X) * Z .- transpose(Z) * X, d)
    return rw_rank(K, d) ÷ 2
end

# A pure or mixed state that is not uniformly random: brickwork two-qudit
# Cliffords with interleaved Z measurements, starting from a mixed state with
# m0 generators, so the generators stay local.
function rw_local_state(rng, TT, d, n, m0; layers=3)
    tab = TT(d, n)
    random_state!(rng, tab; m=m0)
    for layer in 1:layers
        for q in (isodd(layer) ? 1 : 2):2:(n - 1)
            apply!(tab, random_clifford(rng, d, [q, q + 1]))
        end
        for q in 1:n
            rand(rng) < 0.3 && measure!(rng, tab, SinglePauli(q, 0, 1); phase_policy=2)
        end
    end
    return tab
end

rw_subsets(rng, n) = unique([Int[], collect(1:n), [1], [n], collect(1:(n ÷ 2)),
                             sort(randperm(rng, n)[1:rand(rng, 0:n)]),
                             sort(randperm(rng, n)[1:rand(rng, 0:n)])])

@testset "Narrow and bit-packed rank workspaces" begin
    @testset "Workspace shapes" begin
        for TT in (StabilizerTableau, DestabilizerTableau), n in (1, 5, 64, 65)
            t2 = TT(2, n)
            @test size(t2.narrow_workspace) == (2n, n)
            @test size(t2.bit_workspace) == (cld(2n, 64), n)
            t3 = TT(251, n)
            @test size(t3.narrow_workspace) == (2n, n)
            @test size(t3.bit_workspace) == (0, 0)
            t4 = TT(257, n)
            @test size(t4.narrow_workspace) == (0, 0)
            @test size(t4.bit_workspace) == (0, 0)
            c = copy(t2)
            @test size(c.narrow_workspace) == size(t2.narrow_workspace)
            @test c.narrow_workspace !== t2.narrow_workspace
            @test c.bit_workspace !== t2.bit_workspace
        end
    end

    @testset "UInt8 update is exact for every prime d <= 255 and multiplier" begin
        rng = Xoshiro(20261118)
        for d in 2:255
            all(d % p != 0 for p in 2:isqrt(d)) || continue
            s = UInt8.(repeat(0:(d - 1), 3))
            for a in 1:(d - 1)
                x = rand(rng, UInt8(0):UInt8(d - 1), length(s))
                expected = UInt8.(mod.(Int.(x) .- a .* Int.(s), d))
                @test RW._submul_u8!(copy(x), s, a, d) == expected
            end
        end
    end

    @testset "rank_fp_cols! agrees across element types, echelon form included" begin
        rng = Xoshiro(7)
        for d in (2, 3, 5, 7, 31, 127, 131, 251), (rows, cols) in ((1, 1), (6, 4), (9, 13), (40, 33))
            for trial in 1:6
                inner = rand(rng, 1:min(rows, cols))
                A = trial <= 3 ? rand(rng, 0:(d - 1), rows, cols) :
                    mod.(rand(rng, 0:(d - 1), rows, inner) * rand(rng, 0:(d - 1), inner, cols), d)
                for pivot_rows in unique((rows, rows ÷ 2, 0))
                    ref, R = rw_colreduce(A, d, pivot_rows)
                    AI = copy(A)
                    A8 = UInt8.(A)
                    inv = RW.PrecomputedInvMod(d)
                    @test RW.rank_fp_cols!(AI, d, inv, pivot_rows) == ref
                    @test RW.rank_fp_cols!(A8, d, inv, pivot_rows) == ref
                    # The columns the callers keep reading are the same for both
                    # element types and for the reference.
                    @test AI[:, (ref + 1):end] == Int.(A8[:, (ref + 1):end]) == R[:, (ref + 1):end]
                end
            end
        end
        @test_throws ArgumentError RW.rank_fp_cols!(zeros(UInt8, 2, 2), 257, RW.PrecomputedInvMod(257))
    end

    @testset "GF(2) packing and rank" begin
        rng = Xoshiro(11)
        for n in (1, 2, 31, 32, 33, 64, 70), m in unique((0, 1, n ÷ 2, n))
            tab = StabilizerTableau(2, n)
            random_state!(rng, tab; m)
            for sites in (collect(1:n), sort(randperm(rng, n)[1:rand(rng, 1:n)]))
                N = length(sites)
                W = zeros(UInt64, cld(2n, 64), max(m, 1))
                fill!(W, typemax(UInt64))
                nw = RW._pack_sites_gf2!(W, tab.stab, n, m, sites, N)
                @test nw == cld(2N, 64)
                for j in 1:m, k in 1:N
                    q = sites[k]
                    @test (W[(2k - 2) >> 6 + 1, j] >> ((2k - 2) & 63)) & 1 == tab.stab[q, j]
                    @test (W[(2k - 1) >> 6 + 1, j] >> ((2k - 1) & 63)) & 1 == tab.stab[n + q, j]
                end
                # bits past the last row are cleared
                m > 0 && 2N % 64 != 0 && @test W[nw, 1] >> (2N % 64) == 0
                @test RW._rank_gf2_cols!(view(W, 1:nw, 1:m), 2N) ==
                      (m == 0 ? 0 : rw_rank(rw_restrict(tab, sites), 2))
            end
        end
    end

    # 257 is the smallest prime above the UInt8 range and takes the Int path;
    # 131 is not accepted by the Barrett tier of src/modular.jl.
    @testset "$(nameof(TT)), d=$d" for TT in (StabilizerTableau, DestabilizerTableau),
                                       d in (2, 3, 5, 31, 127, 131, 251, 257)
        rng = Xoshiro(hash((TT, d)) % UInt)
        for n in (1, 2, 5, 9, 33, 70)
            d > 31 && n > 33 && continue
            states = Any[]
            for m in unique((0, 1, n ÷ 2, n - 1, n))
                tab = TT(d, n)
                random_state!(rng, tab; m)
                push!(states, tab)
            end
            n >= 2 && push!(states, rw_local_state(rng, TT, d, n, 0),
                            rw_local_state(rng, TT, d, n, n))
            push!(states, TT(d, n; state=:ghz), TT(d, n; state=:product, basis=:X))

            for tab in states
                n, m = tab.n, tab.m
                before = copy(tab.stab)

                # entanglement_entropy on assorted subsystems, with and without
                # the validity check, and the empty and full subsystems.
                for A in rw_subsets(rng, n)
                    ref = rw_entropy(tab, A)
                    @test entanglement_entropy(tab, A) == ref
                    @test entanglement_entropy(tab, A; check=false) == ref
                end

                # profiles, natural and permuted
                @test entanglement_profile(tab) == [rw_entropy(tab, 1:l) for l in 0:n]
                order = randperm(rng, n)
                @test entanglement_profile(tab, order) ==
                      [rw_entropy(tab, order[1:l]) for l in 0:n]

                # information measures on random disjoint subsystems
                perm = randperm(rng, n)
                cuts = sort(rand(rng, 0:n, 3))
                A = perm[1:cuts[1]]
                B = perm[(cuts[1] + 1):cuts[2]]
                C = perm[(cuts[2] + 1):cuts[3]]
                S(X) = rw_entropy(tab, X)
                @test mutual_information(tab, A, B) ==
                      (isempty(A) || isempty(B) ? 0 : S(A) + S(B) - S(vcat(A, B)))
                @test mutual_information(tab, A) ==
                      (isempty(A) || length(A) == n ? 0 : S(A) + S(setdiff(1:n, A)) - S(1:n))
                @test conditional_mutual_information(tab, A, B, C) ==
                      (isempty(A) || isempty(B) ? 0 :
                       S(vcat(A, C)) + S(vcat(B, C)) - S(C) - S(vcat(A, B, C)))
                @test tripartite_information(tab, A, B, C) ==
                      (isempty(A) || isempty(B) || isempty(C) ? 0 :
                       S(A) + S(B) + S(C) - S(vcat(A, B)) - S(vcat(A, C)) - S(vcat(B, C)) +
                       S(vcat(A, B, C)))
                @test entanglement_negativity(tab, A, B) == rw_negativity(tab, A, B)
                @test entanglement_negativity(tab, A) ==
                      rw_negativity(tab, A, setdiff(1:n, A))

                @test RW._generator_rank(tab) == m
                @test is_pure(tab; verify=true) == (m == n)
                @test tab.stab == before
            end
        end
    end

    @testset "A subsystem stored in a rank workspace is copied first" begin
        for d in (2, 3)
            tab = StabilizerTableau(d, 6)
            random_state!(Xoshiro(3), tab; m=6)
            expected = entanglement_entropy(tab, [1, 2])
            # A 2-element view into the UInt8 workspace, filled with sites 1, 2.
            ws = tab.narrow_workspace
            ws[1, 1] = 1
            ws[2, 1] = 2
            @test entanglement_entropy(tab, view(ws, 1:2, 1)) == expected
            @test entanglement_profile(tab, view(ws, 1:6, 1) .= 1:6) ==
                  entanglement_profile(tab)
        end
    end

    @testset "No allocations once compiled" begin
        rw_alloc_entropy(tab, A) = @allocated entanglement_entropy(tab, A)
        rw_alloc_profile(dest, tab) = @allocated entanglement_profile!(dest, tab)
        rw_alloc_rank(tab) = @allocated RW._generator_rank(tab)
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31, 257), m in (4, 8)
            tab = TT(d, 8)
            random_state!(Xoshiro(5), tab; m)
            A = [1, 3, 4]
            dest = zeros(Int, 9)
            rw_alloc_entropy(tab, A)
            rw_alloc_profile(dest, tab)
            rw_alloc_rank(tab)
            @test rw_alloc_entropy(tab, A) == 0
            @test rw_alloc_profile(dest, tab) == 0
            @test rw_alloc_rank(tab) == 0
        end
    end
end
