using QuditClifford
using Test
using Random

# The rank-based quantities eliminate on bit-packed words for d = 2 and, for
# d >= 3, in a workspace of the tableau's storage element type.
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
rw_restrict(tab, sites) = stabilizer_matrix(tab)[vcat(collect(sites), tab.n .+ collect(sites)), 1:tab.m]

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
    G = stabilizer_matrix(tab)[1:(2n), 1:m]
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

# The reference values of the four information measures.
function rw_information(tab, A, B, C)
    n = tab.n
    S(X) = rw_entropy(tab, X)
    mi = isempty(A) || isempty(B) ? 0 : S(A) + S(B) - S(vcat(A, B))
    mic = isempty(A) || length(A) == n ? 0 : S(A) + S(setdiff(1:n, A)) - S(1:n)
    cmi = isempty(A) || isempty(B) ? 0 :
          S(vcat(A, C)) + S(vcat(B, C)) - S(C) - S(vcat(A, B, C))
    i3 = isempty(A) || isempty(B) || isempty(C) ? 0 :
         S(A) + S(B) + S(C) - S(vcat(A, B)) - S(vcat(A, C)) - S(vcat(B, C)) +
         S(vcat(A, B, C))
    return (mi, mic, cmi, i3, rw_negativity(tab, A, B), rw_negativity(tab, A, setdiff(1:n, A)))
end

rw_measured(tab, A, B, C) = (mutual_information(tab, A, B), mutual_information(tab, A),
                             conditional_mutual_information(tab, A, B, C),
                             tripartite_information(tab, A, B, C),
                             entanglement_negativity(tab, A, B), entanglement_negativity(tab, A))

# Pack the 0/1 rows of `A` into words, row i at bit i - 1 of each column.
function rw_pack_rows(A::AbstractMatrix)
    nr, nc = size(A)
    W = zeros(UInt64, cld(nr, 64), nc)
    for j in 1:nc, i in 1:nr
        W[(i - 1) >> 6 + 1, j] |= UInt64(A[i, j]) << ((i - 1) & 63)
    end
    return W
end

rw_unpack_rows(W, nr) = [Int((W[(i - 1) >> 6 + 1, j] >> ((i - 1) & 63)) & 1)
                         for i in 1:nr, j in 1:size(W, 2)]

rw_subsets(rng, n) = unique([Int[], collect(1:n), [1], [n], collect(1:(n ÷ 2)),
                             sort(randperm(rng, n)[1:rand(rng, 0:n)]),
                             sort(randperm(rng, n)[1:rand(rng, 0:n)])])

rw_types(d) = filter(E -> RW._eltype_admissible(E, d), (UInt8, UInt16, Int))

@testset "Narrow and bit-packed rank workspaces" begin
    @testset "Workspace shapes" begin
        for TT in (StabilizerTableau, DestabilizerTableau), n in (1, 5, 64, 65)
            for E in rw_types(2)
                t2 = TT(2, n; eltype = E)
                @test t2.rank_workspace isa Matrix{E}
                @test size(t2.rank_workspace) == (0, 0)
                @test size(t2.bit_workspace) == (cld(2n, 64) + 2, n)
                @test RW._rank_workspace(t2) === t2.bit_workspace
                c2 = copy(t2)
                @test size(c2.bit_workspace) == size(t2.bit_workspace)
                @test c2.bit_workspace !== t2.bit_workspace
            end
            for d in (251, 257), E in rw_types(d)
                t3 = TT(d, n; eltype = E)
                @test t3.rank_workspace isa Matrix{E}
                @test size(t3.rank_workspace) == (2n, n)
                @test size(t3.bit_workspace) == (0, 0)
                @test RW._rank_workspace(t3) === t3.rank_workspace
                c3 = copy(t3)
                @test size(c3.rank_workspace) == size(t3.rank_workspace)
                @test c3.rank_workspace !== t3.rank_workspace
            end
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
                @test RW.submul_mod!(copy(x), s, a, d) == expected
            end
        end
    end

    @testset "rank_fp_cols! agrees across element types, echelon form included" begin
        rng = Xoshiro(7)
        for d in (2, 3, 5, 7, 31, 127, 131, 251, 257), (rows, cols) in ((1, 1), (6, 4), (9, 13), (40, 33))
            for trial in 1:6
                inner = rand(rng, 1:min(rows, cols))
                A = trial <= 3 ? rand(rng, 0:(d - 1), rows, cols) :
                    mod.(rand(rng, 0:(d - 1), rows, inner) * rand(rng, 0:(d - 1), inner, cols), d)
                for pivot_rows in unique((rows, rows ÷ 2, 0))
                    ref, R = rw_colreduce(A, d, pivot_rows)
                    AI = copy(A)
                    inv = RW.PrecomputedInvMod(d)
                    @test RW.rank_fp_cols!(AI, d, inv, pivot_rows) == ref
                    # The columns the callers keep reading are the same for
                    # every element type and for the reference.
                    @test AI[:, (ref + 1):end] == R[:, (ref + 1):end]
                    for E in filter(!=(Int), rw_types(d))
                        AE = E.(A)
                        @test RW.rank_fp_cols!(AE, d, inv, pivot_rows) == ref
                        @test Int.(AE[:, (ref + 1):end]) == R[:, (ref + 1):end]
                    end
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
                nw = RW._pack_sites_gf2!(W, QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), m, sites, N)
                @test nw == cld(2N, 64)
                for j in 1:m, k in 1:N
                    q = sites[k]
                    @test (W[(2k - 2) >> 6 + 1, j] >> ((2k - 2) & 63)) & 1 == stabilizer_matrix(tab)[q, j]
                    @test (W[(2k - 1) >> 6 + 1, j] >> ((2k - 1) & 63)) & 1 == stabilizer_matrix(tab)[n + q, j]
                end
                # bits past the last row are cleared
                m > 0 && 2N % 64 != 0 && @test W[nw, 1] >> (2N % 64) == 0
                @test RW._rank_gf2_cols!(view(W, 1:nw, 1:m), 2N) ==
                      (m == 0 ? 0 : rw_rank(rw_restrict(tab, sites), 2))
            end
        end
    end

    @testset "GF(2) rank with a pivot row limit keeps the echelon contract" begin
        rng = Xoshiro(13)
        for (rows, cols) in ((1, 1), (7, 5), (64, 9), (65, 40), (130, 70), (200, 33))
            for trial in 1:6
                inner = rand(rng, 1:min(rows, cols))
                A = trial <= 3 ? rand(rng, 0:1, rows, cols) :
                    mod.(rand(rng, 0:1, rows, inner) * rand(rng, 0:1, inner, cols), 2)
                for pivot_rows in unique((rows, rows ÷ 2, 63, 64, 65, 1, 0))
                    pivot_rows <= rows || continue
                    ref, R = rw_colreduce(A, 2, pivot_rows)
                    W = rw_pack_rows(A)
                    @test RW._rank_gf2_cols!(W, pivot_rows) == ref
                    @test rw_unpack_rows(W, rows)[:, (ref + 1):end] == R[:, (ref + 1):end]
                end
            end
        end
        @test_throws DimensionMismatch RW._rank_gf2_cols!(zeros(UInt64, 1, 2), 65)
    end

    @testset "Bit-packed blocks start on a fresh word" begin
        rng = Xoshiro(17)
        n, m = 70, 50
        tab = StabilizerTableau(2, n)
        random_state!(rng, tab; m)
        W = fill(typemax(UInt64), cld(2n, 64) + 2, n)
        sites1 = sort(randperm(rng, n)[1:33])
        sites2 = sort(randperm(rng, n)[1:1])
        o1 = RW._copy_sites!(W, QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), m, sites1, 33, 0)
        @test o1 == 128 == RW._block_end(W, 0, 33)
        o2 = RW._copy_sites!(W, QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), m, sites2, 1, o1)
        @test o2 == 192 == RW._block_end(W, o1, 1)
        @test RW._copy_sites!(W, QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), m, sites2, 0, o2) == o2
        U = rw_unpack_rows(W, o2)
        for j in 1:m
            @test U[1:2:66, j] == stabilizer_matrix(tab)[sites1, j]
            @test U[2:2:66, j] == stabilizer_matrix(tab)[n .+ sites1, j]
            @test all(iszero, U[67:128, j])
            @test U[129:130, j] == [stabilizer_matrix(tab)[sites2[1], j], stabilizer_matrix(tab)[n + sites2[1], j]]
            @test all(iszero, U[131:192, j])
        end
        @test_throws ArgumentError RW._copy_sites!(W, QuditClifford._stab(tab, QuditClifford._layout_marker(tab)), m, sites2, 1, 2)
        inv = RW.PrecomputedInvMod(2)
        @test RW._block_rank!(W, 128, 192, 1:m, 2, inv) ==
              rw_rank(rw_restrict(tab, sites2), 2)
        @test_throws ArgumentError RW._block_rank!(W, 2, 192, 1:m, 2, inv)
        @test RW._block_end(zeros(UInt8, 1, 1), 5, 3) == 11
    end

    @testset "GF(2) symplectic Gram matrix" begin
        rng = Xoshiro(19)
        for (NS, k, piv) in ((1, 2, 0), (31, 5, 3), (32, 64, 1), (33, 65, 2), (70, 130, 0))
            nw = cld(2NS, 64)
            w0 = 1
            wK = w0 + nw + 1
            W = rand(rng, UInt64, wK + cld(k, 64), piv + k)
            # the site rows hold only whole sites; bits past them are zero
            2NS % 64 != 0 && (W[w0 + nw, :] .&= (one(UInt64) << (2NS % 64)) - 1)
            S = rw_unpack_rows(W[(w0 + 1):(w0 + nw), :], 2NS)
            X = S[1:2:end, (piv + 1):end]
            Z = S[2:2:end, (piv + 1):end]
            Kref = mod.(transpose(X) * Z .+ transpose(Z) * X, 2)
            before = W[(w0 + 1):(w0 + nw), :]
            RW._symplectic_gram_gf2!(W, w0, nw, piv, k, wK)
            @test rw_unpack_rows(W[(wK + 1):end, 1:k], k) == Kref
            @test W[(w0 + 1):(w0 + nw), :] == before
        end
        @test_throws BoundsError RW._symplectic_gram_gf2!(zeros(UInt64, 2, 3), 0, 1, 0, 3, 2)
    end

    # Qubit states large enough that the bit-packed row blocks span several
    # words and do not end on a word boundary, and partitions that fill the
    # bit workspace to its last word: three blocks of 2N = 66 bits at n = 99
    # take six words, and n = 3 split into singletons takes three.
    @testset "$(nameof(TT)), d=2, multi-word blocks" for TT in (StabilizerTableau, DestabilizerTableau)
        rng = Xoshiro(hash((TT, :multiword)) % UInt)
        for n in (3, 33, 65, 99, 130)
            states = Any[]
            for m in unique((n ÷ 2, n - 1, n))
                tab = TT(2, n)
                random_state!(rng, tab; m)
                push!(states, tab)
            end
            push!(states, rw_local_state(rng, TT, 2, n, 0), rw_local_state(rng, TT, 2, n, n),
                  TT(2, n; state=:ghz))
            partitions = Any[]
            for _ in 1:3
                perm = randperm(rng, n)
                cuts = sort(rand(rng, 0:n, 3))
                push!(partitions, (perm[1:cuts[1]], perm[(cuts[1] + 1):cuts[2]],
                                   perm[(cuts[2] + 1):cuts[3]]))
            end
            N = n ÷ 3
            perm = randperm(rng, n)
            push!(partitions, (perm[1:N], perm[(N + 1):2N], perm[(2N + 1):3N]),
                  (perm[1:1], perm[2:2], perm[3:min(n, 3)]),
                  (perm[1:(n - 2)], perm[(n - 1):(n - 1)], perm[n:n]))
            for tab in states, (A, B, C) in partitions
                before = copy(stabilizer_matrix(tab))
                @test rw_measured(tab, A, B, C) == rw_information(tab, A, B, C)
                @test stabilizer_matrix(tab) == before
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
                before = copy(stabilizer_matrix(tab))

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

                @test RW._generator_rank(tab, RW._layout_marker(tab)) == m
                @test is_pure(tab; verify=true) == (m == n)
                @test stabilizer_matrix(tab) == before
            end
        end
    end

    @testset "A subsystem stored in a rank workspace is copied first" begin
        for d in (2, 3)
            tab = StabilizerTableau(d, 6)
            random_state!(Xoshiro(3), tab; m=6)
            expected = entanglement_entropy(tab, [1, 2])
            # A 2-element view into the rank workspace for d, filled with
            # sites 1, 2.
            ws = RW._rank_workspace(tab)
            ws[1] = 1
            ws[2] = 2
            @test entanglement_entropy(tab, view(ws, 1:2)) == expected
            @test entanglement_profile(tab, view(ws, 1:6) .= 1:6) ==
                  entanglement_profile(tab)
        end
    end

    @testset "No allocations once compiled" begin
        # One entry per function, so a failure shows which one allocated.
        rw_allocs(tab, dest, A, B, C) = (
            (@allocated entanglement_entropy(tab, A)),
            (@allocated entanglement_profile!(dest, tab)),
            (@allocated RW._generator_rank(tab, RW._layout_marker(tab))),
            (@allocated mutual_information(tab, A, B)),
            (@allocated mutual_information(tab, A)),
            (@allocated tripartite_information(tab, A, B, C)),
            (@allocated conditional_mutual_information(tab, A, B, C)),
            (@allocated entanglement_negativity(tab, A, B)),
            (@allocated entanglement_negativity(tab, A)),
        )
        # n = 70 at d = 2 puts the bit-packed row blocks across several words.
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31, 257),
            E in rw_types(d), n in (8, 70), m in (n ÷ 2, n)
            d != 2 && n > 8 && continue
            tab = TT(d, n; eltype = E)
            random_state!(Xoshiro(5), tab; m)
            A = [1, 3, 4]
            B = collect(5:2:n)
            C = collect(6:2:n)
            dest = zeros(Int, n + 1)
            rw_allocs(tab, dest, A, B, C)
            @test rw_allocs(tab, dest, A, B, C) == ntuple(_ -> 0, 9)
        end
    end
end
