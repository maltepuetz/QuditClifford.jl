using QuditClifford
using Test
using Random

@testset "Entanglement entropy" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                @testset "Small systems" begin
                    # Product state |00⟩ with generators Z₁ and Z₂.
                    tab_prod = zeros(Int, 5, 2)
                    tab_prod[3, 1] = 1  # Z₁
                    tab_prod[4, 2] = 1  # Z₂
                    tab_prod = TT(2, tab_prod; m=2, storephase=true)
                    @test entanglement_entropy(tab_prod, [1]) == 0

                    # Bell state |Φ+⟩ with generators X₁X₂ and Z₁Z₂.
                    tab_bell = zeros(Int, 5, 2)
                    tab_bell[1, 1] = 1  # X₁
                    tab_bell[2, 1] = 1  # X₂
                    tab_bell[3, 2] = 1  # Z₁
                    tab_bell[4, 2] = 1  # Z₂
                    tab_bell = TT(2, tab_bell; m=2, storephase=true)
                    @test entanglement_entropy(tab_bell, [1]) == 1

                    # Mixed state (m < n).
                    tab_mixed = zeros(Int, 5, 2)
                    tab_mixed[3, 1] = 1  # Z₁ only
                    tab_mixed = TT(2, tab_mixed; m=1, storephase=true)
                    @test entanglement_entropy(tab_mixed, [1]) == 0
                    @test entanglement_entropy(tab_mixed, [2]) == 1
                    @test entanglement_entropy(tab_mixed, [1, 2]) == 1
                end

                @testset "Larger mixed systems" begin
                    # Qubits (d=2), start from maximally mixed and measure commuting operators.
                    tab = TT(2, 5; storephase=true) # m=0

                    # Maximally mixed: S(A) = |A|
                    @test entanglement_entropy(tab, [1]) == 1
                    @test entanglement_entropy(tab, [2, 4]) == 2
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 5

                    # Measure Z₁
                    measure!(tab, SinglePauli(1, 0, 1))
                    @test entanglement_entropy(tab, [1]) == 0
                    @test entanglement_entropy(tab, [2]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 1
                    @test entanglement_entropy(tab, [3, 4, 5]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 4

                    # Measure X₂
                    measure!(tab, SinglePauli(2, 1, 0))
                    @test entanglement_entropy(tab, [1]) == 0
                    @test entanglement_entropy(tab, [2]) == 0
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [2, 3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 3

                    # Measure X₄X₅
                    measure!(tab, DoublePauli(4, 1, 0, 5, 1, 0))
                    @test entanglement_entropy(tab, [4, 5]) == 1
                    @test entanglement_entropy(tab, [4]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 2

                    # Measure Z₄Z₅ (now qudits 4&5 become pure Bell pair)
                    measure!(tab, DoublePauli(4, 0, 1, 5, 0, 1))
                    @test entanglement_entropy(tab, [4, 5]) == 0
                    @test entanglement_entropy(tab, [4]) == 1
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [3, 4, 5]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 1

                    # test whether subsystem can be given in arbitrary order
                    @test entanglement_entropy(tab, [5, 4]) == 0
                    @test entanglement_entropy(tab, [5, 3, 4]) == 1
                    @test entanglement_entropy(tab, [4, 3, 2, 1, 5]) == 1
                end

                @testset "Larger pure systems" begin
                    # Qubits (d=2), three Bell pairs on 6 qubits.
                    tab = zeros(Int, 13, 6)
                    # Pair (1,2): X₁X₂ and Z₁Z₂
                    tab[1, 1] = 1
                    tab[2, 1] = 1
                    tab[7, 2] = 1
                    tab[8, 2] = 1
                    # Pair (3,4): X₃X₄ and Z₃Z₄
                    tab[3, 3] = 1
                    tab[4, 3] = 1
                    tab[9, 4] = 1
                    tab[10, 4] = 1
                    # Pair (5,6): X₅X₆ and Z₅Z₆
                    tab[5, 5] = 1
                    tab[6, 5] = 1
                    tab[11, 6] = 1
                    tab[12, 6] = 1

                    tab = TT(2, tab; m=6, storephase=true)

                    @test entanglement_entropy(tab, [1]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [1, 3]) == 2
                    @test entanglement_entropy(tab, [1, 3, 5]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4]) == 0
                    @test entanglement_entropy(tab, [2, 4, 6]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 1
                end
            end

            @testset "Qudit (d=3)" begin
                @testset "Small systems" begin
                    # Product state |00⟩ with generators Z₁ and Z₂.
                    tab_prod = zeros(Int, 5, 2)
                    tab_prod[3, 1] = 1  # Z₁
                    tab_prod[4, 2] = 1  # Z₂
                    tab_prod = TT(3, tab_prod; m=2, storephase=true)
                    @test entanglement_entropy(tab_prod, [1]) == 0

                    # Maximally entangled qutrit state with generators X₁X₂ and Z₁Z₂.
                    tab_bell = zeros(Int, 5, 2)
                    tab_bell[1, 1] = 1  # X₁
                    tab_bell[2, 1] = 1  # X₂
                    tab_bell[3, 2] = 1  # Z₁
                    tab_bell[4, 2] = 1  # Z₂
                    tab_bell = TT(3, tab_bell; m=2, storephase=true)
                    @test entanglement_entropy(tab_bell, [1]) == 1

                    # Mixed state (m < n).
                    tab_mixed = zeros(Int, 5, 2)
                    tab_mixed[3, 1] = 1  # Z₁ only
                    tab_mixed = TT(3, tab_mixed; m=1, storephase=true)
                    @test entanglement_entropy(tab_mixed, [1]) == 0
                    @test entanglement_entropy(tab_mixed, [2]) == 1
                    @test entanglement_entropy(tab_mixed, [1, 2]) == 1
                end

                @testset "Larger mixed systems" begin
                    # Qudits (d=3), start from maximally mixed and measure commuting operators.
                    tab = TT(3, 5; storephase=true) # m=0

                    # Maximally mixed: S(A) = |A|
                    @test entanglement_entropy(tab, [1]) == 1
                    @test entanglement_entropy(tab, [2, 4]) == 2
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 5

                    # Measure Z₁
                    measure!(tab, SinglePauli(1, 0, 1))
                    @test entanglement_entropy(tab, [1]) == 0
                    @test entanglement_entropy(tab, [2]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 1
                    @test entanglement_entropy(tab, [3, 4, 5]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 4

                    # Measure X₂
                    measure!(tab, SinglePauli(2, 1, 0))
                    @test entanglement_entropy(tab, [1]) == 0
                    @test entanglement_entropy(tab, [2]) == 0
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [2, 3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 3

                    # Measure X₄X₅
                    measure!(tab, DoublePauli(4, 1, 0, 5, 1, 0))
                    @test entanglement_entropy(tab, [4, 5]) == 1
                    @test entanglement_entropy(tab, [4]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 2

                    # Measure Z₄Z₅^{-1} (commutes with X₄X₅ for d=3)
                    measure!(tab, DoublePauli(4, 0, 1, 5, 0, 2))
                    @test entanglement_entropy(tab, [4, 5]) == 0
                    @test entanglement_entropy(tab, [4]) == 1
                    @test entanglement_entropy(tab, [3]) == 1
                    @test entanglement_entropy(tab, [3, 4, 5]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 1

                    # test whether subsystem can be given in arbitrary order
                    @test entanglement_entropy(tab, [5, 4]) == 0
                    @test entanglement_entropy(tab, [5, 3, 4]) == 1
                    @test entanglement_entropy(tab, [4, 3, 2, 1, 5]) == 1
                end

                @testset "Larger pure systems" begin
                    # Qudits (d=3), three generalized Bell pairs on 6 qutrits.
                    tab = zeros(Int, 13, 6)
                    # Pair (1,2): X₁ X₂^{-1} and Z₁Z₂
                    tab[1, 1] = 1
                    tab[2, 1] = 2
                    tab[7, 2] = 1
                    tab[8, 2] = 1
                    # Pair (3,4): X₃ X₄^{-1} and Z₃Z₄
                    tab[3, 3] = 1
                    tab[4, 3] = 2
                    tab[9, 4] = 1
                    tab[10, 4] = 1
                    # Pair (5,6): X₅ X₆^{-1} and Z₅Z₆
                    tab[5, 5] = 1
                    tab[6, 5] = 2
                    tab[11, 6] = 1
                    tab[12, 6] = 1

                    tab = TT(3, tab; m=6, storephase=true)

                    @test entanglement_entropy(tab, [1]) == 1
                    @test entanglement_entropy(tab, [1, 2]) == 0
                    @test entanglement_entropy(tab, [1, 3]) == 2
                    @test entanglement_entropy(tab, [1, 3, 5]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3]) == 1
                    @test entanglement_entropy(tab, [1, 2, 3, 4]) == 0
                    @test entanglement_entropy(tab, [2, 4, 6]) == 3
                    @test entanglement_entropy(tab, [1, 2, 3, 4, 5]) == 1
                end
            end
        end
    end
end

# `rank_fp_cols!` is the kernel behind both `entanglement_entropy` and
# `is_pure`, and it returns only a pivot count -- so it is free to leave the
# matrix in plain column echelon form rather than reduced. These tests pin the
# count itself, against a reference written independently of the
# implementation so the two can genuinely disagree.

"""Rank over F_d by plain row reduction. Deliberately unclever and unrelated
to the column-wise implementation under test."""
function _reference_rank(A0::AbstractMatrix{<:Integer}, d::Int)
    A = Matrix{Int}(mod.(A0, d))
    nrows, ncols = size(A)
    rank = 0
    row = 1
    for col in 1:ncols
        row > nrows && break
        piv = 0
        for r in row:nrows
            if A[r, col] != 0
                piv = r
                break
            end
        end
        piv == 0 && continue
        if piv != row
            A[row, :], A[piv, :] = A[piv, :], A[row, :]
        end
        A[row, :] .= mod.(A[row, :] .* invmod(A[row, col], d), d)
        for r in 1:nrows
            r == row && continue
            f = A[r, col]
            f == 0 && continue
            A[r, :] .= mod.(A[r, :] .- f .* A[row, :], d)
        end
        rank += 1
        row += 1
    end
    return rank
end

qc_rank(A, d) = QuditClifford.rank_fp_cols!(copy(A), d, QuditClifford.PrecomputedInvMod(d))

@testset "rank_fp_cols! pivot count" begin
    @testset "Structured matrices with known rank" begin
        for d in (2, 3, 5, 7)
            @test qc_rank(zeros(Int, 6, 4), d) == 0

            unit = zeros(Int, 6, 4)
            for j in 1:4
                unit[j, j] = 1
            end
            @test qc_rank(unit, d) == 4

            # A repeated column is not a new direction.
            dup = copy(unit)
            dup[:, 3] .= dup[:, 1]
            @test qc_rank(dup, d) == 3

            # Nor is a scalar multiple of one, which only shows up mod d.
            scaled = zeros(Int, 4, 3)
            scaled[1, 1] = 1
            scaled[2, 1] = 1
            scaled[:, 2] .= mod.((d - 1) .* scaled[:, 1], d)
            @test qc_rank(scaled, d) == 1

            # Wider than tall: rank is capped by the rows.
            wide = zeros(Int, 2, 5)
            wide[1, 1] = 1
            wide[2, 2] = 1
            wide[1, 4] = 1
            @test qc_rank(wide, d) == 2
        end
    end

    @testset "Random matrices agree with the reference" begin
        rng = Random.MersenneTwister(20260910)
        for d in (2, 3, 5, 7)
            for (rows, cols) in ((1, 1), (4, 4), (8, 3), (3, 8), (12, 12), (16, 9))
                for _ in 1:12
                    A = rand(rng, 0:(d - 1), rows, cols)
                    @test qc_rank(A, d) == _reference_rank(A, d)
                end
                # Rank-deficient by construction: an inner dimension below both
                # sides, which dense random matrices almost never produce.
                for inner in (1, min(rows, cols) ÷ 2)
                    inner == 0 && continue
                    for _ in 1:8
                        B = rand(rng, 0:(d - 1), rows, inner)
                        C = rand(rng, 0:(d - 1), inner, cols)
                        A = mod.(B * C, d)
                        @test qc_rank(A, d) == _reference_rank(A, d)
                    end
                end
            end
        end
    end
end

# Every nonempty proper subsystem of a generalized GHZ state carries exactly
# one unit of entropy, whatever d, n or which sites -- and the empty and full
# subsystems carry none. `entanglement_entropy`'s own docstring advertises this
# and nothing tested it.
#
# This covers the physics, not the rank kernel: it still passes if
# `rank_fp_cols!` eliminates in the wrong direction. The kernel's guard is the
# `rank_fp_cols! pivot count` testset above.
@testset "GHZ entanglement entropy is one unit for any proper subsystem" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5), n in (2, 3, 4, 6, 8)
        tab = TT(d, n; state=:ghz)

        @test entanglement_entropy(tab, Int[]) == 0
        @test entanglement_entropy(tab, collect(1:n)) == 0

        subsystems = Any[[1], [n], collect(1:(n ÷ 2)), collect(1:(n - 1))]
        n >= 4 && push!(subsystems, [1, 3], [2, n], collect(2:(n - 1)))

        for sub in subsystems
            isempty(sub) && continue
            length(sub) == n && continue
            @test entanglement_entropy(tab, sub) == 1
        end
    end
end
