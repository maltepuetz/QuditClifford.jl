using QuditClifford
using Test
using Random

@inline function _measurement_symp(
    A::AbstractMatrix{<:Integer},
    colA::Int,
    B::AbstractMatrix{<:Integer},
    colB::Int,
    n::Int,
)
    value = 0
    @inbounds for q in 1:n
        value += A[q, colA] * B[n+q, colB]
        value -= A[n+q, colA] * B[q, colB]
    end
    return value
end

@testset "Measurements" begin
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)]
        @testset "$label" begin
            @testset "Qubit (d=2)" begin
                # Fixed seed for deterministic random-outcome tests.
                Random.seed!(42)

                # Pure qubit stabilizer |0⟩ with generator Z.
                # Noncommuting measurement (X) should be random, replace the generator,
                # and set the stabilizer phase to match the sampled outcome.
                outcomes = Int[]
                for _ in 1:10
                    tab = reshape(Int[0, 1, 0], 3, 1)
                    tab = TT(2, tab; m=1, storephase=true)

                    out = measure!(tab, Int[1, 0]) # measure X
                    @test out in 0:1
                    @test tab.m == 1
                    @test tab.stab[1, 1] == 1  # X
                    @test tab.stab[2, 1] == 0
                    @test tab.stab[3, 1] == mod(2 * out, 4)
                    push!(outcomes, out)
                end
                @test sort!(unique(outcomes)) == [0, 1]  # ensure we got both outcomes at least once

                # Commuting measurement in span (Z) should be deterministic and leave the state.
                tab_det = reshape(Int[0, 1, 0], 3, 1)
                tab_det = TT(2, tab_det; m=1, storephase=true)
                out_det = measure!(tab_det, Int[0, 1]) # measure Z
                @test out_det == 0
                @test tab_det.stab[1, 1] == 0
                @test tab_det.stab[2, 1] == 1
                @test tab_det.stab[3, 1] == 0

                # Maximally mixed qubit (m=0): commuting but not in span should append a generator.
                outcomes2 = Int[]
                for _ in 1:10
                    tab2 = zeros(Int, 3, 1)
                    tab2 = TT(2, tab2; m=0, storephase=true)

                    out2 = measure!(tab2, Int[0, 1]) # measure Z
                    @test out2 in 0:1
                    @test tab2.m == 1
                    @test tab2.stab[1, 1] == 0
                    @test tab2.stab[2, 1] == 1
                    @test tab2.stab[3, 1] == mod(2 * out2, 4)
                    push!(outcomes2, out2)
                end
                @test sort!(unique(outcomes2)) == [0, 1]  # ensure we got both outcomes at least once

                @testset "Phase policy does not mutate input operators" begin
                    # Immutable operator: phase-fix is applied internally for measurement only.
                    tab_imm = reshape(Int[0, 1, 0], 3, 1)
                    tab_imm = TT(2, tab_imm; m=1, storephase=true)
                    op_imm = SinglePauli(1, 1, 1, 0)  # invalid for d=2: parity=1, phase=0
                    out_imm = measure!(tab_imm, op_imm; outcome=1, phase_policy=1)
                    @test out_imm == 1
                    @test op_imm.phase == 0
                    @test tab_imm.stab[1, 1] == 1
                    @test tab_imm.stab[2, 1] == 1
                    @test tab_imm.stab[3, 1] == 3  # kgen = 2*outcome + fixed_phase = 2 + 1

                    # Mutable operator: phase is also left unchanged.
                    tab_mut = reshape(Int[0, 1, 0], 3, 1)
                    tab_mut = TT(2, tab_mut; m=1, storephase=true)
                    op_mut = GeneralPauli(Int[1, 1], 0)  # invalid for d=2: parity=1, phase=0
                    out_mut = measure!(tab_mut, op_mut; outcome=1, phase_policy=1)
                    @test out_mut == 1
                    @test op_mut.phase == 0
                    @test op_mut.xz == Int[1, 1]
                    @test tab_mut.stab[1, 1] == 1
                    @test tab_mut.stab[2, 1] == 1
                    @test tab_mut.stab[3, 1] == 3
                end

                @testset "Phase policy 0 warns without mutating input" begin
                    tab_warn = reshape(Int[0, 1, 0], 3, 1)
                    tab_warn = TT(2, tab_warn; m=1, storephase=true)
                    op_warn = GeneralPauli(Int[1, 1], 0)  # invalid for d=2: parity=1, phase=0
                    @test_logs (:warn, r"Measuring a non-Hermitian Pauli") begin
                        out_warn = measure!(tab_warn, op_warn; outcome=1, phase_policy=0)
                        @test out_warn == 1
                    end
                    @test op_warn.phase == 0
                    @test op_warn.xz == Int[1, 1]
                    @test tab_warn.stab[1, 1] == 1
                    @test tab_warn.stab[2, 1] == 1
                    @test tab_warn.stab[3, 1] == 2  # kgen = 2*outcome + original_phase
                end

                @testset "Phase policy validation and silent opt-out" begin
                    invalid_op = SinglePauli(1, 1, 1, 0)

                    tab_silent = TT(2, 1; state=:product, basis=:Z)
                    out_silent = @test_logs measure!(
                        tab_silent,
                        invalid_op;
                        outcome=1,
                        phase_policy=2,
                    )
                    @test out_silent == 1
                    @test tab_silent.stab[:, 1] == Int[1, 1, 2]
                    @test invalid_op.phase == 0

                    tab_invalid = TT(2, 1; state=:product, basis=:Z)
                    @test_throws ArgumentError measure!(
                        tab_invalid,
                        invalid_op;
                        outcome=0,
                        phase_policy=3,
                    )

                    # The optimized sparse representations must apply the same
                    # Hermiticity correction as their dense equivalent.
                    sparse_invalid_ops = (
                        TriplePauli(1, 1, 1, 2, 0, 0, 3, 0, 0),
                        NPauli((1,), (1,), (1,)),
                    )
                    for sparse_op in sparse_invalid_ops
                        tab_sparse = TT(2, 3; state=:product, basis=:Z)
                        @test measure!(
                            tab_sparse,
                            sparse_op;
                            outcome=0,
                            phase_policy=1,
                        ) == 0
                        @test tab_sparse.stab[:, 1] == Int[1, 0, 0, 1, 0, 0, 1]
                    end
                end

                @testset "Phase-free deterministic measurements" begin
                    tab_nophase = TT(2, 1; state=:product, basis=:Z, storephase=false)
                    before = copy(tab_nophase.stab)
                    @test measure!(tab_nophase, SinglePauli(1, 0, 1); outcome=1) == 0
                    @test tab_nophase.stab == before
                end
            end

            @testset "Qudit (d=3)" begin
                # Fixed seed for deterministic random-outcome tests.
                Random.seed!(42)

                # Pure qutrit stabilizer |0⟩ with generator Z.
                # Noncommuting measurement (X) should be random, replace the generator,
                # and set the stabilizer phase to match the sampled outcome.
                outcomes = Int[]
                for _ in 1:20
                    tab = reshape(Int[0, 1, 0], 3, 1)
                    tab = TT(3, tab; m=1, storephase=true)

                    out = measure!(tab, Int[1, 0]) # measure X
                    @test out in 0:2
                    @test tab.m == 1
                    @test tab.stab[1, 1] == 1  # X
                    @test tab.stab[2, 1] == 0
                    @test tab.stab[3, 1] == mod(-out, 3)
                    push!(outcomes, out)
                end
                @test sort!(unique(outcomes)) == [0, 1, 2]  # ensure we got all outcomes at least once

                # Commuting measurement in span (Z) should be deterministic and leave the state.
                tab_det = reshape(Int[0, 1, 0], 3, 1)
                tab_det = TT(3, tab_det; m=1, storephase=true)
                out_det = measure!(tab_det, Int[0, 1]) # measure Z
                @test out_det == 0
                @test tab_det.stab[1, 1] == 0
                @test tab_det.stab[2, 1] == 1
                @test tab_det.stab[3, 1] == 0

                # Maximally mixed qutrit (m=0): commuting but not in span should append a generator.
                outcomes2 = Int[]
                for _ in 1:20
                    tab2 = zeros(Int, 3, 1)
                    tab2 = TT(3, tab2; m=0, storephase=true)

                    out2 = measure!(tab2, Int[0, 1]) # measure Z
                    @test out2 in 0:2
                    @test tab2.m == 1
                    @test tab2.stab[1, 1] == 0
                    @test tab2.stab[2, 1] == 1
                    @test tab2.stab[3, 1] == mod(-out2, 3)
                    push!(outcomes2, out2)
                end
                @test sort!(unique(outcomes2)) == [0, 1, 2]  # ensure we got all outcomes at least once
            end
        end
    end
end

@testset "Symplectic commutation specializations" begin
    v = Int[1, 2, 0, 2, 1, 1]
    xz = Int[2, 0, 1, 1, 2, 2]
    expected = QuditClifford.commutation(v, xz)
    matrix = hcat(v, reverse(v))

    operators = (
        xz,
        GeneralPauli(xz, 0),
        TriplePauli(1, 2, 1, 2, 0, 2, 3, 1, 2),
        NPauli((1, 2, 3), (2, 0, 1), (1, 2, 2)),
    )

    # Sparse specializations must preserve the dense symplectic convention.
    for op in operators
        @test QuditClifford.commutation_vec_op(v, op) == expected
        @test QuditClifford.commutation_col(matrix, 1, op) == expected
    end
end

@testset "Measurement agrees across sparse Pauli representations" begin
    dense = GeneralPauli(Int[0, 0, 0, 0, 1, 1], 0)
    triple = TriplePauli(1, 0, 0, 2, 0, 1, 3, 0, 1)
    sparse = NPauli((2, 3), (0, 0), (1, 1))

    for (label, TT) in [
        ("StabilizerTableau", StabilizerTableau),
        ("DestabilizerTableau", DestabilizerTableau),
    ]
        @testset "$label" begin
            results = Matrix{Int}[]
            for op in (dense, triple, sparse)
                raw = zeros(Int, 7, 3)
                raw[4, 1] = 1 # initial generator Z₁
                tab = TT(3, raw; m=1, storephase=true)

                @test measure!(tab, op; outcome=2) == 2
                @test tab.m == 2
                @test tab.stab[:, 1] == Int[0, 0, 0, 1, 0, 0, 0]
                @test tab.stab[:, 2] == Int[0, 0, 0, 0, 1, 1, 1]
                push!(results, copy(tab.stab))
            end
            @test all(==(first(results)), results)
        end
    end
end

@testset "Multi-generator noncommuting measurement" begin
    for d in (2, 3), storephase in (false, true)
        outcome = d - 1
        op = DoublePauli(1, 0, 1, 2, 0, 1) # Z₁Z₂

        stab = StabilizerTableau(d, 2; state=:product, basis=:X, storephase=storephase)
        destab = DestabilizerTableau(d, 2; state=:product, basis=:X, storephase=storephase)

        # Both X-basis generators fail to commute with Z₁Z₂. The first is
        # replaced and the second must be multiplied by the saved pivot.
        @test measure!(stab, op; outcome=outcome) == outcome
        @test measure!(destab, op; outcome=outcome) == outcome

        expected_exponent = if !storephase
            0
        elseif d == 2
            2 * outcome
        else
            outcome
        end

        for tab in (stab, destab)
            @test tab.m == tab.n == 2
            @test QuditClifford.is_commuting(tab)
            @test QuditClifford.is_independent(tab)
            @test is_pure(tab; verify=true)
            @test expect_int!(tab, op) == expected_exponent
        end

        for i in 1:destab.m, j in 1:destab.m
            value = mod(_measurement_symp(destab.destab, i, destab.stab, j, destab.n), d)
            @test value == (i == j ? 1 : 0)
        end

        canonicalize!(stab)
        canonicalize!(destab)
        @test stab.stab == destab.stab
    end
end

# Largest-magnitude Int of the given sign congruent to `r` mod `d`.
function _huge_representative(r::Int, d::Int, negative::Bool)
    negative && return typemin(Int) + mod(r - mod(typemin(Int), d), d)
    return typemax(Int) - mod(mod(typemax(Int), d) - r, d)
end

# Representatives of `xs`, alternating in sign so that sums of products wrap too.
_huge_tuple(xs::NTuple{D,Int}, d::Int, offset::Int) where D =
    ntuple(i -> _huge_representative(xs[i], d, isodd(i + offset)), D)

# The operator with exponents `xs`, `zs` on `qudits` and phase `phase`, of the
# given `kind`, written with reduced or with huge exponents.
function _huge_measurement_op(kind::Symbol, n::Int, d::Int, qudits, xs, zs, phase::Int, huge::Bool)
    pm = d == 2 ? 4 : d
    if huge
        xs = _huge_tuple(xs, d, 0)
        zs = _huge_tuple(zs, d, 1)
        phase = _huge_representative(phase, pm, true)
    end
    D = length(qudits)
    if kind === :single
        return SinglePauli(qudits[1], xs[1], zs[1], phase)
    elseif kind === :double
        return DoublePauli(qudits[1], xs[1], zs[1], qudits[2], xs[2], zs[2], phase)
    elseif kind === :triple
        return TriplePauli(qudits[1], xs[1], zs[1], qudits[2], xs[2], zs[2],
                           qudits[3], xs[3], zs[3], phase)
    elseif kind === :npauli
        return NPauli(qudits, xs, zs, phase)
    end
    # Dense forms: every unsupported exponent is a huge representative of 0.
    xz = huge ? collect(_huge_tuple(ntuple(_ -> 0, 2n), d, 0)) : zeros(Int, 2n)
    for i in 1:D
        xz[qudits[i]] = xs[i]
        xz[qudits[i]+n] = zs[i]
    end
    kind === :general && return GeneralPauli(xz, phase)
    return vcat(xz, phase) # :vector
end

# Bytes allocated by measuring `op` on X₁ᵃ, X₂ᵃ, X₃ᵃ (`m == 3`) or X₁ᵃX₂ᵃX₃ᵃ
# (`m == 1`), after a warm-up on an identical tableau.
function _huge_measure_allocated(TT, d::Int, storephase::Bool, op, m::Int)
    n = 3
    raw = zeros(Int, 2n + storephase, n)
    if m == n
        for q in 1:n
            raw[q, q] = d - 1
        end
    else
        raw[1:n, 1] .= d - 1
    end
    warm = TT(d, copy(raw); m=m, storephase=storephase)
    tab = TT(d, copy(raw); m=m, storephase=storephase)
    measure!(warm, op; outcome=1, phase_policy=2)
    return @allocated measure!(tab, op; outcome=1, phase_policy=2)
end

@testset "Measurement reduces huge exponents before commutation" begin
    n = 3
    support = Dict(
        :single => (1,), :double => (1, 2), :triple => (1, 2, 3),
        :npauli => (1, 3), :general => (1, 2, 3), :vector => (1, 2, 3),
    )
    for (label, TT) in [("StabilizerTableau", StabilizerTableau), ("DestabilizerTableau", DestabilizerTableau)],
        d in (2, 3, 5), storephase in (false, true)

        a = d - 1 # generator exponent, ≠ 1 for odd d so products overflow
        rows = 2n + storephase
        pure = zeros(Int, rows, n) # X₁ᵃ, X₂ᵃ, X₃ᵃ
        for q in 1:n
            pure[q, q] = a
        end
        mixed = zeros(Int, rows, n) # X₁ᵃX₂ᵃX₃ᵃ
        mixed[1:n, 1] .= a

        @testset "$label d=$d storephase=$storephase $kind" for kind in keys(support)
            qudits = support[kind]
            D = length(qudits)
            ones_ = ntuple(_ -> 1, D)
            zeros_ = ntuple(_ -> 0, D)
            first_ = ntuple(i -> i == 1 ? 1 : 0, D)
            cases = (
                # Commutes with every Xᵃ and lies in their span.
                (:inspan, pure, n, ones_, zeros_),
                # Anticommutes with each generator on its support.
                (:noncommuting, pure, n, zeros_, ones_),
                # Commutes with X₁ᵃX₂ᵃX₃ᵃ but is not a power of it.
                (:append, mixed, 1, first_, zeros_),
            )
            for (branch, raw, m, xs, zs) in cases
                outcome = d - 1
                phase = 1
                tab = TT(d, copy(raw); m=m, storephase=storephase)
                ref = TT(d, copy(raw); m=m, storephase=storephase)
                op = _huge_measurement_op(kind, n, d, qudits, xs, zs, phase, true)
                op_ref = _huge_measurement_op(kind, n, d, qudits, xs, zs, phase, false)

                expected = measure!(ref, op_ref; outcome=outcome, phase_policy=2)
                if branch === :inspan
                    @test expected == (storephase ? phase : 0)
                    @test ref.m == m
                else
                    @test expected == outcome
                    @test ref.m == (branch === :append ? m + 1 : m)
                end

                @test measure!(tab, op; outcome=outcome, phase_policy=2) == expected
                @test tab.m == ref.m
                @test tab.stab == ref.stab
                if TT === DestabilizerTableau
                    @test tab.destab == ref.destab
                    @test tab.xdotz_cache == ref.xdotz_cache
                end
            end
        end
    end

    @testset "Sparse operands stay allocation-free" begin
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3), storephase in (false, true)
            for (kind, m, xs, zs) in ((:triple, 3, (1, 1, 1), (0, 0, 0)), # in span
                                      (:triple, 3, (0, 0, 0), (1, 1, 1)), # noncommuting
                                      (:triple, 1, (1, 0, 0), (0, 0, 0)), # append
                                      (:npauli, 1, (1, 0), (0, 0)))
                qudits = kind === :triple ? (1, 2, 3) : (1, 3)
                op = _huge_measurement_op(kind, 3, d, qudits, xs, zs, 1, true)
                @test _huge_measure_allocated(TT, d, storephase, op, m) == 0
            end
        end
    end

    @testset "Reproducer: d=3, generators X₁², X₂², X₃²" begin
        for TT in (StabilizerTableau, DestabilizerTableau)
            raw = zeros(Int, 6, 3)
            raw[1, 1] = raw[2, 2] = raw[3, 3] = 2
            tab = TT(3, copy(raw); m=3, storephase=false)
            ref = TT(3, copy(raw); m=3, storephase=false)
            x = typemin(Int)                          # ≡ 1 mod 3
            z = typemax(Int) - mod(typemax(Int), 3)   # ≡ 0 mod 3
            @test measure!(tab, SinglePauli(1, x, z); outcome=1) == 0
            canonicalize!(tab)
            canonicalize!(ref)
            @test tab.stab == ref.stab
        end
    end
end
