using QuditClifford
using Test
using Random

# Differential tests of `measure!`'s commuting branches -- an operator that
# commutes with every generator either lies in their span (a fixed outcome) or
# is appended as a new generator -- against the canonicalizing span pipeline
# and the dense destabilizer construction of an appended generator.
#
# A `DestabilizerTableau` must match the reference bit for bit: outcome, raised
# error, RNG consumption, generators, destabilizers and `xdotz_cache`. A
# `StabilizerTableau` no longer canonicalizes for a commuting measurement, so
# its outcome, error and RNG consumption must match, the state must be the
# same, and its generator columns must be exactly the ones it had, plus the
# appended column the reference writes.

const CM = QuditClifford

# The destabilizer of an appended generator, built over all 2n rows.
cmref_after_append!(::StabilizerTableau, ::Int, ::AbstractVector{Int}) = nothing
function cmref_after_append!(tab::DestabilizerTableau, newcol::Int, res::AbstractVector{Int})
    n = tab.n
    d = tab.d
    m = newcol - 1
    pair(v, col) = sum(v[q] * tab.stab[n+q, col] - v[n+q] * tab.stab[q, col] for q in 1:n)
    q = findfirst(i -> res[i] != 0 || res[n+i] != 0, 1:n)
    q === nothing && throw(ArgumentError("Failed to construct destabilizer: residual is zero."))
    v = zeros(Int, 2n)
    res[q] != 0 ? (v[n+q] = 1) : (v[q] = 1)
    for j in 1:m
        t = mod(pair(v, j), d)
        t == 0 && continue
        v .= mod.(v .- t .* tab.destab[1:(2n), j], d)
    end
    β = mod(pair(v, newcol), d)
    β == 0 && throw(ArgumentError("Failed to construct destabilizer: β = 0."))
    tab.destab[1:(2n), newcol] .= mod.(invmod(β, d) .* v, d)
    for j in 1:m
        t = mod(pair(view(tab.destab, :, j), newcol), d)
        t == 0 && continue
        tab.destab[1:(2n), j] .= mod.(tab.destab[1:(2n), j] .- t .* tab.destab[1:(2n), newcol], d)
    end
    tab.xdotz_cache[newcol] = CM.dot_xz_col(tab.stab, n, newcol, d)
    return nothing
end

# `measure!`'s commuting branch as it was: `in_span_and_coeffs!`, which
# canonicalizes a `StabilizerTableau`, and the dense append above. Only called
# for operators that commute with every generator.
function cmref_commuting!(rng, tab, op; outcome = nothing, phase_policy = 0)
    d = tab.d
    n = tab.n
    m = tab.m
    kop = tab.storephase ? CM._effective_measurement_phase(op, d, phase_policy) : op.phase
    kop = mod(kop, CM.phase_modulus(d))
    tab.m == n && !tab.storephase && return 0
    in_span = CM.in_span_and_coeffs!(tab, op)
    res = tab.res_workspace
    if in_span
        tab.storephase || return 0
        kacc = CM.phase_exponent_from_coeffs!(tab)
        if d == 2
            k = mod(-kacc + kop, 4)
            isodd(k) && throw(ArgumentError(
                "the measured operator has eigenvalue $(k == 1 ? "i" : "-i") on this state, " *
                "so it has no outcome in 0:1: the operator or a stored generator is not " *
                "Hermitian. Pass phase_policy=1 to measure the Hermitian operator, or use " *
                "expect_int! for the phase exponent."))
            return k >> 1
        end
        return mod(-kacc + kop, d)
    end
    (m < n) || throw(ArgumentError("Cannot append generator: tableau at full capacity (m==n)."))
    t = CM._random_outcome(rng, outcome, d)
    newcol = m + 1
    tab.m = newcol
    CM.set_operator!(tab, newcol, op)
    tab.storephase && (tab.stab[2n+1, newcol] = CM._measured_generator_phase(t, kop, d))
    cmref_after_append!(tab, newcol, res)
    tab.iscanonical = false
    return t
end

cmref_exp(rng, d) = rand(rng, -2d:2d)

# A sparse Pauli of every type, or a dense one.
function cmref_random_op(rng, d, n)
    kind = rand(rng, 1:5)
    ph = rand(rng, -5:5)
    if kind == 2 && n >= 2
        q1, q2 = randperm(rng, n)[1:2]
        return DoublePauli(q1, cmref_exp(rng, d), cmref_exp(rng, d),
                           q2, cmref_exp(rng, d), cmref_exp(rng, d), ph)
    elseif kind == 3 && n >= 3
        q = randperm(rng, n)[1:3]
        return TriplePauli(q[1], cmref_exp(rng, d), cmref_exp(rng, d),
                           q[2], cmref_exp(rng, d), cmref_exp(rng, d),
                           q[3], cmref_exp(rng, d), cmref_exp(rng, d), ph)
    elseif kind == 4 && n >= 4
        q = Tuple(randperm(rng, n)[1:4])
        return NPauli(q, ntuple(_ -> cmref_exp(rng, d), 4), ntuple(_ -> cmref_exp(rng, d), 4), ph)
    elseif kind == 5
        xz = zeros(Int, 2n)
        for q in randperm(rng, n)[1:rand(rng, 1:n)]
            xz[q] = cmref_exp(rng, d)
            xz[n+q] = cmref_exp(rng, d)
        end
        return GeneralPauli(xz, ph)
    end
    q = rand(rng, 1:n)
    r = rand(rng)
    r < 0.3 && return SinglePauli(q, 0, 1)
    r < 0.5 && return SinglePauli(q, 1, 0)
    return SinglePauli(q, cmref_exp(rng, d), cmref_exp(rng, d), ph)
end

# A product of a few generators with random powers and a random phase, so the
# span solve has several nonzero coefficients to find.
function cmref_span_op(rng, tab)
    n = tab.n
    d = tab.d
    xz = zeros(Int, 2n)
    for j in randperm(rng, tab.m)[1:min(tab.m, rand(rng, 1:4))]
        xz .= mod.(xz .+ rand(rng, 1:(d - 1)) .* tab.stab[1:(2n), j], d)
    end
    return GeneralPauli(xz, rand(rng, -5:5))
end

cmref_op(rng, tab) = tab.m > 0 && rand(rng) < 0.4 ? cmref_span_op(rng, tab) :
                     cmref_random_op(rng, tab.d, tab.n)

# Pure states (`:local` near a product state, `:random`) and mixed ones
# (`:mixed` from random measurements, `:young` from a short monitored circuit
# on the maximally mixed state).
function cmref_state(rng, TT, d, n, flavor, storephase)
    tab = TT(d, n; state = flavor in (:mixed, :young) ? :mixed : :Z, storephase = storephase)
    local_gate!() = (q = rand(rng, 1:n);
                     apply!(tab, random_clifford(rng, d, n >= 2 ? [q, mod1(q + 1, n)] : [q])))
    if flavor === :random
        random_state!(rng, tab)
    elseif flavor === :local
        foreach(_ -> local_gate!(), 1:2n)
    elseif flavor === :mixed
        for _ in 1:(n ÷ 2)
            measure!(rng, tab, cmref_random_op(rng, d, n); phase_policy = 1)
        end
    elseif flavor === :young
        for _ in 1:3, q in 1:n
            local_gate!()
            rand(rng) < 0.3 && measure!(rng, tab, SinglePauli(q, 0, 1))
        end
    end
    return tab
end

function cmref_same(a, b)
    a.m == b.m && a.stab == b.stab && a.iscanonical == b.iscanonical &&
        a.xdotz_cache[1:a.m] == b.xdotz_cache[1:b.m] &&
        (!(a isa DestabilizerTableau) || a.destab == b.destab)
end

cmref_try(f) = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
    try
        f()
    catch e
        e isa ArgumentError || rethrow()
        (:threw, sprint(showerror, e))
    end
end

# The phase of a qubit generator made Hermitian, so that generators stay
# Hermitian and the stabilizer group never holds -I. Measuring a non-Hermitian
# operator under phase_policy 0 or 2 can store one, and then the phase of a
# product depends on the order of the factors.
function cmref_hermitian!(tab)
    (tab.d == 2 && tab.storephase) || return tab
    n = tab.n
    for j in 1:tab.m
        isodd(tab.stab[2n+1, j] - CM.dot_xz_col(tab.stab, n, j, 2)) || continue
        tab.stab[2n+1, j] = mod(tab.stab[2n+1, j] + 1, 4)
    end
    return tab
end

function cmref_fuzz(TT, d, n, flavor, storephase, nsteps, seed)
    rng = Xoshiro(seed)
    tab = cmref_state(rng, TT, d, n, flavor, storephase)
    counts = Dict(:fixed => 0, :append => 0, :threw => 0, :canonical => 0)
    for step in 1:nsteps
        op = cmref_op(rng, tab)
        policy = rand(rng, 0:2)
        out = rand(rng) < 0.5 ? nothing : rand(rng, 0:(d - 1))
        if !CM._commutes_with_generators(tab, op)
            # The random branch only moves the state along; test/measurement_kernel.jl
            # checks it.
            measure!(rng, tab, op; phase_policy = 1)
            continue
        end
        TT === StabilizerTableau && rand(rng) < 0.25 && canonicalize!(tab)
        pre = copy(tab)
        ref = copy(tab)
        ra = Xoshiro(seed + step)
        rb = copy(ra)
        resa = cmref_try(() -> cmref_commuting!(ra, ref, op; outcome = out, phase_policy = policy))
        resb = cmref_try(() -> measure!(rb, tab, op; outcome = out, phase_policy = policy))
        ok = resa == resb && rand(ra, UInt64) == rand(rb, UInt64)
        m0 = pre.m
        if TT === DestabilizerTableau
            ok &= cmref_same(ref, tab)
        else
            ok &= tab == ref && tab.m == ref.m
            if tab.m == m0
                # fixed outcome or error: the generators are untouched
                ok &= tab.stab == pre.stab && tab.iscanonical == pre.iscanonical
            else
                ok &= tab.stab[:, 1:m0] == pre.stab[:, 1:m0] &&
                      tab.stab[:, m0 + 1] == ref.stab[:, m0 + 1] &&
                      all(iszero, tab.stab[:, (m0 + 2):n]) && !tab.iscanonical
            end
        end
        if !ok
            @info "mismatch" TT d n flavor storephase seed step op policy out resa resb
            return false, counts
        end
        if resb isa Tuple
            counts[:threw] += 1
        elseif tab.m > m0
            counts[:append] += 1
            cmref_hermitian!(tab)
        else
            counts[:fixed] += 1
        end
        pre.iscanonical && (counts[:canonical] += 1)
        if rand(rng) < 0.2 && n >= 2
            q = rand(rng, 1:n)
            apply!(tab, random_clifford(rng, d, [q, mod1(q + 1, n)]))
        end
    end
    return true, counts
end

@testset "measure!'s commuting branches match the canonicalizing reference" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 5, 31, 257),
        storephase in (true, false)
        @testset "$(nameof(TT)) d=$d storephase=$storephase" begin
            total = Dict(:fixed => 0, :append => 0, :threw => 0, :canonical => 0)
            cases = ((1, :random), (2, :mixed), (3, :local), (5, :young), (8, :mixed),
                     (12, :random), (24, :local), (24, :young), (40, :mixed), (70, :young))
            for (n, flavor) in cases
                for seed in 1:(n <= 12 ? 5 : 2)
                    ok, c = cmref_fuzz(TT, d, n, flavor, storephase, n <= 12 ? 40 : 80,
                                       1000 * d + 17 * n + seed)
                    @test ok
                    mergewith!(+, total, c)
                end
            end
            @test total[:fixed] > 0
            @test total[:append] > 0
            TT === StabilizerTableau && @test total[:canonical] > 0
            d == 2 && storephase && @test total[:threw] > 0
        end
    end
end

@testset "Span solve on a copy finds the dual-basis coefficients" begin
    # Coefficients of an in-span operator are unique, so the copy solve on a
    # non-canonical StabilizerTableau must give exactly the DestabilizerTableau's
    # pairings. n = 64, 65 and 130 put the generator rows on word boundaries of
    # the bit-packed workspace; d = 131 takes the UInt16 lanes of the narrow
    # update and 257 the Int workspace.
    for d in (2, 3, 131, 257), n in (1, 2, 7, 64, 65, 130), seed in 1:2
        rng = Xoshiro(10d + n + seed)
        D = DestabilizerTableau(d, n; state = :Z)
        random_state!(rng, D)
        # Drop generators from the end, as in a mixed state.
        m = seed == 1 ? n : rand(rng, 0:n)
        S = StabilizerTableau(d, D.stab[:, 1:m]; m = m, check = false)
        @test !S.iscanonical
        for _ in 1:3
            xz = zeros(Int, 2n)
            for j in 1:m
                xz .= mod.(xz .+ rand(rng, 0:(d - 1)) .* D.stab[1:(2n), j], d)
            end
            op = GeneralPauli(xz, 0)
            CM.coeffs_from_generators!(D, op)
            stab_before = copy(S.stab)
            CM._span_coeffs_keep_basis!(S, op)
            @test S.c_workspace[1:m] == D.c_workspace[1:m]
            @test CM._in_span_from_coeffs!(S, op)
            @test S.stab == stab_before
            @test !S.iscanonical
        end
    end
end

@testset "GF(2) back-substitution packs coefficients into Int32 or Int64" begin
    # generator_workspace holds Int, which is Int32 on 32-bit hosts: the packed
    # coefficient words must survive being split into two entries.
    for n in (40, 64, 65, 130)
        rng = Xoshiro(n)
        D = DestabilizerTableau(2, n; state = :Z)
        random_state!(rng, D)
        S = StabilizerTableau(2, copy(D.stab); m = n, check = false)
        xz = mod.(D.stab[1:(2n), :] * rand(rng, 0:1, n), 2)
        CM.coeffs_from_generators!(D, GeneralPauli(xz, 0))
        for T in (Int32, Int64)
            CM._copy_generator_rows!(S.bit_workspace, S.stab, n, n)
            c = zeros(Int, n)
            rank = CM._solve_generator_rows!(S.bit_workspace, copy(xz), zeros(Int, n), c,
                                             zeros(T, 2n), n, n)
            @test rank == n
            @test c == D.c_workspace[1:n]
        end
    end
    cw = zeros(Int32, 4)
    CM._set_coeff_bit!(cw, 1)
    CM._set_coeff_bit!(cw, 33)
    CM._set_coeff_bit!(cw, 64)
    CM._set_coeff_bit!(cw, 65)
    @test CM._coeff_word(cw, 1) == (UInt64(1) | UInt64(1) << 32 | UInt64(1) << 63)
    @test CM._coeff_word(cw, 2) == 1
end

@testset "Dependent generators get coefficient zero" begin
    # Outside the contract (check=false): generator 2 repeats generator 1, so
    # no pivot takes lead 2, and the solve still finds op = g1.
    for d in (2, 3, 257)
        n = 6
        raw = copy(StabilizerTableau(d, n; state = :Z).stab)
        raw[:, 2] = raw[:, 1]
        S = StabilizerTableau(d, raw; m = n, check = false)
        S.iscanonical = false
        op = SinglePauli(1, 0, 1)
        CM._span_coeffs_keep_basis!(S, op)
        @test S.c_workspace[1:n] == [1, 0, 0, 0, 0, 0]
        @test CM._in_span_from_coeffs!(S, op)
    end
end

@testset "Generator rows are copied site by site" begin
    for (d, ws_of) in ((2, t -> t.bit_workspace), (3, t -> t.narrow_workspace),
                       (257, t -> t.workspace)), n in (1, 9, 64, 65, 130)
        rng = Xoshiro(n + d)
        S = StabilizerTableau(d, n; state = :Z)
        random_state!(rng, S)
        for m in unique((0, 1, n ÷ 2, n - 1, n))
            m < 0 && continue
            S.m = m
            ws = ws_of(S)
            CM._copy_generator_rows!(ws, S.stab, n, m)
            h = size(ws, 1) >> 1
            for q in 1:n, j in 1:m
                if d == 2
                    w = (j - 1) >> 6 + 1
                    b = (j - 1) & 63
                    @test (ws[w, q] >> b) & 1 == S.stab[q, j]
                    @test (ws[h + w, q] >> b) & 1 == S.stab[n + q, j]
                else
                    @test ws[j, q] == S.stab[q, j]
                    @test ws[h + j, q] == S.stab[n + q, j]
                end
            end
            if d == 2 && m % 64 != 0
                # bits past the last generator stay zero
                @test all(q -> ws[cld(m, 64), q] >> (m % 64) == 0, 1:n)
            end
        end
        S.m = n
    end
end

@testset "measure! keeps a StabilizerTableau's generator basis on commuting operators" begin
    for d in (2, 3)
        rng = Xoshiro(d)
        n = 12
        S = StabilizerTableau(d, n; state = :Z)
        random_state!(rng, S)
        before = copy(S.stab)
        @test !S.iscanonical
        # Fixed outcome: columns, order and flag unchanged.
        xz = mod.(S.stab[1:(2n), 3] .+ S.stab[1:(2n), 7], d)
        # Hermitian at d = 2: phase exponent ≡ x·z (mod 2).
        op = GeneralPauli(xz, d == 2 ? CM.dot_xz_col(reshape(xz, :, 1), n, 1, 2) : 0)
        expected = expect_int!(copy(S), op)
        t = measure!(S, op)
        @test S.stab == before
        @test !S.iscanonical
        @test (d == 2 ? 2t : t) == expected
        # A canonical tableau stays canonical.
        canonicalize!(S)
        canon = copy(S.stab)
        measure!(S, op)
        @test S.iscanonical
        @test S.stab == canon
        # Appending: the old columns stay as they were.
        M = StabilizerTableau(d, S.stab[:, 1:5]; m = 5, check = false)
        M.iscanonical = false
        old = copy(M.stab)
        target = findfirst(q -> CM._commutes_with_generators(M, SinglePauli(q, 0, 1)) &&
                                !CM.in_span_and_coeffs!(copy(M), SinglePauli(q, 0, 1)), 1:n)
        @test target !== nothing
        measure!(M, SinglePauli(target, 0, 1); outcome = 0)
        @test M.m == 6
        @test M.stab[:, 1:5] == old[:, 1:5]
        @test !M.iscanonical
    end
end

function cmref_allocs(rng, tab, op)
    warm = copy(tab)
    t = copy(tab)
    measure!(rng, warm, op; outcome = 0, phase_policy = 1)
    return @allocated measure!(rng, t, op; outcome = 0, phase_policy = 1)
end

@testset "measure! allocates nothing in the commuting branches" begin
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 31, 257)
        n = 64
        rng = Xoshiro(5)
        # Pure, not canonical: Z on a qudit is a fixed outcome after the gates
        # only on product-like sites, so take products of generators.
        tab = TT(d, n; state = :Z)
        for q in 1:2:(n - 1)
            apply!(tab, random_clifford(rng, d, [q, q + 1]))
        end
        inspan = GeneralPauli(mod.(tab.stab[1:(2n), 2] .+ tab.stab[1:(2n), 9], d), 0)
        @test cmref_allocs(rng, tab, inspan) == 0
        # Mixed: an operator outside the span is appended.
        mixed = TT(d, n; state = :mixed)
        for q in 1:n
            measure!(rng, mixed, SinglePauli(q, 0, 1); outcome = 0)
            q == n ÷ 2 && break
        end
        for q in 1:2:(n ÷ 2)
            apply!(mixed, random_clifford(rng, d, [q, q + 1]))
        end
        @test cmref_allocs(rng, mixed, SinglePauli(n, 0, 1)) == 0
        @test cmref_allocs(rng, mixed, DoublePauli(n - 1, 1, 0, n, 1, 0)) == 0
    end
end
