# Helpers for the tests of storage element types and packed storage, shared by
# test/storage_types.jl, test/bits.jl and test/measurement_kernel.jl. Each of
# them includes this file unless another has done so.
using QuditClifford, Test, Random, LinearAlgebra

const st_QC = QuditClifford

st_types(d) = filter(E -> st_QC._eltype_admissible(E, d), (UInt8, UInt16, Int, Bits))

# The element type of the stored blocks of a tableau of storage type `E`: a
# packed block holds words.
st_block_eltype(E) = E === Bits ? UInt64 : E

# A signed integer type wider than one machine word on neither platform, so
# never `Int` itself.
const st_OtherSigned = Int === Int64 ? Int32 : Int64

# ---- differential driver: every operation on every element type ----

# A random Pauli in reduced form, Hermitian at d = 2: `kind` 1 gives a
# `SinglePauli`, 2 a `DoublePauli`, 3 a `GeneralPauli` on a random support.
function st_rand_pauli(rng, d, n, kind)
    k = kind == 1 ? 1 : kind == 2 ? 2 : rand(rng, 1:n)
    qs = randperm(rng, n)[1:k]
    xs = [rand(rng, 0:d-1) for _ in 1:k]
    zs = [rand(rng, 0:d-1) for _ in 1:k]
    ph = d == 2 ? mod(sum(xs .* zs), 2) + 2 * rand(rng, 0:1) : rand(rng, 0:d-1)
    kind == 1 && return SinglePauli(qs[1], xs[1], zs[1], ph)
    kind == 2 && return DoublePauli(qs[1], xs[1], zs[1], qs[2], xs[2], zs[2], ph)
    v = zeros(Int, 2n)
    for (q, x, z) in zip(qs, xs, zs)
        v[q] = x
        v[n+q] = z
    end
    return GeneralPauli(v, ph)
end

# Generator `j` of `ref` as a `GeneralPauli`: in the stabilizer group, so a
# measurement of it is deterministic and its expectation value is nonzero.
function st_generator_pauli(ref, j)
    S = stabilizer_matrix(ref)
    n = ref.n
    return GeneralPauli(S[1:2n, j], ref.storephase ? S[2n+1, j] : 0)
end

function st_rand_gate(rng, d, n)
    q, q2 = randperm(rng, n)[1:2]
    c = rand(rng, 1:7)
    c == 1 && return Fourier(q)
    c == 2 && return Phase(q)
    c == 3 && return Multiplier(q, rand(rng, 1:d-1))
    c == 4 && return PauliGate(q, rand(rng, 0:d-1), rand(rng, 0:d-1))
    c == 5 && return SUM(q, q2, rand(rng, 1:d-1))
    c == 6 && return CPhase(q, q2, rand(rng, 1:d-1))
    return SWAP(q, q2)
end

# Three disjoint random subsystems, each possibly empty.
function st_rand_parts(rng, n)
    p = randperm(rng, n)
    a = rand(rng, 0:n)
    b = rand(rng, 0:n-a)
    c = rand(rng, 0:n-a-b)
    return p[1:a], p[a+1:a+b], p[a+b+1:a+b+c]
end

# The next operation, drawn from `rng` and the current state of `ref`. It is a
# function of a tableau and that tableau's own RNG, applied to every tableau of
# the run.
function st_rand_op(rng, ref)
    d, n, m = ref.d, ref.n, ref.m
    c = rand(rng, 1:24)
    if c <= 4
        g = st_rand_gate(rng, d, n)
        return (tab, r) -> apply!(tab, g)
    elseif c == 5
        k = rand(rng, (1, 2, 3, 4))
        C = random_clifford(rng, d, randperm(rng, n)[1:k])
        return (tab, r) -> apply!(tab, C)
    elseif c <= 9
        op = (m > 0 && rand(rng, 1:3) == 1) ? st_generator_pauli(ref, rand(rng, 1:m)) :
             st_rand_pauli(rng, d, n, rand(rng, 1:3))
        return rand(rng, Bool) ? ((tab, r) -> measure!(r, tab, op)) :
                                 ((tab, r) -> measure!(tab, op; outcome = 0))
    elseif c <= 11
        op = (m > 0 && rand(rng, Bool)) ? st_generator_pauli(ref, rand(rng, 1:m)) :
             st_rand_pauli(rng, d, n, rand(rng, 1:3))
        return c == 10 ? ((tab, r) -> expect!(tab, op)) : ((tab, r) -> expect_int!(tab, op))
    elseif c == 12
        op = st_rand_pauli(rng, d, n, rand(rng, (1, 3)))
        return (tab, r) -> dephase!(tab, op)
    elseif c == 13
        sites = randperm(rng, n)[1:rand(rng, 1:min(n, 3))]
        return (tab, r) -> traceout!(tab, sites)
    elseif c == 14
        sites = randperm(rng, n)[1:rand(rng, 1:min(n, 3))]
        basis = rand(rng, (:Z, :X, :Y))
        return basis === :Z ? ((tab, r) -> reset!(tab, sites)) :
                              ((tab, r) -> reset!(tab, sites; basis))
    elseif c == 15
        return (tab, r) -> canonicalize!(tab)
    elseif c == 16
        mm = rand(rng, (n, n, rand(rng, 0:n)))
        return (tab, r) -> random_state!(r, tab; m = mm)
    elseif c == 17
        A, _, _ = st_rand_parts(rng, n)
        return (tab, r) -> entanglement_entropy(tab, A)
    elseif c == 18
        return (tab, r) -> entanglement_profile(tab)
    elseif c == 19
        A, B, _ = st_rand_parts(rng, n)
        return (tab, r) -> mutual_information(tab, A, B)
    elseif c == 20
        A, B, C = st_rand_parts(rng, n)
        return (tab, r) -> conditional_mutual_information(tab, A, B, C)
    elseif c == 21
        A, B, C = st_rand_parts(rng, n)
        return (tab, r) -> tripartite_information(tab, A, B, C)
    elseif c == 22
        A, B, _ = st_rand_parts(rng, n)
        return (tab, r) -> entanglement_negativity(tab, A, B)
    elseif c == 23
        return (tab, r) -> is_pure(tab; verify = true)
    end
    # A further gate, so that two-qudit gates make up a good share.
    g = st_rand_gate(rng, d, n)
    return (tab, r) -> apply!(tab, g)
end

# What an operation returned: a tableau stands for itself, and a thrown
# exception is compared by type and message.
function st_call(op, tab, r)
    v = try
        op(tab, r)
    catch e
        return (:error, typeof(e), sprint(showerror, e))
    end
    return v isa AbstractTableau ? (v === tab ? :self : :other) : v
end

# The symplectic products of destabilizer `i` with stabilizer `j` over `1:m`,
# in `Int64` so that no `d` here overflows them.
function st_duality(tab)
    n, m, d = tab.n, tab.m, tab.d
    S = Int64.(stabilizer_matrix(tab)[1:2n, 1:m])
    D = Int64.(destabilizer_matrix(tab)[:, 1:m])
    return mod.(transpose(D[1:n, :]) * S[n+1:2n, :] - transpose(D[n+1:2n, :]) * S[1:n, :], d)
end

function st_xdotz(tab)
    n, d = tab.n, tab.d
    S = Int64.(stabilizer_matrix(tab))
    return [Int(mod(sum(S[q, j] * S[n+q, j] for q in 1:n), d)) for j in 1:tab.m]
end

# Whether the capacity generators `m+1:n` of `tab` are all zero, phases and
# destabilizers included.
function st_capacity_zero(tab)
    m = tab.m
    ok = all(iszero, stabilizer_matrix(tab)[:, m+1:end])
    tab isa DestabilizerTableau && (ok &= all(iszero, destabilizer_matrix(tab)[:, m+1:end]))
    return ok
end

# A layout change of `tab` drawn from `rng`: `qudit_major!`, `generator_major!`,
# or `copy!` into a fresh tableau of the other layout, which then takes the
# place of `tab`. Returns the tableau to continue with.
function st_switch!(rng, tab)
    c = rand(rng, 1:3)
    c == 1 && return qudit_major!(tab)
    c == 2 && return generator_major!(tab)
    other = tableau_layout(tab) === :qudit ? :generator : :qudit
    dst = typeof(tab).name.wrapper(tab.d, tab.n; storephase = tab.storephase,
                                   eltype = tableau_eltype(tab), layout = other)
    return copy!(dst, tab)
end

# Apply `nops` operations drawn from `oprng` to every tableau in `tabs`, the
# first being the `eltype = Int`, generator-major reference, each with its own
# copy of `seed` as RNG, and compare every tableau with the reference after
# every operation: generators, `pivcol_of_row`, the RNG state and packed
# padding (`st_padding_zero`) included. Before each operation every other
# tableau changes its layout with probability 1/8 (`st_switch!`, drawn from an
# RNG of its own), so `tabs` is updated in place. Returns the number of
# comparisons that failed.
function st_run!(tabs, seed, oprng, nops)
    ref = tabs[1]
    rngs = [Xoshiro(seed) for _ in tabs]
    swrng = Xoshiro(20_000 + seed)
    layouts = [tableau_layout(tab) for tab in tabs]
    d, n = ref.d, ref.n
    bad = 0
    for i in 1:nops
        for k in 2:length(tabs)
            rand(swrng, 1:8) == 1 || continue
            tabs[k] = st_switch!(swrng, tabs[k])
            layouts[k] = tableau_layout(tabs[k])
        end
        op = st_rand_op(oprng, ref)
        vals = [st_call(op, tab, r) for (tab, r) in zip(tabs, rngs)]
        Sref = stabilizer_matrix(ref)
        Dref = ref isa DestabilizerTableau ? destabilizer_matrix(ref) : nothing
        ok = true
        for (k, tab) in enumerate(tabs)
            same = stabilizer_matrix(tab) == Sref && tab.m == ref.m &&
                   tab.iscanonical == ref.iscanonical && isequal(vals[k], vals[1]) &&
                   tab.pivcol_of_row == ref.pivcol_of_row && rngs[k] == rngs[1]
            Dref === nothing || (same &= destabilizer_matrix(tab) == Dref)
            tab isa DestabilizerTableau && (same &= tab.xdotz_cache[1:tab.m] == st_xdotz(tab))
            same &= tableau_layout(tab) === layouts[k]
            same &= st_capacity_zero(tab) && st_padding_zero(tab)
            if !same
                ok = false
                @error "differs from the Int reference" i eltype = tableau_eltype(tab) layout = tableau_layout(tab) vals[k] vals[1]
            end
        end
        if ref isa DestabilizerTableau && st_duality(ref) != Matrix{Int64}(I, ref.m, ref.m)
            ok = false
            @error "the Int reference breaks destabilizer duality" i
        end
        if d == 2 && ref.storephase &&
           !all(mod(Sref[2n+1, j], 2) ==
                mod(sum(Sref[q, j] * Sref[n+q, j] for q in 1:n), 2) for j in 1:ref.m)
            ok = false
            @error "the Int reference has a non-Hermitian qubit generator" i
        end
        ok || (bad += 1)
        ok || break
    end
    return bad
end

# Whether every padding bit of a packed tableau is zero: the parent words
# outside each block's view, the bits outside the `n × n` bit square inside the
# view, the capacity generators `m+1:n` (phases and destabilizers included,
# and the `xdotz_cache` entries of a `DestabilizerTableau`), and the bits past
# the length of each plane. Always `true` for unpacked storage.
st_padding_zero(tab) = tableau_eltype(tab) === Bits ? st_bits_padding_zero(tab) : true

function st_bits_padding_zero(tab)
    n, m = tab.n, tab.m
    blocks = tab isa DestabilizerTableau ?
        (tab.stab_x, tab.stab_z, tab.destab_x, tab.destab_z) : (tab.stab_x, tab.stab_z)
    ok = true
    for A in blocks
        P = parent(A)
        nw = size(A, 1)
        cols = parentindices(A)[2]
        ok &= size(A, 2) == 64nw
        ok &= all(iszero(P[r, c]) for r in axes(P, 1), c in axes(P, 2)
                  if r > nw || !(c in cols))
        for c in axes(A, 2), w in 1:nw
            lo = 64 * (w - 1)                 # bits lo+1:lo+64 of this word
            valid = c > n ? 0 : clamp(n - lo, 0, 64)
            ok &= valid == 64 || A[w, c] >> valid == 0
        end
    end
    S = stabilizer_matrix(tab)
    ok &= all(iszero, S[:, m+1:end])
    if tab isa DestabilizerTableau
        ok &= all(iszero, destabilizer_matrix(tab)[:, m+1:end])
        ok &= all(iszero, tab.xdotz_cache[m+1:n])
    end
    for v in (tab.phase, tab.xdotz_cache)
        r = v.len % 64
        ok &= size(v.planes, 1) == cld(v.len, 64)
        r == 0 || isempty(v.planes) || (ok &= all(v.planes[end, b] >> r == 0 for b in axes(v.planes, 2)))
    end
    return ok
end

# Constructor calls with a literal `eltype`, wrapped so that `@inferred` sees the
# constant the way a caller's compiled code does.
st_new_u8(::Type{TT}, n) where {TT} = TT(3, n; eltype = UInt8)
st_new_u16(::Type{TT}, n) where {TT} = TT(257, n; eltype = UInt16)
st_new_int(::Type{TT}, d, n) where {TT} = TT(d, n; eltype = Int)
st_new_state_u8(::Type{TT}, n) where {TT} = TT(3, n, :product; eltype = UInt8)
st_new_raw_u8(::Type{TT}, M) where {TT} = TT(3, M; eltype = UInt8)
st_new_raw_int(::Type{TT}, d, M) where {TT} = TT(d, M; eltype = Int)
st_new_bits(::Type{TT}, n) where {TT} = TT(2, n; eltype = Bits)
st_new_raw_bits(::Type{TT}, M) where {TT} = TT(2, M; eltype = Bits, layout = :qudit)
