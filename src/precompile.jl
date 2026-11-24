# Precompile workload: runs only while the package image is generated, and can
# be disabled through PrecompileTools' Preferences. It draws every random
# number from a local `Xoshiro`; the methods that default to the global RNG are
# compiled by `precompile` without being called, or are called only where they
# draw nothing (a deterministic measurement).

PrecompileTools.@setup_workload begin
    rng = Random.Xoshiro(1)
    ops(d) = (SinglePauli(1, 1, 0), DoublePauli(1, 0, 1, 2, 0, d - 1),
              TriplePauli(1, 1, 0, 2, 1, 0, 3, 1, 0), NPauli((1, 2), (1, 1), (0, 0)),
              GeneralPauli([1, 0, 0, 0, 0, 0, 0, 0], 0))
    gates = (Fourier(1), Phase(2), Multiplier(1, 1), PauliGate(1, 1, 0),
             SUM(1, 2), CPhase(2, 3), SWAP(1, 4))
    # Takes its tableaux built, so that it is compiled for their concrete
    # types only: built inside, their type would depend on the runtime `d`,
    # and inference would compile the calls for every storage type.
    function workload(rng, tab, product, mixed)
        d = tab.d
        for op in ops(d)
            measure!(rng, tab, op)
            measure!(rng, tab, op; outcome=0)
            expect!(tab, op); expect_int!(tab, op); dephase!(copy(tab), op)
        end
        # Default RNG: Z on a Z-basis product state is deterministic, so no
        # number is drawn.
        measure!(product, SinglePauli(1, 0, 1))
        # The commuting branches: an appended generator, then a fixed outcome.
        measure!(rng, mixed, SinglePauli(1, 0, 1)); measure!(rng, mixed, SinglePauli(1, 0, 1))
        expect_int!(mixed, GeneralPauli([0, 0, 0, 0, 1, 0, 0, 0], 0))
        for g in gates
            apply!(tab, g)
        end
        U = random_clifford(rng, d, 1:2)
        random_clifford(rng, d, [1, 2])
        apply!(tab, random_clifford!(rng, U)); apply!(tab, inv(U) ∘ U)
        conjugate(U, SinglePauli(1, 1, 0), 4)
        canonicalize!(tab); is_pure(tab); tab == copy(tab); hash(tab)
        stabilizer_matrix(tab)
        tab isa DestabilizerTableau && destabilizer_matrix(tab)
        entanglement_entropy(tab, [1, 2]); entanglement_profile(tab)
        mutual_information(tab, [1], [2]); entanglement_negativity(tab, [1], [2])
        random_state!(rng, tab; m=2)
        measure!(rng, tab, SinglePauli(4, 0, 1))
        traceout!(tab, 1); reset!(tab, 1); reset!(tab; state=:product)
        sprint(show, tab); sprint(show, MIME"text/plain"(), tab)
        precompile(random_state!, (typeof(tab),))
        return tab
    end
    # The packed tableaux are built without the keyword constructor. Called
    # from this block with `eltype` (or with `storephase`), it made loading the
    # package about 1.1 s slower on Julia 1.13, spent verifying a method call
    # edge; with `state` alone it does not.
    packed(TT, state, layout) = _packed_preset_tableau(TT, 4, state, state === :product ? :Z : nothing,
                                                       true, PrecomputedInvMod(2), layout)
    PrecompileTools.@compile_workload begin
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            workload(rng, TT(d, 4; state=:ghz), TT(d, 4; state=:product),
                     TT(d, 4; state=:mixed))
        end
        # Packed storage, in both layouts; the switch between them.
        for TT in (StabilizerTableau, DestabilizerTableau), layout in (:generator, :qudit)
            tab = workload(rng, packed(TT, :ghz, layout), packed(TT, :product, layout),
                           packed(TT, :mixed, layout))
            qudit_major!(tab); generator_major!(tab)
        end
        # The loops above store UInt8 and Bits; d = 257 stores UInt16. Int
        # storage (`eltype = Int`) is left out: caching it beside UInt8
        # noticeably slows loading the package.
        for TT in (StabilizerTableau, DestabilizerTableau)
            tab = TT(257, 4; state=:ghz)
            apply!(tab, Fourier(1)); apply!(tab, SUM(1, 2))
            measure!(rng, tab, SinglePauli(1, 1, 0))
            measure!(rng, tab, SinglePauli(2, 0, 1); outcome=0)
            entanglement_entropy(tab, [1, 2])
        end
        # The calls above inline these constructors into the workload, so
        # they are cached on their own for user code.
        precompile(SinglePauli, ntuple(_ -> Int, 3))
        precompile(DoublePauli, ntuple(_ -> Int, 6))
        precompile(TriplePauli, ntuple(_ -> Int, 9))
        precompile(random_clifford, (Int, UnitRange{Int}))
        precompile(random_clifford, (Int, Vector{Int}))
        precompile(random_clifford!, (CliffordOperator,))
    end
end
