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
    PrecompileTools.@compile_workload begin
        for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3)
            tab = TT(d, 4; state=:ghz)
            TT(d, 4; state=:product)
            for op in ops(d)
                measure!(rng, tab, op)
                measure!(rng, tab, op; outcome=0)
                expect!(tab, op); expect_int!(tab, op); dephase!(copy(tab), op)
            end
            # Default RNG: Z on a Z-basis product state is deterministic, so
            # no number is drawn.
            measure!(TT(d, 4; state=:product), SinglePauli(1, 0, 1))
            for g in gates
                apply!(tab, g)
            end
            U = random_clifford(rng, d, 1:2)
            random_clifford(rng, d, [1, 2])
            apply!(tab, random_clifford!(rng, U)); apply!(tab, inv(U) ∘ U)
            conjugate(U, SinglePauli(1, 1, 0), 4)
            canonicalize!(tab); is_pure(tab); tab == copy(tab); hash(tab)
            entanglement_entropy(tab, [1, 2]); entanglement_profile(tab)
            mutual_information(tab, [1], [2]); entanglement_negativity(tab, [1], [2])
            random_state!(rng, tab; m=2)
            measure!(rng, tab, SinglePauli(4, 0, 1))
            traceout!(tab, 1); reset!(tab, 1); reset!(tab; state=:product)
            sprint(show, tab); sprint(show, MIME"text/plain"(), tab)
            T = typeof(tab)
            precompile(random_state!, (T,))
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
