using QuditClifford, Test, Random

const st_QC = QuditClifford

st_types(d) = filter(E -> st_QC._eltype_admissible(E, d), (UInt8, UInt16, Int))

# A signed integer type wider than one machine word on neither platform, so
# never `Int` itself.
const st_OtherSigned = Int === Int64 ? Int32 : Int64

@testset "storage element types" begin

@testset "admissible element types" begin
    @test st_QC._default_eltype.((2, 3, 251, 257, 65521, 65537)) ==
          (UInt8, UInt8, UInt8, UInt16, UInt16, Int)
    for TT in (StabilizerTableau, DestabilizerTableau)
        @test_throws ArgumentError TT(257, 2; eltype = UInt8)
        @test_throws ArgumentError TT(3, 2; eltype = st_OtherSigned)
        @test_throws ArgumentError TT(3, 2; eltype = UInt32)
        @test_throws ArgumentError TT(3, 2; eltype = Float64)
        @test_throws ArgumentError TT(257, [0 1; 1 0; 0 0; 0 0; 0 0]; eltype = UInt8)
        @test typeof(TT(3, 2; eltype = UInt16)).parameters[2] === UInt16
        @test typeof(TT(3, 2)).parameters[2] === Int
        @test st_QC._storage_eltype(TT(3, 2; eltype = UInt8)) === UInt8
    end
end

@testset "presets in every element type" begin
    n = 3
    for TT in (StabilizerTableau, DestabilizerTableau), d in (2, 3, 251, 257),
        E in st_types(d), storephase in (true, false),
        (state, basis) in ((:mixed, :Z), (:product, [:X, :Y, :Z]), (:ghz, :Z))

        tab = TT(d, n; state, basis, storephase, eltype = E)
        ref = TT(d, n; state, basis, storephase, eltype = Int)
        @test stabilizer_matrix(tab) == stabilizer_matrix(ref)
        @test eltype(stabilizer_matrix(tab)) === Int
        @test size(tab.stab_x) == (n, n)
        @test Base.eltype(tab.stab_x) === E && Base.eltype(tab.stab_z) === E
        if TT === DestabilizerTableau
            @test destabilizer_matrix(tab) == destabilizer_matrix(ref)
            @test eltype(destabilizer_matrix(tab)) === Int
            @test Base.eltype(tab.destab_x) === E && Base.eltype(tab.destab_z) === E
        end
    end
end

@testset "raw input reduced before narrowing" begin
    S = [-1 0; 252 3; 0 -251; 1 1; 5 7]
    for (d, M) in ((251, S), (257, S .+ 257), (257, S .- 257))
        ref = stabilizer_matrix(StabilizerTableau(d, M; eltype = Int, check = false))
        @test all(0 .≤ ref[1:4, :] .< d)
        for E in st_types(d)
            tab = StabilizerTableau(d, M; eltype = E, check = false)
            @test Base.eltype(tab.stab_x) === E
            @test stabilizer_matrix(tab) == ref
            fresh = StabilizerTableau(d, 2; eltype = E)
            st_QC._load_stabilizer_matrix!(fresh, M)
            @test stabilizer_matrix(fresh) == ref
        end
    end
    # The destabilizer loader reduces as well.
    for E in st_types(251)
        tab = DestabilizerTableau(251, 2; eltype = E)
        st_QC._load_destabilizer_matrix!(tab, [-1 0; 252 3; 0 -251; 1 1])
        @test destabilizer_matrix(tab) == [250 0; 1 3; 0 0; 1 1]
    end
end

@testset "copy and copy! keep the element type" begin
    for TT in (StabilizerTableau, DestabilizerTableau), E in (UInt8, UInt16)
        tab = TT(3, 4; state = :ghz, eltype = E)
        c = copy(tab)
        @test typeof(c) === typeof(tab)
        @test stabilizer_matrix(c) == stabilizer_matrix(tab)
        @test c.stab_x !== tab.stab_x

        dst = TT(3, 4; eltype = E)
        @test copy!(dst, tab) === dst
        @test stabilizer_matrix(dst) == stabilizer_matrix(tab) && dst.m == tab.m
        if TT === DestabilizerTableau
            @test destabilizer_matrix(dst) == destabilizer_matrix(tab)
        end

        other = TT(3, 4; eltype = E === UInt8 ? UInt16 : UInt8)
        @test_throws ArgumentError copy!(other, tab)
        @test_throws ArgumentError copy!(TT(3, 4; eltype = Int), tab)

        reset!(tab; state = :product)
        @test typeof(tab) === typeof(c)
        @test Base.eltype(tab.stab_x) === E
        @test stabilizer_matrix(tab) ==
              stabilizer_matrix(TT(3, 4; state = :product, eltype = Int))
    end
end

@testset "equality and hash ignore the element type" begin
    for d in (3, 251), TT in (StabilizerTableau, DestabilizerTableau)
        a = TT(d, 4; state = :ghz, eltype = UInt8)
        b = TT(d, 4; state = :ghz, eltype = Int)
        @test a == b
        @test hash(a) == hash(b)
        @test length(Set([a, b])) == 1
        apply!(b, SUM(1, 2))
        @test a != b
    end
end

end
