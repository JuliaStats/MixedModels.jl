using Base.Test, CategoricalArrays, RData, MixedModels

if !isdefined(:dat) || !isa(dat, Dict{Symbol, Any})
    dat = convert(Dict{Symbol,Any}, load(joinpath(dirname(@__FILE__), "dat.rda")))
end

@testset "UBlk" begin
    ex22 = UniformBlockDiagonal(reshape(Vector(1.0:12.0), (2, 2, 3)))
    Lblk = UniformBlockDiagonal(fill(0., (2,2,3)))
    vf1 = VectorFactorReTerm(categorical(repeat(1:3, inner=4)),
        hcat(ones(12), repeat([-1.0, 1.0], outer=6))', :G, ["(Intercept)", "U"], [2])
    vf2 = VectorFactorReTerm(categorical(repeat(['A','B'], outer=6)),
        hcat(ones(12), repeat([-1.0, 0.0, 1.0], inner=2, outer=2))', :G, ["(Intercept)", "U"], [2])
    prd = vf2'vf1
    
    @testset "size" begin
        @test size(ex22) == (6, 6)
        @test size(ex22, 1) == 6
        @test size(ex22, 2) == 6
        @test size(ex22.data) == (2, 2, 3)
        @test length(ex22.facevec) == 3
        @test size(vf1) == (12, 6)
        @test size(vf2) == (12, 4)
        @test size(prd) == (4, 6)
        @test nnz(prd) == 24
    end

    @testset "elements" begin
        @test ex22[1, 1] == 1
        @test ex22[2, 1] == 2
        @test ex22[3, 1] == 0
        @test ex22[2, 2] == 4
        @test ex22[3, 3] == 5
        @test ex22[:, 3] == [0,0,5,6,0,0]
        @test ex22[5, 6] == 11
    end

    @testset "facevec" begin
        @test ex22.facevec[3] == reshape(9:12, (2,2))
    end

    @testset "scaleInflate" begin
        MixedModels.scaleInflate!(Lblk, ex22, vf1)
        @test Lblk.facevec[1] == [2. 3.; 2. 5.]
        setθ!(vf1, [1.,1.,1.])
        Λ = vf1.Λ
        MixedModels.scaleInflate!(Lblk, ex22, vf1)
        target = Λ'ex22.facevec[1]*Λ + I
        @test Lblk.facevec[1] == target
        @test MixedModels.scaleInflate!(full(Lblk), ex22, vf1)[1:2, 1:2] == target
    end

    @testset "updateL" begin
        @test ones(2, 2) == MixedModels.rankUpdate!(1.0, ones(2), Hermitian(zeros(2, 2)))
        d3 = dat[:d3]
        Vf1 = VectorFactorReTerm(d3[:G], hcat(ones(130418), d3[:U])', :G, ["(Intercept)", "U"], [2])
        Vf2 = VectorFactorReTerm(d3[:H], hcat(ones(130418), d3[:U])', :H, ["(Intercept)", "U"], [2])
        @test getΛ(Vf1) == LowerTriangular(eye(2))
        setθ!(Vf2, [1.75, 0.0, 1.0])
        A11 = Vf1'Vf1
        L11 = MixedModels.cholUnblocked!(MixedModels.scaleInflate!(UniformBlockDiagonal(fill(0., size(A11.data))), A11, Vf1), Val{:L})
        L21 = Vf2'Vf1
        A21cb1 = copy(L21.colblocks[1])
        MixedModels.Λc_mul_B!(Vf2, MixedModels.A_mul_Λ!(L21, Vf1))
        L21cb1 = copy(L21.colblocks[1])
        @test L21cb1 == Vf2.Λ * A21cb1 * Vf1.Λ
        Base.LinAlg.A_rdiv_Bc!(L21, LowerTriangular(L11))
        @test L21.colblocks[1] == Base.LinAlg.A_rdiv_Bc!(L21cb1, LowerTriangular(L11.facevec[1]))
        A22 = Vf2'Vf2
        L22 = MixedModels.scaleInflate!(UniformBlockDiagonal(fill(0., size(A22.data))), A22, Vf2)
        for b in L21.rowblocks[1]
            BLAS.syr!('L', -1.0, b, L22.facevec[1])
        end
    end
end
