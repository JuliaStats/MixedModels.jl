using LinearAlgebra
using MixedModels
using Random
using SparseArrays
using StatsModels
using Test

const LMM = LinearMixedModel

@testset "UBlk" begin
    ex22 = UniformBlockDiagonal(reshape(Vector(1.0:12.0), (2, 2, 3)))
    Lblk = UniformBlockDiagonal(fill(0., (2,2,3)))
    ds = (Y = rand(12), A = repeat(['N','Y'], outer=6), G = repeat('a':'c', inner=4),
        H = repeat('A':'B', outer=6), U = repeat([-1,0,1], inner=2, outer=2))
    sch = schema(ds, Dict(:A=>EffectsCoding()))
    vf1 = modelcols(apply_schema(@formula(Y ~ 1 + A + (1+A|G)), sch, LMM), ds)[2][2]
    vf2 = modelcols(apply_schema(@formula(Y ~ 1 + U + (1+U|H)), sch, LMM), ds)[2][2]
    prd = vf2'vf1

    @testset "size" begin
        @test size(ex22) == (6, 6)
        @test size(ex22, 1) == 6
        @test size(ex22, 2) == 6
        @test size(ex22.data) == (2, 2, 3)
    #    @test length(ex22.facevec) == 3
        @test size(vf1) == (12, 6)
        @test size(vf2) == (12, 4)
        @test size(prd) == (4, 6)
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
        @test view(ex22.data, :, :, 3) == reshape(9:12, (2,2))
    end

    @testset "copyscaleinflate" begin
        MixedModels.copyscaleinflate!(Lblk, ex22, vf1)
        @test view(Lblk.data, :, :, 1) == [2. 3.; 2. 5.]
        setθ!(vf1, [1.,1.,1.])
        Λ = vf1.λ
        MixedModels.copyscaleinflate!(Lblk, ex22, vf1)
        target = Λ'view(ex22.data, :, :, 1)*Λ + I
        @test view(Lblk.data, :, :, 1) == target
    end

    @testset "updateL" begin
        @test ones(2, 2) == MixedModels.rankUpdate!(Hermitian(zeros(2, 2)), ones(2), 1., 1.)
        d3 = MixedModels.dataset(:d3)
        sch = schema(d3)
        vf1 = modelcols(apply_schema(@formula(y ~ 1 + u + (1+u|g)), sch, LMM), d3)[2][2]
        vf2 = modelcols(apply_schema(@formula(y ~ 1 + u + (1+u|h)), sch, LMM), d3)[2][2]
        @test vf1.λ == LowerTriangular(Matrix(I, 2, 2))
        setθ!(vf2, [1.75, 0.0, 1.0])
        A11 = vf1'vf1
        L11 = MixedModels.cholUnblocked!(MixedModels.copyscaleinflate!(UniformBlockDiagonal(fill(0., size(A11.data))), A11, vf1), Val{:L})
        L21 = vf2'vf1
        @test isa(L21, BlockedSparse)
        @test L21[1,1] == 30.0
        @test size(L21) == (344, 9452)
        @test size(L21, 1) == 344
        MixedModels.lmulΛ!(vf2', MixedModels.rmulΛ!(L21, vf1))
        @test size(Matrix(L21)) == size(sparse(L21))
#        L21cb1 = copy(L21.colblocks[1])
#        @test L21cb1 == Vf2.Λ * A21cb1 * Vf1.Λ
#        rdiv!(L21, adjoint(LowerTriangular(L11)))
#        @test_broken L21.colblocks[1] == rdiv!(L21cb1, adjoint(LowerTriangular(L11.facevec[1])))
         A22 = vf2'vf2
         L22 = MixedModels.copyscaleinflate!(UniformBlockDiagonal(fill(0., size(A22.data))), A22, vf2)
    end
end
