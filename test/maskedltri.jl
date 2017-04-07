using MixedModels, Base.Test

@testset "paramtri" begin
    m1 = MixedModels.MaskedLowerTri(LowerTriangular(eye(1)), [1])
    @test getθ(m1) == [1.]
    @test lowerbd(m1) == [0.]
    @test MixedModels.nθ(m1) == 1

    m2 = MixedModels.MaskedLowerTri(LowerTriangular(eye(3)), [1,2,3,5,6,9])
    @test getθ(m2) == [1.,0.,0.,1.,0.,1.]
    @test lowerbd(m2) == [0., -Inf, -Inf, 0., -Inf, 0.]
    setθ!(m2, [1.:6;])
    @test m2.m == reshape([1.,2,3,0,4,5,0,0,6], (3,3))

    m3 = MixedModels.MaskedLowerTri(LowerTriangular(eye(3)), [1,5,9])
    @test getθ(m3) == ones(3)
    @test lowerbd(m3) == zeros(3)
    setθ!(m3, [1.:3;])
    @test m3.m == reshape([1.,0,0,0,2,0,0,0,3], (3,3))
end
