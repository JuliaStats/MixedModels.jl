using Base.Test, MixedModels

const ex22 = UniformBlockDiagonal([reshape(collect(1:4) + k, (2, 2)) for k in 0:4:8])

@testset "size" begin
    @test size(ex22) == (6, 6)
    @test size(ex22, 1) == 6
    @test size(ex22, 2) == 6
    @test size(ex22.data) == (2, 2, 3)
    @test length(ex22.facevec) == 3
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