using MixedModels, Test

@testset "GHnorm" begin
    gh2 = GHnorm(2)
    @test gh2.z  == [-1.0, 1.0]
    @test gh2.w == [0.5, 0.5]
    @test GHnorm(2) === gh2
    gh9 = GHnorm(9.0)
    @test sum(gh9.w) ≈ 1
    @test length(gh9) == 9
end
