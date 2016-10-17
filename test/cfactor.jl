@testset "cfactor" begin
    de3 = Diagonal(ones(3))
    de2 = Diagonal(ones(2))
    dr3 = Diagonal(randn(3))

    @test_throws DimensionMismatch MixedModels.downdate!(dr3,de2,de3)
    MixedModels.downdate!(dr3,de3,de3)
    sp10 = sprand(10,3,0.2)
    MixedModels.downdate!(rand(3,3),sp10,sp10)
    MixedModels.downdate!(rand(3,3),sp10)
end
