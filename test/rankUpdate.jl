using Base.Test, MixedModels

# methods that are not otherwise tested
@testset "rankUpdate" begin
    @test ones(2, 2) == MixedModels.rankUpdate!(1.0, ones(2), Hermitian(zeros(2, 2)))
end
