using MixedModels
using Test

using MixedModels: dataset

@testset "formula misspecification" begin
    dyesetuff = dataset(:dyestuff)
    @test MixedModel(@formula(yield ~ 0 + (1|batch)), dyesetuff) isa LinearMixedModel
    @test MixedModel(@formula(yield ~ 1 + (1|batch)), dyestuff) isa LinearMixedModel
    @test_throws r"Formula contains no random effects"  MixedModel(@formula(yield ~ 0 + batch), dyestuff)
    @test_throws r"Formula contains no random effects"  MixedModel(@formula(yield ~ 1), dyestuff)

    @test MixedModel(@formula(yield ~ 0 + (1|batch)), dyestuff, Poisson()) isa GeneralizedLinearMixedModel
    @test MixedModel(@formula(yield ~ 1 + (1|batch)), dyestuff, Poisson()) isa GeneralizedLinearMixedModel
    @test_throws r"Formula contains no random effects"  MixedModel(@formula(yield ~ 0 + batch), dyestuff, Poisson())
    @test_throws r"Formula contains no random effects"  MixedModel(@formula(yield ~ 1), dyestuff, Poisson())
end
