using MixedModels
using Test

using MixedModels: dataset

@testset "formula misspecification" begin
    dyestuff = dataset(:dyestuff)

    @test MixedModel(@formula(yield ~ 0 + (1|batch)), dyestuff) isa LinearMixedModel
    @test MixedModel(@formula(yield ~ 1 + (1|batch)), dyestuff) isa LinearMixedModel
    @test_throws MixedModels._MISSING_RE_ERROR MixedModel(@formula(yield ~ 0 + batch), dyestuff)
    @test_throws MixedModels._MISSING_RE_ERROR MixedModel(@formula(yield ~ 1), dyestuff)

    @test MixedModel(@formula(yield ~ 0 + (1|batch)), dyestuff, Poisson()) isa GeneralizedLinearMixedModel
    @test MixedModel(@formula(yield ~ 1 + (1|batch)), dyestuff, Poisson()) isa GeneralizedLinearMixedModel
    @test_throws MixedModels._MISSING_RE_ERROR  MixedModel(@formula(yield ~ 0 + batch), dyestuff, Poisson())
    @test_throws MixedModels._MISSING_RE_ERROR  MixedModel(@formula(yield ~ 1), dyestuff, Poisson())
end
