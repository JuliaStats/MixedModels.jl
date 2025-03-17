using MixedModels
using Suppressor
using Test

@testset "linear, and lmm wrapper" begin
    m1 = fit(MixedModel, @formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff); progress=false)
    @test first(m1.θ) ≈ 0.7525806757718846 rtol=1.0e-5
    m2 = lmm(@formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff); progress=false)
    @test isa(m2, LinearMixedModel)
    @test first(m2.θ) ≈ 0.7525806757718846 rtol=1.0e-5
    @test deviance(m1) ≈ deviance(m2) #TODO: maybe add an `rtol`?
end

@testset "generalized" begin
    gm1 = fit(MixedModel, @formula(use ~ 1 + urban + livch + age + abs2(age) + (1|dist)),
              MixedModels.dataset(:contra), Bernoulli(); progress=false)
    @test deviance(gm1) ≈ 2372.7286 atol=1.0e-3
end

@testset "Normal-IdentityLink" begin
    @test isa(fit(MixedModel, @formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff), Normal(); progress=false),
              LinearMixedModel)
    @test_throws(ArgumentError("use LinearMixedModel for Normal distribution with IdentityLink"),
                 fit(GeneralizedLinearMixedModel,
                     @formula(yield ~ 1 + (1|batch)),
                     MixedModels.dataset(:dyestuff); progress=false))
end

@testset "Normal Distribution GLMM" begin
    @test @suppress isa(fit(MixedModel, @formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff),
                         Normal(), SqrtLink(); progress=false),
                        GeneralizedLinearMixedModel)
end
