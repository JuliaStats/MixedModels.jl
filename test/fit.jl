using MixedModels, Feather, Test

@testset "linear" begin
    m1 = fit(MixedModel, @formula(yield ~ 1 + (1|batch)), testdata(:dyestuff))
    @test first(m1.θ) ≈ 0.7525806757718846 rtol=1.0e-5
end

@testset "generalized" begin
    gm1 = fit(MixedModel, @formula(use ~ 1 + urban + livch + age + abs2(age) + (1|dist)),
              testdata(:contra), Bernoulli())
    @test deviance(gm1) ≈ 2372.7286 atol=1.0e-3
end

@testset "Normal-IdentityLink" begin
    @test isa(fit(MixedModel, @formula(yield ~ 1 + (1|batch)), testdata(:dyestuff), Normal()),
              LinearMixedModel)
    @test_throws(ArgumentError("use LinearMixedModel for Normal distribution with IdentityLink"),
                 fit(GeneralizedLinearMixedModel,
                     @formula(yield ~ 1 + (1|batch)),
                     testdata(:dyestuff)))
end

@testset "Normal Distribution GLMM" begin
    @test_broken(isa(fit(MixedModel, @formula(yield ~ 1 + (1|batch)), testdata(:dyestuff),
                         Normal(), LogLink),
                     GeneralizedLinearMixedModel))
end
