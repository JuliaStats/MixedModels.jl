using MixedModels
using Suppressor
using Test

@testset "linear, and lmm wrapper" begin
    m1 = fit(MixedModel, @formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff); progress=false)
    @test first(m1.θ) ≈ 0.7525806757718846 rtol=1.0e-5
    m2 = lmm(@formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff); progress=false)
    @test isa(m2, LinearMixedModel)
    @test first(m2.θ) ≈ 0.7525806757718846 rtol=1.0e-5
    @test deviance(m1) ≈ deviance(m2)
    @test isa(lmm(@formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff); progress=false, REML = true), LinearMixedModel)

    @test mss(m1) ≈ 30780.69 atol=0.005
    m0 = fit(MixedModel, @formula(yield ~ 0 + (1|batch)), MixedModels.dataset(:dyestuff); progress=false)
    @test_throws ArgumentError("Mean sum of squares is defined only for models with an intercept term.") mss(m0)

    # example from https://github.com/JuliaStats/MixedModels.jl/issues/194
    # copied from tetst/pls.jl
    data = (
        a = [1.55945122,0.004391538,0.005554163,-0.173029772,4.586284429,0.259493671,-0.091735715,5.546487603,0.457734831,-0.030169602],
        b = [0.24520519,0.080624178,0.228083467,0.2471453,0.398994279,0.037213859,0.102144973,0.241380251,0.206570975,0.15980803],
        c = PooledArray(["H","F","K","P","P","P","D","M","I","D"]),
        w1 = [20,40,35,12,29,25,65,105,30,75],
        w2 = [0.04587156,0.091743119,0.080275229,0.027522936,0.066513761,0.05733945,0.149082569,0.240825688,0.068807339,0.172018349],
    )
    m2 = lmm(@formula(a ~ 1 + b + (1|c)), data; wts = data.w1, progress=false)
    @test m2.θ ≈ [0.295181729258352]  atol = 1.e-4
    @test stderror(m2) ≈  [0.9640167, 3.6309696] atol = 1.e-4
    @test vcov(m2) ≈ [0.9293282 -2.557527; -2.5575267 13.183940] atol = 1.e-4
end

@testset "generalized" begin
    gm1 = fit(MixedModel, @formula(use ~ 1 + urban + livch + age + abs2(age) + (1|dist)),
              MixedModels.dataset(:contra), Bernoulli(); progress=false)
    @test deviance(gm1) ≈ 2372.7286 atol=1.0e-3

    gm2 = glmm(@formula(use ~ 1 + urban + livch + age + abs2(age) + (1|dist)),
    MixedModels.dataset(:contra), Bernoulli(); progress=false)
    @test deviance(gm2) ≈ 2372.7286 atol=1.0e-3
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
