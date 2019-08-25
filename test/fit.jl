using MixedModels, RData, Test

if !@isdefined(dat) || !isa(dat, Dict{Symbol, DataFrame})
    const dat = Dict(Symbol(k) => v for (k, v) in
        load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")))
end

@testset "linear" begin
    m1 = fit(MixedModel, @formula(Y ~ 1 + (1|G)), dat[:Dyestuff])
    @test first(m1.θ) ≈ 0.7525806757718846 rtol=1.0e-5
end

@testset "generalized" begin
    gm1 = fit(MixedModel, @formula(use ~ 1 + urb + l + a + abs2(a) + (1|d)),
              dat[:Contraception], Bernoulli())
    @test deviance(gm1) ≈ 2372.7286 atol=1.0e-3
end

@testset "Normal-IdentityLink" begin
    @test isa(fit(MixedModel, @formula(Y ~ 1 + (1|G)), dat[:Dyestuff], Normal()),
              LinearMixedModel)
    @test_throws(ArgumentError("use LinearMixedModel for Normal distribution with IdentityLink"),
                 fit(GeneralizedLinearMixedModel,
                     @formula(Y ~ 1 + (1|G)),
                     dat[:Dyestuff]))
end

@testset "Normal Distribution GLMM" begin
    @test_broken(isa(fit(MixedModel, @formula(Y ~ 1 + (1|G)), dat[:Dyestuff],
                         Normal(), SqrtLink),
                     GeneralizedLinearMixedModel))
end
