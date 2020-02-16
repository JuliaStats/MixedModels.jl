using MixedModels, Test

@testset "likelihoodratio test" begin
    slp = MixedModels.dataset(:sleepstudy);
    fm0 = fit(MixedModel,@formula(reaction ~ 1 + (1+days|subj)),slp);
    fm1 = fit(MixedModel,@formula(reaction ~ 1 + days + (1+days|subj)),slp);
    lrt = MixedModels.likelihoodratiotest(fm0,fm1);

    @test [deviance(fm0), deviance(fm1)] == lrt.deviance
    @test deviance(fm0) - deviance(fm1) == first(lrt.tests.deviancediff)
    @test first(lrt.tests.dofdiff) == 1
    @test sum(map(length,lrt.tests)) == 3
    @test sum(map(length,lrt.pvalues)) == 1
    @test sum(map(length,lrt.models)) == 4
    @test length(lrt.formulae) == 2

    fm0 = fit(MixedModel,@formula(reaction ~ 1 + (1+days|subj)),slp, REML=true);
    fm1 = fit(MixedModel,@formula(reaction ~ 1 + days + (1+days|subj)),slp, REML=true);
    fm10 = fit(MixedModel,@formula(reaction ~ 1 + days + (1|subj)),slp, REML=true);

    @test_throws ArgumentError MixedModels.likelihoodratiotest(fm0,fm1);
    lrt =  MixedModels.likelihoodratiotest(fm1,fm10);
end
