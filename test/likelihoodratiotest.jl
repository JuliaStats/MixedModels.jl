using MixedModels, Test
import GLM: ProbitLink

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
    show(IOBuffer(),lrt);


    # mix of REML and ML
    fm0 = fit(MixedModel,@formula(reaction ~ 1 + (1+days|subj)),slp, REML=true);
    @test_throws ArgumentError MixedModels.likelihoodratiotest(fm0,fm1)

    # differing FE with REML
    fm1 = fit(MixedModel,@formula(reaction ~ 1 + days + (1+days|subj)),slp, REML=true);
    fm10 = fit(MixedModel,@formula(reaction ~ 1 + days + (1|subj)),slp, REML=true);
    @test_throws ArgumentError MixedModels.likelihoodratiotest(fm0,fm1);

    contra = MixedModels.dataset(:contra);
    gm0 = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urbdist)), contra, Bernoulli(), fast=true);
    gm1 = fit(MixedModel, @formula(use ~ 1+age+abs2(age)+urban+livch+(1|urbdist)), contra, Bernoulli(), fast=true);
    lrt = MixedModels.likelihoodratiotest(gm0,gm1);
    @test [deviance(gm0), deviance(gm1)] == lrt.deviance
    @test deviance(gm0) - deviance(gm1) == first(lrt.tests.deviancediff)
    @test first(lrt.tests.dofdiff) == 1
    @test sum(map(length,lrt.tests)) == 3
    @test sum(map(length,lrt.pvalues)) == 1
    @test sum(map(length,lrt.models)) == 4
    @test length(lrt.formulae) == 2

    # mismatched links
    gm_probit = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urbdist)), contra, Bernoulli(), ProbitLink(), fast=true);
    @test_throws ArgumentError MixedModels.likelihoodratiotest(gm0,gm_probit)

    # mismatched families
    gm_poisson = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urbdist)), contra, Poisson(), fast=true);
    @test_throws ArgumentError MixedModels.likelihoodratiotest(gm0,gm_poisson)
end
