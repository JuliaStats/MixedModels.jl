using MixedModels
using Test

using MixedModels: dataset, likelihoodratiotest
using GLM: ProbitLink
using StatsModels: lrtest, isnested

include("modelcache.jl")

@testset "isnested" begin
    slp = dataset(:sleepstudy)

    # these tests don't actually depend on the models being fit, 
    # so we just construct them

    # mismatched RE terms 
    m1 = LinearMixedModel(@formula(reaction ~ 1 + days + (1+days|subj)), slp)
    m2 = LinearMixedModel(@formula(reaction ~ 1 + days + (0+days|subj)), slp)
    @test !isnested(m1, m2)

    # mismatched FE
    m1 = LinearMixedModel(@formula(reaction ~ 1 + days + (1|subj)), slp)
    m2 = LinearMixedModel(@formula(reaction ~ 0 + days + (1|subj)), slp)
    @test !isnested(m1, m2)

    # mismatched grouping vars
    kb07  = dataset(:kb07)
    m1 = LinearMixedModel(@formula(rt_trunc ~ 1 + (1|subj)), kb07)
    m2 = LinearMixedModel(@formula(rt_trunc ~ 1 + (1|item)), kb07)
    @test !isnested(m1, m2)

    # fixed-effects specification in REML and 
    # conversion of internal ArgumentError into @error for StatsModels.isnested
    kb07  = dataset(:kb07)
    m1 = fit(MixedModel, @formula(rt_trunc ~ 1 + prec + (1|subj)), kb07, REML=true)
    m2 = fit(MixedModel, @formula(rt_trunc ~ 1 + prec + (1+prec|subj)), kb07, REML=true)
    @test isnested(m1, m2)
    m2 = fit(MixedModel, @formula(rt_trunc ~ 1 + (1+prec|subj)), kb07, REML=true)
    @test !isnested(m1, m2)

end

@testset "likelihoodratio test" begin
    slp = dataset(:sleepstudy);
    


    fm0 = fit(MixedModel,@formula(reaction ~ 1 + (1+days|subj)),slp);
    fm1 = fit(MixedModel,@formula(reaction ~ 1 + days + (1+days|subj)),slp);

    lrt = likelihoodratiotest(fm0,fm1);

    @test [deviance(fm0), deviance(fm1)] == lrt.deviance
    @test deviance(fm0) - deviance(fm1) == only(lrt.tests.deviancediff)
    @test only(lrt.tests.dofdiff) == 1
    @test sum(map(length,lrt.tests)) == 3
    @test sum(map(length,lrt.pvalues)) == 1
    @test sum(map(length,lrt.models)) == 4
    @test length(lrt.formulae) == 2
    show(IOBuffer(),lrt);
    @test :pvalues in propertynames(lrt)


    # mix of REML and ML
    fm0 = fit(MixedModel,@formula(reaction ~ 1 + (1+days|subj)),slp, REML=true);
    @test_throws ArgumentError likelihoodratiotest(fm0,fm1)

    # differing FE with REML
    fm1 = fit(MixedModel,@formula(reaction ~ 1 + days + (1+days|subj)),slp, REML=true);
 
    @test_throws ArgumentError likelihoodratiotest(fm0,fm1)

    contra = MixedModels.dataset(:contra);
    gm0 = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urbdist)), contra, Bernoulli(), fast=true);
    gm1 = fit(MixedModel, @formula(use ~ 1+age+abs2(age)+urban+livch+(1|urbdist)), contra, Bernoulli(), fast=true);
    lrt = likelihoodratiotest(gm0,gm1);
    @test [deviance(gm0), deviance(gm1)] == lrt.deviance
    @test deviance(gm0) - deviance(gm1) == only(lrt.tests.deviancediff)
    @test first(lrt.tests.dofdiff) == 1
    @test sum(length, lrt.tests) == 3
    @test sum(length, lrt.pvalues) == 1
    @test sum(length, lrt.models) == 4
    @test length(lrt.formulae) == 2

    # mismatched links
    gm_probit = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urbdist)), contra, Bernoulli(), ProbitLink(), fast=true);
    @test_throws ArgumentError likelihoodratiotest(gm0,gm_probit)

    # mismatched families
    gm_poisson = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urbdist)), contra, Poisson(), fast=true);
    @test_throws ArgumentError MixedModels.likelihoodratiotest(gm0,gm_poisson)
end
