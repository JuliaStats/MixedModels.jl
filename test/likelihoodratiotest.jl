using DataFrames
using GLM
using MixedModels
using Suppressor
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
    m1 = fit(MixedModel, @formula(rt_trunc ~ 1 + prec + (1|subj)), kb07, REML=true, progress=false)
    m2 = fit(MixedModel, @formula(rt_trunc ~ 1 + prec + (1+prec|subj)), kb07, REML=true, progress=false)
    @test isnested(m1, m2)
    m2 = fit(MixedModel, @formula(rt_trunc ~ 1 + (1+prec|subj)), kb07, REML=true, progress=false)
    @test @suppress !isnested(m1, m2)
end

@testset "likelihoodratio test" begin
    slp = dataset(:sleepstudy);

    fm0 = fit(MixedModel,@formula(reaction ~ 1 + (1+days|subj)),slp, progress=false);
    fm1 = fit(MixedModel,@formula(reaction ~ 1 + days + (1+days|subj)),slp, progress=false);
    lm0 = lm(@formula(reaction ~ 1), slp)
    lm1 = lm(@formula(reaction ~ 1 + days), slp)

    @test MixedModels._iscomparable(lm0, fm1)
    @test !MixedModels._iscomparable(lm1, fm0)

    lrt = likelihoodratiotest(fm0,fm1)

    @test [deviance(fm0), deviance(fm1)] == lrt.deviance
    @test deviance(fm0) - deviance(fm1) == only(lrt.tests.deviancediff)
    @test only(lrt.tests.dofdiff) == 1
    @test sum(map(length,lrt.tests)) == 3
    @test sum(map(length,lrt.pvalues)) == 1
    @test sum(map(length,lrt.models)) == 4
    @test length(lrt.formulae) == 2
    show(IOBuffer(),lrt);
    @test :pvalues in propertynames(lrt)

    lrt = likelihoodratiotest(lm1,fm1)
    @test lrt.deviance ≈ likelihoodratiotest(lm1.model,fm1).deviance
    @test lrt.dof == [3, 6]
    @test lrt.deviance ≈ -2 * loglikelihood.([lm1, fm1])
    shown = sprint(show, lrt)
    @test occursin("-2 logLik", shown)
    @test !occursin("deviance", shown)

    # non nested FE between non-mixed and mixed
    @test_throws ArgumentError likelihoodratiotest(lm1, fm0)

    # mix of REML and ML
    fm0 = fit(MixedModel,@formula(reaction ~ 1 + (1+days|subj)),slp, REML=true, progress=false);
    @test_throws ArgumentError likelihoodratiotest(fm0,fm1)
    @test_throws ArgumentError likelihoodratiotest(lm0,fm0)

    # differing FE with REML
    fm1 = fit(MixedModel,@formula(reaction ~ 1 + days + (1+days|subj)),slp, REML=true, progress=false);

    @test_throws ArgumentError likelihoodratiotest(fm0,fm1)
    contra = MixedModels.dataset(:contra);
    # glm doesn't like categorical responses, so we convert it to numeric ourselves
    # TODO: upstream fix
    cc = DataFrame(contra);
    cc.usenum = ifelse.(cc.use .== "Y", 1 , 0)
    gmf = glm(@formula(usenum ~ 1+age+urban+livch), cc, Bernoulli());
    gmf2 = glm(@formula(usenum ~ 1+age+abs2(age)+urban+livch), cc, Bernoulli());
    gm0 = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urban&dist)), contra, Bernoulli(), fast=true, progress=false);
    gm1 = fit(MixedModel, @formula(use ~ 1+age+abs2(age)+urban+livch+(1|urban&dist)), contra, Bernoulli(), fast=true, progress=false);

    lrt = likelihoodratiotest(gmf, gm1)
    @test [-2 * loglikelihood(gmf), deviance(gm1)] ≈ lrt.deviance
    @test -2 * loglikelihood(gmf) - deviance(gm1) ≈ only(lrt.tests.deviancediff)
    shown = sprint(show, lrt)
    @test !occursin("-2 logLik", shown)
    @test occursin("deviance", shown)

    lrt = likelihoodratiotest(gm0,gm1);
    @test [deviance(gm0), deviance(gm1)] == lrt.deviance
    @test deviance(gm0) - deviance(gm1) == only(lrt.tests.deviancediff)
    @test first(lrt.tests.dofdiff) == 1
    @test sum(length, lrt.tests) == 3
    @test sum(length, lrt.pvalues) == 1
    @test sum(length, lrt.models) == 4
    @test length(lrt.formulae) == 2

    # mismatched links
    gm_probit = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urban&dist)), contra, Bernoulli(), ProbitLink(), fast=true, progress=false);
    @test_throws ArgumentError likelihoodratiotest(gmf, gm_probit)
    @test_throws ArgumentError likelihoodratiotest(gm0, gm_probit)

    # mismatched families
    gm_poisson = fit(MixedModel, @formula(use ~ 1+age+urban+livch+(1|urban&dist)), contra, Poisson(), fast=true, progress=false);
    @test_throws ArgumentError likelihoodratiotest(gmf, gm_poisson)
    @test_throws ArgumentError likelihoodratiotest(gm0, gm_poisson)

    @test !MixedModels._iscomparable(lm0, gm0)
    @test !MixedModels._iscomparable(gmf, fm1)

    @test MixedModels._iscomparable(gmf, gm0)
    @test !MixedModels._iscomparable(gmf2, gm0)

    @test MixedModels._isnested(gmf.mm.m, gm0.X)
    @test !MixedModels._isnested(gmf2.mm.m, gm0.X)
    # this skips the linear term so that the model matrices
    # have the same column rank
    @test !MixedModels._isnested(gmf2.mm.m[:,Not(2)], gm0.X)
end
