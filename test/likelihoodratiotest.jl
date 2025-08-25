using DataFrames
using GLM
using MixedModels
using Suppressor
using Test

using MixedModels: likelihoodratiotest
using MixedModelsDatasets: dataset
using GLM: ProbitLink
using StatsModels: lrtest, isnested

include("modelcache.jl")

@testset "isnested" begin
    slp = dataset(:sleepstudy)

    # these tests don't actually depend on the models being fit,
    # so we just construct them

    # mismatched RE terms
    m1 = LinearMixedModel(@formula(reaction ~ 1 + days + (1 + days | subj)), slp)
    m2 = LinearMixedModel(@formula(reaction ~ 1 + days + (0 + days | subj)), slp)
    @test !isnested(m1, m2)

    # mismatched FE
    m1 = LinearMixedModel(@formula(reaction ~ 1 + days + (1 | subj)), slp)
    m2 = LinearMixedModel(@formula(reaction ~ 0 + days + (1 | subj)), slp)
    @test !isnested(m1, m2)

    # mismatched grouping vars
    kb07 = dataset(:kb07)
    m1 = LinearMixedModel(@formula(rt_trunc ~ 1 + (1 | subj)), kb07)
    m2 = LinearMixedModel(@formula(rt_trunc ~ 1 + (1 | item)), kb07)
    @test !isnested(m1, m2)

    # fixed-effects specification in REML and
    # conversion of internal ArgumentError into @error for StatsModels.isnested
    kb07 = dataset(:kb07)
    m1 = fit(
        MixedModel,
        @formula(rt_trunc ~ 1 + prec + (1 | subj)),
        kb07;
        REML=true,
        progress=false,
    )
    m2 = fit(
        MixedModel,
        @formula(rt_trunc ~ 1 + prec + (1 + prec | subj)),
        kb07;
        REML=true,
        progress=false,
    )
    @test isnested(m1, m2)
    m2 = fit(
        MixedModel,
        @formula(rt_trunc ~ 1 + (1 + prec | subj)),
        kb07;
        REML=true,
        progress=false,
    )
    @test @suppress !isnested(m1, m2)
end

@testset "likelihoodratio test" begin
    slp = dataset(:sleepstudy)

    fm0 = fit(MixedModel, @formula(reaction ~ 1 + (1 + days | subj)), slp; progress=false)
    fm1 = fit(
        MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), slp; progress=false
    )
    lm0 = lm(@formula(reaction ~ 1), slp)
    lm1 = lm(@formula(reaction ~ 1 + days), slp)

    @test MixedModels.isnested(lm0, fm1)
    @test !MixedModels.isnested(lm1, fm0)

    lrt = likelihoodratiotest(fm0, fm1)

    @test (deviance(fm0), deviance(fm1)) == lrt.deviance
    @test sprint(show, lrt) == "Likelihood-ratio test: 2 models fitted on 180 observations\nModel Formulae\n1: reaction ~ 1 + (1 + days | subj)\n2: reaction ~ 1 + days + (1 + days | subj)\n────────────────────────────────────────────\n     DoF  -2 logLik       χ²  χ²-dof  P(>χ²)\n────────────────────────────────────────────\n[1]    5  1775.4759                         \n[2]    6  1751.9393  23.5365       1  <1e-05\n────────────────────────────────────────────"
    @test last(lrt.pvalues) == pvalue(lrt)

    lrt = likelihoodratiotest(lm1, fm1)
    @test pvalue(lrt) ≈ 5.9e-32 atol=1e-16

    lrt = likelihoodratiotest(lm0, fm0, fm1)
    @suppress @test_throws ArgumentError pvalue(lrt)

    # non nested FE between non-mixed and mixed
    @suppress @test_throws ArgumentError likelihoodratiotest(lm1, fm0)

    # mix of REML and ML
    fm0 = fit(
        MixedModel,
        @formula(reaction ~ 1 + (1 + days | subj)),
        slp;
        REML=true,
        progress=false,
    )
    @suppress @test_throws ArgumentError likelihoodratiotest(fm0, fm1)
    @suppress @test_throws ArgumentError likelihoodratiotest(lm0, fm0)

    # differing FE with REML
    fm1 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        slp;
        REML=true,
        progress=false,
    )

    @suppress @test_throws ArgumentError likelihoodratiotest(fm0, fm1)

    contra = MixedModels.dataset(:contra)
    # glm doesn't like categorical responses, so we convert it to numeric ourselves
    # TODO: upstream fix
    cc = DataFrame(dataset(:contra))
    cc.usenum = ifelse.(cc.use .== "Y", 1, 0)
    gmf = glm(@formula(usenum ~ 1 + age + urban + livch), cc, Bernoulli())
    gmf2 = glm(@formula(usenum ~ 1 + age + abs2(age) + urban + livch), cc, Bernoulli())
    gm0 = fit(
        MixedModel,
        @formula(use ~ 1 + age + urban + livch + (1 | urban & dist)),
        dataset(:contra),
        Bernoulli();
        fast=true,
        progress=false,
    )
    gm1 = fit(
        MixedModel,
        @formula(use ~ 1 + age + abs2(age) + urban + livch + (1 | urban & dist)),
        dataset(:contra),
        Bernoulli();
        fast=true,
        progress=false,
    )

    lrt = likelihoodratiotest(gmf, gm1)
    @test  2 * only(diff(collect(lrt.loglikelihood))) ≈ 95.0725 atol=0.0001

    lrt = likelihoodratiotest(gm0, gm1)
    @test  2 * only(diff(collect(lrt.loglikelihood))) ≈ 38.0713 atol=0.0001

    # mismatched links
    gm_probit = fit(
        MixedModel,
        @formula(use ~ 1 + age + urban + livch + (1 | urban & dist)),
        dataset(:contra),
        Bernoulli(),
        ProbitLink();
        fast=true,
        progress=false,
    )
    @suppress @test_throws ArgumentError likelihoodratiotest(gmf, gm_probit)
    @suppress @test_throws ArgumentError likelihoodratiotest(gm0, gm_probit)

    # mismatched families
    gm_poisson = fit(
        MixedModel,
        @formula(use ~ 1 + age + urban + livch + (1 | urban & dist)),
        dataset(:contra),
        Poisson();
        fast=true,
        progress=false,
    )
    @suppress @test_throws ArgumentError likelihoodratiotest(gmf, gm_poisson)
    @suppress @test_throws ArgumentError likelihoodratiotest(gm0, gm_poisson)
    # this skips the linear term so that the model matrices
    # have the same column rank
    @test !MixedModels._isnested(gmf2.mm.m[:, Not(2)], gm0.X)
end
