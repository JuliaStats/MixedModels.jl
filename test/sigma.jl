using MixedModels
using Test
using StableRNGs

@testset "fixed sigma" begin
    σ = 3
    n = 100
    dat = (; x = ones(n),
            z = collect(1:n),
            y = σ*randn(StableRNG(42), n))

    fmσ1 = fit(MixedModel, @formula(y ~ 0 + (1|z)), dat;
               contrasts=Dict(:z => Grouping()),
               σ=1)
    @test isempty(fixef(fmσ1))
    # verify that we report the exact value requested
    @test fmσ1.σ == 1
    # verify that the constrain actually worked
    @test pwrss(fmσ1) / nobs(fmσ1) ≈ 1.0
    @test only(fmσ1.θ) ≈ σ atol=0.1

    fmσ1 = fit(MixedModel, @formula(y ~ 0 + (1|z)), dat;
               contrasts=Dict(:z => Grouping()),
               σ=3.14)
    @test isempty(fixef(fmσ1))
    # verify that we report the exact value requested
    @test fmσ1.σ == 3.14
    # verify that the constrain actually worked
    @test pwrss(fmσ1) / nobs(fmσ1) ≈ 3.14^2 atol=0.5
    # the shrinkage forces things to zero because 3.14/3 is very close to 0
    @test only(fmσ1.θ) ≈ 0 atol=0.1
end

# specifying sigma was done to allow for doing meta-analytic models
# the example from metafor that doesn't work with lme4 and R-based nlme
# can be done here!
# https://www.metafor-project.org/doku.php/tips:rma_vs_lm_lme_lmer
#
# using RCall
# using MixedModels
# R"""
# library(metafor)
# dat <- escalc(measure="ZCOR", ri=ri, ni=ni, data=dat.molloy2014)
# dat$study <- 1:nrow(dat)
# """
# @rget dat

# fit(MixedModel, @formula(yi ~ 1 + (1 | study)), dat;
#     wts=1 ./ dat.vi,
#     REML=true,
#     contrasts=Dict(:study => Grouping()),
#     σ=1)
