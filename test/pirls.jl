using DataFrames
using Distributions
using MixedModels
using PooledArrays
using StableRNGs
using Tables
using Test

using GLM: Link
using MixedModels: dataset

include("modelcache.jl")

@testset "GLMM from MixedModel" begin
    f = first(gfms[:contra])
    d = dataset(:contra)
    @test MixedModel(f, d, Bernoulli()) isa GeneralizedLinearMixedModel
    @test MixedModel(f, d, Bernoulli(), ProbitLink()) isa GeneralizedLinearMixedModel
end

@testset "Type for instance" begin
    vaform = @formula(r2 ~ 1 + anger + gender + btype + situ + (1|subj) + (1|item))
    verbagg = dataset(:verbagg)
    @test_throws ArgumentError fit(MixedModel, vaform, verbagg, Bernoulli, LogitLink)
    @test_throws ArgumentError fit(MixedModel, vaform, verbagg, Bernoulli(), LogitLink)
    @test_throws ArgumentError fit(MixedModel, vaform, verbagg, Bernoulli, LogitLink())
    @test_throws ArgumentError fit(GeneralizedLinearMixedModel, vaform, verbagg, Bernoulli, LogitLink)
    @test_throws ArgumentError fit(GeneralizedLinearMixedModel, vaform, verbagg, Bernoulli(), LogitLink)
    @test_throws ArgumentError fit(GeneralizedLinearMixedModel, vaform, verbagg, Bernoulli, LogitLink())
    @test_throws ArgumentError GeneralizedLinearMixedModel(vaform, verbagg, Bernoulli, LogitLink)
    @test_throws ArgumentError GeneralizedLinearMixedModel(vaform, verbagg, Bernoulli(), LogitLink)
    @test_throws ArgumentError GeneralizedLinearMixedModel(vaform, verbagg, Bernoulli, LogitLink())
end

@testset "contra" begin
    contra = dataset(:contra)
    thin = 5
    gm0 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); fast=true, progress=false, thin)
    fitlog = gm0.optsum.fitlog
    @test length(fitlog) == (div(gm0.optsum.feval, thin) + 1) # for the initial value
    @test first(fitlog) == (gm0.optsum.initial, gm0.optsum.finitial)
    @test gm0.lowerbd == zeros(1)
    @test isapprox(gm0.θ, [0.5720734451352923], atol=0.001)
    @test !issingular(gm0)
    @test issingular(gm0, [0])
    @test isapprox(deviance(gm0), 2361.657188518064, atol=0.001)
    # the first 9 BLUPs -- I don't think there's much point in testing all 102
    blups = [-0.5853637711570235, -0.9546542393824562, -0.034754249031292345, # values are the same but in different order
              0.2894692928724314, 0.6381376605845264, -0.2513134928312374,
              0.031321447845204374, 0.10836110432794945, 0.24632286640099466]
    @test only(ranef(gm0))[1:9] ≈ blups atol=1e-4
    retbl = raneftables(gm0)
    @test isone(length(retbl))
    @test isa(retbl, NamedTuple)
    @test Tables.istable(only(retbl))
    @test !dispersion_parameter(gm0)
    @test dispersion(gm0, false) == 1
    @test dispersion(gm0, true) == 1
    @test sdest(gm0) === missing
    @test varest(gm0) === missing
    @test gm0.σ === missing
    @test Distribution(gm0) == Distribution(gm0.resp)
    @test Link(gm0) == Link(gm0.resp)

    gm1 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); progress=false);
    @test isapprox(gm1.θ, [0.573054], atol=0.005)
    @test lowerbd(gm1) == vcat(fill(-Inf, 7), 0.)
    @test isapprox(deviance(gm1), 2361.54575, rtol=0.00001)
    @test isapprox(loglikelihood(gm1), -1180.77288, rtol=0.00001)

    @test dof(gm0) == length(gm0.β) + length(gm0.θ)
    @test nobs(gm0) == 1934
    refit!(gm0; fast=true, nAGQ=7, progress=false)
    @test isapprox(deviance(gm0), 2360.9838, atol=0.001)
    gm1 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); nAGQ=7, progress=false)
    @test isapprox(deviance(gm1), 2360.8760, atol=0.001)
    @test gm1.β == gm1.beta
    @test gm1.θ == gm1.theta
    gm1y = gm1.y
    @test length(gm1y) == size(gm1.X, 1)
    @test eltype(gm1y) == eltype(gm1.X)
    @test gm1y == (MixedModels.dataset(:contra).use .== "Y")
    @test response(gm1) == gm1y
    @test !islinear(gm1)
    @test :θ in propertynames(gm0)

    @testset "GLMM rePCA" begin
        @test length(MixedModels.PCA(gm0)) == 1
        @test length(MixedModels.rePCA(gm0)) == 1
        @test length(gm0.rePCA) == 1
    end
    # the next three values are not well defined in the optimization
    #@test isapprox(logdet(gm1), 75.7217, atol=0.1)
    #@test isapprox(sum(abs2, gm1.u[1]), 48.4747, atol=0.1)
    #@test isapprox(sum(gm1.resp.devresid), 2237.349, atol=0.1)
    show(IOBuffer(), gm1)
    show(IOBuffer(), BlockDescription(gm0))

    gm_slope = fit(MixedModel, gfms[:contra][2], contra, Bernoulli(); progress=false);
    @test !issingular(gm_slope)
    @test issingular(gm_slope, zeros(5))

end

@testset "cbpp" begin
    cbpp = dataset(:cbpp)
    gm2 = fit(MixedModel, first(gfms[:cbpp]), cbpp, Binomial(); wts=float(cbpp.hsz), progress=false, init_from_lmm=[:β, :θ])
    @test weights(gm2) == cbpp.hsz
    @test deviance(gm2,true) ≈ 100.09585619892968 atol=0.0001
    @test sum(abs2, gm2.u[1]) ≈ 9.723054788538546 atol=0.0001
    @test logdet(gm2) ≈ 16.90105378801136 atol=0.001
    @test isapprox(sum(gm2.resp.devresid), 73.47174762237978, atol=0.001)
    @test isapprox(loglikelihood(gm2), -92.02628186840045, atol=0.001)
    @test !dispersion_parameter(gm2)
    @test dispersion(gm2, false) == 1
    @test dispersion(gm2, true) == 1
    @test sdest(gm2) === missing
    @test varest(gm2) === missing
    @test gm2.σ === missing

    @testset "GLMM refit" begin
        gm2r = deepcopy(gm2)
        @test_throws ArgumentError fit!(gm2r; progress=false)

        refit!(gm2r; fast=true, progress=false)
        @test length(gm2r.optsum.final) == 1
        @test gm2r.θ ≈ gm2.θ atol=1e-3

        # swapping successes and failures to give us the same model
        # but with opposite signs. healthy ≈ 1 - response(gm2r)
        # but defining it in terms of the original values avoids some
        # nasty floating point issues
        healthy = @. (cbpp.hsz - cbpp.incid) / cbpp.hsz
        refit!(gm2r, healthy; fast=false, progress=false)
        @test length(gm2r.optsum.final) == 5
        @test gm2r.β ≈ -gm2.β atol=1e-3
        @test gm2r.θ ≈ gm2.θ atol=1e-3
    end

    @testset "constant response" begin
        cbconst = DataFrame(cbpp)
        cbconst.incid = zero(cbconst.incid)
        # we do construction and fitting in two separate steps to make sure
        # that construction succeeds and that the ArgumentError occurs in fitting.
        mcbconst = GeneralizedLinearMixedModel(first(gfms[:cbpp]), cbconst, Binomial(); wts=float(cbpp.hsz))
        @test mcbconst isa GeneralizedLinearMixedModel
        @test_throws ArgumentError("The response is constant and thus model fitting has failed") fit!(mcbconst; progress=false)
    end
end

@testset "verbagg" begin
    gm3 = fit(MixedModel, only(gfms[:verbagg]), dataset(:verbagg), Bernoulli(); progress=false)
    @test deviance(gm3) ≈ 8151.40 rtol=1e-5
    @test lowerbd(gm3) == vcat(fill(-Inf, 6), zeros(2))
    @test fitted(gm3) == predict(gm3)
    # these two values are not well defined at the optimum
    @test isapprox(sum(x -> sum(abs2, x), gm3.u), 273.29646346940785, rtol=1e-3)
    @test sum(gm3.resp.devresid) ≈ 7156.550941446312 rtol=1e-4
end

@testset "grouseticks" begin
    center(v::AbstractVector) = v .- (sum(v) / length(v))
    grouseticks = DataFrame(dataset(:grouseticks))
    grouseticks.ch = center(grouseticks.height)
    gm4 = fit(MixedModel, only(gfms[:grouseticks]), grouseticks, Poisson(); fast=true, progress=false)
    @test isapprox(deviance(gm4), 851.4046, atol=0.001)
    # these two values are not well defined at the optimum
    #@test isapprox(sum(x -> sum(abs2, x), gm4.u), 196.8695297987013, atol=0.1)
    #@test isapprox(sum(gm4.resp.devresid), 220.92685781326136, atol=0.1)
    @test !dispersion_parameter(gm4)
    @test dispersion(gm4, false) == 1
    @test dispersion(gm4, true) == 1
    @test sdest(gm4) === missing
    @test varest(gm4) === missing
    @test gm4.σ === missing
    gm4slow = fit(MixedModel, only(gfms[:grouseticks]), grouseticks, Poisson(); fast=false, progress=false)
    # this tolerance isn't great, but then again the optimum isn't well defined
    # @test gm4.θ ≈ gm4slow.θ rtol=0.05
    # @test gm4.β[2:end] ≈ gm4slow.β[2:end] atol=0.1
    @test isapprox(deviance(gm4), deviance(gm4slow); rtol=0.1)
end

@testset "goldstein" begin # from a 2020-04-22 msg by Ben Goldstein to R-SIG-Mixed-Models
    goldstein = (
        group = PooledArray(repeat(string.('A':'J'), outer=10)),
        y = [
            83, 3, 8, 78, 901, 21, 4, 1, 1, 39,
            82, 3, 2, 82, 874, 18, 5, 1, 3, 50,
            87, 7, 3, 67, 914, 18, 0, 1, 1, 38,
            86, 13, 5, 65, 913, 13, 2, 0, 0, 48,
            90, 5, 5, 71, 886, 19, 3, 0, 2, 32,
            96, 1, 1, 87, 860, 21, 3, 0, 1, 54,
            83, 2, 4, 70, 874, 19, 5, 0, 4, 36,
            100, 11, 3, 71, 950, 21, 6, 0, 1, 40,
            89, 5, 5, 73, 859, 29, 3, 0, 2, 38,
            78, 13, 6, 100, 852, 24, 5, 0, 1, 39
            ],
        )
    gform = @formula(y ~ 1 + (1|group))
    m1 = GeneralizedLinearMixedModel(gform, goldstein, Poisson())
    @test !isfitted(m1)
    fit!(m1; progress=false)
    @test isfitted(m1)
    @test deviance(m1) ≈ 193.5587302384811 rtol=1.e-5
    @test only(m1.β) ≈ 4.192196439077657 atol=1.e-5
    @test only(m1.θ) ≈ 1.838245201739852 atol=1.e-5
    m11 = fit(MixedModel, gform, goldstein, Poisson(); nAGQ=11, progress=false)
    @test deviance(m11) ≈ 193.51028088736842 rtol=1.e-5
    @test only(m11.β) ≈ 4.192196439077657 atol=1.e-5
    @test only(m11.θ) ≈ 1.838245201739852 atol=1.e-5
end

@testset "dispersion" begin

    form = @formula(reaction ~ 1 + days + (1+days|subj))
    dat = dataset(:sleepstudy)

    @test_logs (:warn, r"dispersion parameter") GeneralizedLinearMixedModel(form, dat, Gamma())
    @test_logs (:warn, r"dispersion parameter") GeneralizedLinearMixedModel(form, dat, InverseGaussian())
    @test_logs (:warn, r"dispersion parameter") GeneralizedLinearMixedModel(form, dat, Normal(), SqrtLink())

    # notes for future tests when GLMM with dispersion works
    # @test dispersion_parameter(gm)
    # @test dispersion(gm, false) == val
    # @test dispersion(gm, true) == val
    # @test sdest(gm) == dispersion(gm, false) == gm.σ
    # @test varest(gm) == dispersion(gm, true)

end

@testset "mmec" begin
    # Data on "Malignant melanoma in the European community" from the mlmRev package for R
    # The offset of log.(expected) is to examine the ratio of observed to expected, based on population
    mmec = dataset(:mmec)
    mmform = @formula(deaths ~ 1 + uvb + (1|region))
    gm5 = fit(MixedModel, mmform, mmec, Poisson(); offset=log.(mmec.expected), nAGQ=11, progress=false)
    @test isapprox(deviance(gm5), 655.2533533016059, atol=5.e-3)
    @test isapprox(first(gm5.θ), 0.4121684550775567, atol=1.e-3)
    @test isapprox(first(gm5.β), -0.13860166843315044, atol=1.e-3)
    @test isapprox(last(gm5.β), -0.034414458364713504, atol=1.e-3)
end
