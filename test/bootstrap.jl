using DataFrames
using LinearAlgebra
using MixedModels
using Random
using StableRNGs
using Tables
using Test

using MixedModels: dataset

include("modelcache.jl")

@testset "simulate!(::MixedModel)" begin
    @testset "LMM" begin
        ds = dataset(:dyestuff)
        fm = only(models(:dyestuff))
        # # just in case the fit was modified in a previous test
        # refit!(fm, vec(float.(ds.yield)))
        resp₀ = copy(response(fm))
        # type conversion of ints to floats
        simulate!(StableRNG(1234321), fm, β=[1], σ=1)
        refit!(fm,resp₀)
        refit!(simulate!(StableRNG(1234321), fm))
        @test deviance(fm) ≈ 322.6582 atol=0.001
        refit!(fm, float(ds.yield))
        # Global/implicit RNG method
        Random.seed!(1234321)
        refit!(simulate!(fm))
        # just make sure this worked, don't check fit
        # (because the RNG can change between Julia versions)
        @test response(fm) ≠ resp₀
        simulate!(fm, θ = fm.θ)
        @test_throws DimensionMismatch refit!(fm, zeros(29))
        # restore the original state
        refit!(fm, vec(float.(ds.yield)))
    end

    @testset "Poisson" begin
        center(v::AbstractVector) = v .- (sum(v) / length(v))
        grouseticks = DataFrame(dataset(:grouseticks))
        grouseticks.ch = center(grouseticks.height)
        gm4 = fit(MixedModel, only(gfms[:grouseticks]), grouseticks, Poisson(), fast=true)  # fails in pirls! with fast=false
        gm4sim = refit!(simulate!(StableRNG(42), deepcopy(gm4)))
        @test isapprox(gm4.β, gm4sim.β; atol=norm(stderror(gm4)))
    end

    @testset "Binomial" begin
        cbpp = dataset(:cbpp)
        gm2 = fit(MixedModel, first(gfms[:cbpp]), cbpp, Binomial(), wts=float(cbpp.hsz))
        gm2sim = refit!(simulate!(StableRNG(42), deepcopy(gm2)), fast=true)
        @test isapprox(gm2.β, gm2sim.β; atol=norm(stderror(gm2)))
    end

    @testset "_rand with dispersion" begin
        @test_throws ArgumentError MixedModels._rand(StableRNG(42), Normal(), 1, 1, 1)
        @test_throws ArgumentError MixedModels._rand(StableRNG(42), Gamma(), 1, 1, 1)
        @test_throws ArgumentError MixedModels._rand(StableRNG(42), InverseGaussian(), 1, 1, 1)
    end
end

@testset "bootstrap" begin
    fm = only(models(:dyestuff))
    # two implicit tests
    # 1. type conversion of ints to floats
    # 2. test method for default RNG
    parametricbootstrap(1, fm, β=[1], σ=1)

    bsamp = parametricbootstrap(MersenneTwister(1234321), 100, fm, use_threads=false)
    @test isa(propertynames(bsamp), Vector{Symbol})
    @test length(bsamp.objective) == 100
    @test keys(first(bsamp.bstr)) == (:objective, :σ, :β, :se, :θ)
    @test isa(bsamp.σs, Vector{<:NamedTuple})
    @test length(bsamp.σs) == 100
    allpars = DataFrame(bsamp.allpars)
    @test isa(allpars, DataFrame)
    cov = shortestcovint(shuffle(1.:100.))
    # there is no unique shortest coverage interval here, but the left-most one
    # is currently returned, so we take that. If this behavior changes, then
    # we'll have to change the test
    @test first(cov) == 1.
    @test last(cov) == 95.

    coefp = DataFrame(bsamp.coefpvalues)

    @test isa(coefp, DataFrame)
    @test coefp.iter == 1:100
    @test only(unique(coefp.coefname)) == Symbol("(Intercept)")
    @test propertynames(coefp) == [:iter, :coefname, :β, :se, :z, :p]

    @testset "threaded bootstrap" begin
        bsamp_threaded = parametricbootstrap(MersenneTwister(1234321), 100, fm, use_threads=true)
        # even though it's bad practice with floating point, exact equality should
        # be a valid test here -- if everything is working right, then it's the exact
        # same operations occuring within each bootstrap sample, which IEEE 754
        # guarantees will yield the same result
        @test sort(bsamp_threaded.σ) == sort(bsamp.σ)
        @test sort(bsamp_threaded.θ) == sort(bsamp.θ)
        @test sort(columntable(bsamp_threaded.β).β) == sort(columntable(bsamp.β).β)
        @test sum(issingular(bsamp)) == sum(issingular(bsamp_threaded))
    end


    @testset "Bernoulli simulate! and GLMM boostrap" begin
        contra = dataset(:contra)
        gm0 = fit(MixedModel, only(gfms[:contra]), contra, Bernoulli(), fast=true)
        bs = parametricbootstrap(StableRNG(42), 100, gm0)
        bsci = combine(groupby(DataFrame(bs.β), :coefname),
                       :β => shortestcovint => :ci)
        bsci.lower = first.(bsci.ci)
        bsci.upper = last.(bsci.ci)
        select!(bsci, Not(:ci))
        ciwidth = 2 .* stderror(gm0)
        waldci = DataFrame(coef=fixefnames(gm0),
                           lower=fixef(gm0) .- ciwidth,
                           upper=fixef(gm0) .+ ciwidth)

        # coarse tolerances because we're not doing many bootstrap samples
        @test all(isapprox.(bsci.lower, waldci.lower; atol=0.5))
        @test all(isapprox.(bsci.upper, waldci.upper; atol=0.5))

        σbar = mean(MixedModels.tidyσs(bs)) do x; x.σ end
        @test σbar ≈ 0.56 atol=0.1
        apar = filter!(row -> row.type == "σ", DataFrame(MixedModels.allpars(bs)))
        @test !("Residual" in apar.names)
        @test mean(apar.value) ≈ σbar

        # can't specify dispersion for families without that parameter
        @test_throws ArgumentError parametricbootstrap(StableRNG(42), 100, gm0; σ=2)
        @test sum(issingular(bs)) == 0
    end
end
