using DataFrames
using LinearAlgebra
using MixedModels
using Random
using StableRNGs
using Tables
using Test

using GLM: Link, linkfun, linkinv
using MixedModels: dataset

include("modelcache.jl")

@testset "simulate[!](::AbstractVector)" begin
    @testset "LMM" begin
        slp = DataFrame(dataset(:sleepstudy))
        m = first(models(:sleepstudy))
        mc = deepcopy(m)
        fit!(simulate!(StableRNG(42), mc); progress=false)
        @test simulate(StableRNG(42), m) ≈ mc.y
        y = similar(mc.y)
        @test simulate!(StableRNG(42), y, m) ≈ mc.y
        @test y ≈ mc.y

        @test simulate(StableRNG(42), m, slp) ≈ y
        slptop = first(slp, 90)
        @test simulate(StableRNG(42), m, slptop) ≈ simulate(StableRNG(42), m, slptop; β=m.β, θ=m.θ, σ=m.σ)

        # test of methods using default RNG
        rng = deepcopy(Random.GLOBAL_RNG)
        @test length(simulate(m, slptop)) == nrow(slptop)
        @test length(simulate!(y, m, slptop)) == nrow(slptop)
    end

    @testset "GLMM" begin
        contra = DataFrame(dataset(:contra))
        m = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); fast=true,
                contrasts=Dict(:urban => EffectsCoding()), progress=false)
        mc = deepcopy(m)
        fit!(simulate!(StableRNG(42), mc); progress=false)
        @test simulate(StableRNG(42), m) ≈ mc.y
        y = similar(mc.y)
        @test simulate!(StableRNG(42), y, m) ≈ mc.y
        @test y ≈ mc.y
        @test length(simulate!(StableRNG(42), y, m, contra)) == length(mc.y)
    end
end

@testset "predict" begin
    slp = DataFrame(dataset(:sleepstudy))
    slp2 = transform(slp, :subj => ByRow(x -> (x == "S308" ? "NEW" : x)) => :subj)
    slpm = allowmissing(slp, :reaction)
    @testset "LMM" for m in models(:sleepstudy)[[begin,end]]
        # these currently use approximate equality
        # because of floating point, but realistically
        # this should be exactly equal in most cases
        @test predict(m) ≈ fitted(m)

        @test predict(m, slp; new_re_levels=:error) ≈ fitted(m)
        @test predict(m, slp; new_re_levels=:population) ≈ fitted(m)
        @test predict(m, slp; new_re_levels=:missing) ≈ fitted(m)

        @test_throws ArgumentError predict(m, slp2; new_re_levels=:error)
        ymissing = predict(m, slp2; new_re_levels=:missing)
        @test count(ismissing, ymissing) == 10
        @test ymissing[11:end] ≈ fitted(m)[11:end]
        ypop = predict(m, slp2; new_re_levels=:population)
        @test ypop[1:10] ≈ view(m.X, 1:10, :) * m.β
        @test ypop[11:end] ≈ fitted(m)[11:end]

        @test_throws ArgumentError predict(m, slp[:, Not(:reaction)])
        copyto!(slpm.reaction, slp.reaction)
        slpm[1, :reaction] = missing
        @test_throws ArgumentError predict(m, slpm)
        fill!(slpm.reaction, missing)
        @test_throws ArgumentError predict(m, slpm)
    end

    @testset "transformed response" begin
        slp1 = subset(slp, :days => ByRow(>(0)))
        # this model probably doesn't make much sense, but it has two
        # variables on the left hand side in a FunctionTerm
        m = fit(MixedModel, @formula(reaction / days ~ 1 + (1|subj)), slp1)
        @test response(m) ≈ slp1.reaction ./ slp1.days
        @test_throws ArgumentError predict(m, slp[:, Not(:reaction)])
        # these currently use approximate equality
        # because of floating point, but realistically
        # this should be exactly equal in most cases
        @test predict(m) ≈ fitted(m)

        @test predict(m, slp1; new_re_levels=:error) ≈ fitted(m)
        @test predict(m, slp1; new_re_levels=:population) ≈ fitted(m)
        @test predict(m, slp1; new_re_levels=:missing) ≈ fitted(m)
    end

    @testset "GLMM" begin
        contra = dataset(:contra)
        for fast in [true, false]
            gm0 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); fast, progress=false)

            @test_throws ArgumentError predict(gm0, contra; type=:doh)

            # we can skip a lot of testing if the broad strokes work because
            # internally this is punted off to the same machinery as LMM
            @test predict(gm0) ≈ fitted(gm0)
            # XXX these tolerances aren't great but are required for fast=false fits
            @test predict(gm0, contra; type=:linpred) ≈ gm0.resp.eta rtol=0.1
            @test predict(gm0, contra; type=:response) ≈ gm0.resp.mu rtol=0.01
        end
    end
end
