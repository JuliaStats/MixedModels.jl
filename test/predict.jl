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
    end

    @testset "GLMM" begin
        contra = DataFrame(dataset(:contra))
        m = fit(MixedModel, only(gfms[:contra]), contra, Bernoulli(), fast=true;
                contrasts=Dict(:urban => EffectsCoding()))
        mc = deepcopy(m)
        fit!(simulate!(StableRNG(42), mc))
        @test simulate(StableRNG(42), m) ≈ mc.y
        y = similar(mc.y)
        @test simulate!(StableRNG(42), y, m) ≈ mc.y
        @test y ≈ mc.y
    end
end

@testset "predict" begin
    slp = DataFrame(dataset(:sleepstudy))
    slp2 = transform(slp, :subj => ByRow(x -> (x == "S308" ? "NEW" : x)) => :subj)
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
    end

    @testset "GLMM" begin
        contra = dataset(:contra)
        gm0 = fit(MixedModel, only(gfms[:contra]), contra, Bernoulli(), fast=true)

        @test_throws ArgumentError predict(gm0, contra; type=:doh)

        # we can skip a lot of testing if the broad strokes work because
        # internally this is punted off to the same machinery as LMM
        @test predict(gm0) ≈ fitted(gm0)
        @test predict(gm0, contra; type=:linpred) ≈ gm0.resp.eta
    end
end
