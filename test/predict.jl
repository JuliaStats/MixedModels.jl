using DataFrames
using LinearAlgebra
using MixedModels
using Random
using Suppressor
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

    @testset "single obs" begin
        kb07 = DataFrame(dataset(:kb07))
        m = models(:kb07)[1]
        only(predict(m, kb07[1:1, :])) ≈ first(fitted(m))
    end

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

    @testset "rank deficiency" begin
        @testset "in original fit" begin
            refvals = predict(first(models(:sleepstudy)), slp)

            slprd = transform(slp, :days => ByRow(x -> 2x) => :days2)
            m = @suppress fit(MixedModel, @formula(reaction ~ 1 + days + days2 + (1|subj)), slprd; progress=false)
            # predict assumes that X is the correct length and stored pivoted
            # so these first two tests will fail if we change that storage detail
            @test size(m.X) == (180, 3)
            @test all(2 .* view(m.X, :, m.feterm.piv[2]) .== view(m.X, :, m.feterm.piv[3]))
            @test @suppress predict(m, slprd) == refvals

            slprd0 = transform(slp, :days => zero => :days0)
            m = @suppress fit(MixedModel, @formula(reaction ~ 1 + days0 + days + (1|subj)), slprd0; progress=false)
            @test @suppress predict(m, slprd0) == refvals
            # change the formula order slightly so that the original ordering and hence the
            # permutation vector for pivoting isn't identical
            m = @suppress fit(MixedModel, @formula(reaction ~ 1 + days + days0 + (1|subj)), slprd0; progress=false)
            @test @suppress predict(m, slprd0) == refvals
        end

        @testset "in newdata" begin
            mref = first(models(:sleepstudy))
            # remove days
            refvals = fitted(mref) .- view(mref.X, :, 2) * mref.β[2]
            slp0 = transform(slp, :days => zero => :days)
            vals = @suppress predict(mref, slp0)
            @test all(refvals .≈ vals)
        end

        @testset "in both" begin
            # now what happens when old and new are rank deficient
            mref = first(models(:sleepstudy))
            # remove days
            refvals = fitted(mref) .- view(mref.X, :, 2) * mref.β[2]
            # days gets pivoted out
            slprd = transform(slp, :days => ByRow(x -> 2x) => :days2)
            m = @suppress fit(MixedModel, @formula(reaction ~ 1 + days + days2 + (1|subj)), slprd; progress=false)
            # days2 gets pivoted out
            slp0 = transform(slp, :days => zero => :days2)
            vals = @suppress predict(m, slp0)
            # but in the original fit, days gets pivoted out, so its coef is zero
            # and now we have a column of zeros for days2
            # leaving us with only the intercept
            # this is consistent behavior
            @test all(refvals .≈ vals)

            slp1 = transform(slp, :days => ByRow(one) => :days2)
            vals = @suppress predict(m, slp1)
            refvals = fitted(mref) .- view(mref.X, :, 2) * mref.β[2] .+ last(fixef(m))
            @test all(refvals .≈ vals)
        end
    end

    @testset "transformed response" begin
        slp1 = subset(slp, :days => ByRow(>(0)))
        # this model probably doesn't make much sense, but it has two
        # variables on the left hand side in a FunctionTerm
        m = @suppress fit(MixedModel, @formula(reaction / days ~ 1 + (1|subj)), slp1)
        # make sure that we're getting the transformation
        @test response(m) ≈ slp1.reaction ./ slp1.days
        @test_throws ArgumentError predict(m, slp[:, Not(:reaction)])
        # these currently use approximate equality
        # because of floating point, but realistically
        # this should be exactly equal in most cases
        @test predict(m) ≈ fitted(m)
        @test predict(m, slp1) ≈ fitted(m)


        m = @suppress fit(MixedModel, @formula(log10(reaction) ~ 1 + days + (1|subj)), slp1)
        # make sure that we're getting the transformation
        @test response(m) ≈ log10.(slp1.reaction)
        @test_throws ArgumentError predict(m, slp[:, Not(:reaction)])
        # these currently use approximate equality
        # because of floating point, but realistically
        # this should be exactly equal in most cases
        @test predict(m) ≈ fitted(m)
        @test predict(m, slp1) ≈ fitted(m)
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
