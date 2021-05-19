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

end

@testset "predict" begin
   @testset "LMM" for m in models(:sleepstudy)[[begin,end]]
        # these currently use approximate equality
        # because of floating point, but realistically
        # this should be exactly equal in most cases
        @test predict(m) ≈ fitted(m)
        @test predict(m; use_re=false) ≈ m.X * m.β
        @test predict(m, m.X) ≈ fitted(m)

        slp = DataFrame(dataset(:sleepstudy))

        @test predict(m, slp; new_re_levels=:error) ≈ fitted(m)
        @test predict(m, slp; new_re_levels=:population) ≈ fitted(m)
        @test predict(m, slp; new_re_levels=:missing) ≈ fitted(m)
    end

    @testset "GLMM" begin
        contra = dataset(:contra)
        gm0 = fit(MixedModel, only(gfms[:contra]), contra, Bernoulli(), fast=true)

        @test_throws ArgumentError predict(gm0; type=:doh)

        # we can skip a lot of testing if the broad strokes work because
        # internally this is punted off to the LMM machinery
        @test predict(gm0) ≈ fitted(gm0)
        @test predict(gm0; type=:linpred) ≈ gm0.resp.eta
        gm0pop = gm0.X * gm0.β
        @test predict(gm0, gm0.X; use_re=false, type=:response) ≈ linkinv.(Link(gm0.resp),gm0pop)
        @test predict(gm0, gm0.X; use_re=false, type=:linpred) ≈ gm0pop
    end
end
