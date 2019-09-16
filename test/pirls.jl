using DataFrames, DataFramesMeta, LinearAlgebra, MixedModels, RData, Test

if !@isdefined(dat) || !isa(dat, Dict{Symbol, DataFrame})
    const dat = Dict(Symbol(k) => v for (k, v) in
        load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")))
end

@testset "contra" begin
    contra = @transform(dat[:Contraception], urbdist=categorical(string.(:d, :urb)))
    contraform = @formula(use ~ 1+a+abs2(a)+urb+l+(1|urbdist))
    gm0 = fit(MixedModel, contraform, contra, Bernoulli(), fast=true);
    @test gm0.lowerbd == zeros(1)
    @test isapprox(gm0.θ, [0.5720734451352923], atol=0.001)
    @test isapprox(deviance(gm0,true), 2361.657188518064, atol=0.001)
    gm1 = fit(MixedModel, contraform, contra, Bernoulli());
    @test isapprox(gm1.θ, [0.573054], atol=0.005)
    @test lowerbd(gm1) == vcat(fill(-Inf, 7), 0.)
    @test isapprox(deviance(gm1,true), 2361.54575, rtol=0.00001)
    @test isapprox(loglikelihood(gm1), -1180.77288, rtol=0.00001)
    @test dof(gm0) == length(gm0.β) + length(gm0.θ)
    @test nobs(gm0) == 1934
    fit!(gm0, fast=true, nAGQ=7)
    @test isapprox(deviance(gm0), 2360.9838, atol=0.001)
    fit!(gm0, nAGQ=7)
    @test isapprox(deviance(gm0), 2360.8760, atol=0.001)
    @test gm0.β == gm0.beta
    @test gm0.θ == gm0.theta
    @test isnan(gm0.σ)
    @test length(gm0.y) == size(gm0.X, 1)
    @test :θ in propertynames(gm0)
    gm0.β = gm0.beta
    @test gm0.β == gm0.beta
    gm0.θ = gm0.theta
    @test gm0.θ == gm0.theta
    gm0.βθ = vcat(gm0.β, gm0.theta)
    @test gm0.β == gm0.beta
    @test gm0.θ == gm0.theta
    # the next three values are not well defined in the optimization
    #@test isapprox(logdet(gm1), 75.7217, atol=0.1)
    #@test isapprox(sum(abs2, gm1.u[1]), 48.4747, atol=0.1)
    #@test isapprox(sum(gm1.resp.devresid), 2237.349, atol=0.1)
    show(IOBuffer(), gm1)
end

@testset "cbpp" begin
    cbpp = @transform(dat[:cbpp], prop = :i ./ :s)
    gm2 = fit(MixedModel, @formula(prop ~ 1 + p + (1|h)), cbpp, Binomial(), wts = cbpp[!,:s])
    @test isapprox(deviance(gm2,true), 100.09585619324639, atol=0.0001)
    @test isapprox(sum(abs2, gm2.u[1]), 9.723175126731014, atol=0.0001)
    @test isapprox(logdet(gm2), 16.90099, atol=0.0001)
    @test isapprox(sum(gm2.resp.devresid), 73.47179193718736, atol=0.001)
    @test isapprox(loglikelihood(gm2), -92.02628186555876, atol=0.001)
    @test isnan(sdest(gm2))
    @test varest(gm2) == 1
end

@testset "verbagg" begin
    gm3 = fit(MixedModel, @formula(r2 ~ 1 + a + g + b + s + (1|id)+(1|item)), dat[:VerbAgg],
         Bernoulli())
    @test deviance(gm3) ≈ 8151.40 rtol=1e-5
    @test lowerbd(gm3) == vcat(fill(-Inf, 6), zeros(2))
    @test fitted(gm3) == predict(gm3)
    # these two values are not well defined at the optimum
    @test sum(x -> sum(abs2, x), gm3.u) ≈ 273.31563469936697 rtol=1e-3
    @test sum(gm3.resp.devresid) ≈ 7156.558983084621 rtol=1e-4
end

@testset "grouseticks" begin
    gm4 = fit(MixedModel, @formula(t ~ 1 + y + ch + (1|i) + (1|b) + (1|l)),
              dat[:grouseticks], Poisson(), fast=true)  # fails in pirls! with fast=false
    @test isapprox(deviance(gm4), 851.4046, atol=0.001)
    @test lowerbd(gm4) == vcat(zeros(3))
    # these two values are not well defined at the optimum
    #@test isapprox(sum(x -> sum(abs2, x), gm4.u), 196.8695297987013, atol=0.1)
    #@test isapprox(sum(gm4.resp.devresid), 220.92685781326136, atol=0.1)
end
