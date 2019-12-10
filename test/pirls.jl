using DataFrames, LinearAlgebra, MixedModels, RData, Test
if !@isdefined(dat) || !isa(dat, Dict{Symbol, DataFrame})
    const dat = Dict(Symbol(k) => v for (k, v) in
        load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")))
end

@testset "contra" begin
    contra = dat[:Contraception]
    contra[!, :urbdist] = categorical(string.(contra[!, :d], contra[!, :urb]))
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
    gm1 = fit(MixedModel, contraform, contra, Bernoulli(), nAGQ=7)
    @test isapprox(deviance(gm1), 2360.8760, atol=0.001)
    @test gm1.β == gm1.beta
    @test gm1.θ == gm1.theta
    @test isnan(gm1.σ)
    @test length(gm1.y) == size(gm1.X, 1)
    @test :θ in propertynames(gm0)
    gm0.βθ = vcat(gm0.β, gm0.theta)
    # the next three values are not well defined in the optimization
    #@test isapprox(logdet(gm1), 75.7217, atol=0.1)
    #@test isapprox(sum(abs2, gm1.u[1]), 48.4747, atol=0.1)
    #@test isapprox(sum(gm1.resp.devresid), 2237.349, atol=0.1)
    show(IOBuffer(), gm1)
end

@testset "cbpp" begin
    cbpp = dat[:cbpp]
    cbpp[!, :prop] = cbpp[!, :i] ./ cbpp[!, :s]
    gm2 = fit(MixedModel, @formula(prop ~ 1 + p + (1|h)), cbpp, Binomial(), wts=cbpp[!,:s])
    @test isapprox(deviance(gm2,true), 100.09585619892968, atol=0.0001)
    @test isapprox(sum(abs2, gm2.u[1]), 9.723054788538546, atol=0.0001)
    @test isapprox(logdet(gm2), 16.90105378801136, atol=0.0001)
    @test isapprox(sum(gm2.resp.devresid), 73.47174762237978, atol=0.001)
    @test isapprox(loglikelihood(gm2), -92.02628186840045, atol=0.001)
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
    @test isapprox(sum(x -> sum(abs2, x), gm3.u), 273.29266717430795, rtol=1e-3)
    @test sum(gm3.resp.devresid) ≈ 7156.547357801238 rtol=1e-4
end

@testset "grouseticks" begin
    gm4 = fit(MixedModel, @formula(t ~ 1 + y + ch + (1|i) + (1|b) + (1|l)),
              dat[:grouseticks], Poisson(), fast=true)  # fails in pirls! with fast=false
    @test isapprox(deviance(gm4), 851.4046, atol=0.001)
    # these two values are not well defined at the optimum
    #@test isapprox(sum(x -> sum(abs2, x), gm4.u), 196.8695297987013, atol=0.1)
    #@test isapprox(sum(gm4.resp.devresid), 220.92685781326136, atol=0.1)
end
