using DataFrames, LinearAlgebra, MixedModels, RData, StatsBase, Test

if !@isdefined(dat) || !isa(dat, Dict{Symbol, DataFrame})
    dat = Dict(Symbol(k) => v for (k, v) in load(joinpath(dirname(@__FILE__), "dat.rda")))
end

@testset "contra" begin
    contraception = dat[:Contraception]
    contraception[:a2] = abs2.(contraception[:a])
    contraception[:urbdist] = string.(contraception[:urb], contraception[:d])
    gm0 = fit!(GeneralizedLinearMixedModel(@formula(use ~ 1 + a + a2 + urb + l + (1 | urbdist)), contraception,
                    Bernoulli()), fast = true);
    @test gm0.lowerbd == [0.]
    @test isapprox(getθ(gm0)[1], 0.5720734451352923, atol=0.001)
    @test isapprox(deviance(gm0,true), 2361.657188518064, atol=0.001)
    gm1 = fit(GeneralizedLinearMixedModel, @formula(use ~ 1 + a + a2 + urb + l + (1 | urbdist)), contraception,
        Bernoulli());
    @test isapprox(gm1.θ[1], 0.573054, atol=0.005)
    @test lowerbd(gm1) == push!(fill(-Inf, 7), 0.)
    @test isapprox(deviance(gm1,true), 2361.57129, rtol=0.00001)
    @test isapprox(loglikelihood(gm1), -1180.78565, rtol=0.00001)
    @test StatsBase.dof(gm0) == length(gm0.β) + length(gm0.θ)
    @test StatsBase.nobs(gm0) == 1934
    fit!(gm0, fast=true, nAGQ=7)
    @test isapprox(deviance(gm0, 7), 2360.9838, atol=0.001)
    fit!(gm0, nAGQ=7)
    @test isapprox(deviance(gm0, 7), 2360.9002, atol=0.001)
    @test gm0.β == gm0.beta
    @test gm0.θ == gm0.theta
    @test isnan(gm0.σ)
    @test gm0.formula == @formula(use ~ 1 + a + a2 + urb + l + (1 | urbdist))
    @test length(gm0.y) == size(gm0.X, 1)
    @test :θ in propertynames(gm0)
    # the next three values are not well defined in the optimization
    #@test isapprox(logdet(gm1), 75.7275, atol=0.1)
    #@test isapprox(sum(abs2, gm1.u[1]), 48.4747, atol=0.1)
    #@test isapprox(sum(gm1.resp.devresid), 2237.344, atol=0.1)
    show(IOBuffer(), gm1)
end

@testset "cbpp" begin
    cbpp = dat[:cbpp]
    cbpp[:prop] = cbpp[:i] ./ cbpp[:s]
    gm2 = fit!(GeneralizedLinearMixedModel(@formula(prop ~ 1 + p + (1 | h)), cbpp, Binomial(),
                    wt = cbpp[:s]));
    @test isapprox(deviance(gm2,true), 100.09585619324639, atol=0.0001)
    @test isapprox(sum(abs2, gm2.u[1]), 9.723175126731014, atol=0.0001)
    @test isapprox(logdet(gm2), 16.90113, atol=0.0001)
    @test isapprox(sum(gm2.resp.devresid), 73.47179193718736, atol=0.001)
    @test isapprox(loglikelihood(gm2), -92.02628186555876, atol=0.001)
    @test isnan(sdest(gm2))
    @test varest(gm2) == 1
end

@testset "verbagg" begin
    verbagg = dat[:VerbAgg]
    gm3 = fit(GeneralizedLinearMixedModel, @formula(r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)),
         verbagg, Bernoulli());
    @test isapprox(deviance(gm3), 8151.40, atol=0.01)
    @test lowerbd(gm3) == vcat(fill(-Inf, 6), zeros(2))
    @test fitted(gm3) == predict(gm3)
    # these two values are not well defined at the optimum
    @test isapprox(sum(x -> sum(abs2, x), gm3.u), 273.31563469936697, atol=0.1)
    @test isapprox(sum(gm3.resp.devresid), 7156.558983084621, atol=0.1)
end

@testset "grouseticks" begin
    gm4 = fit!(GeneralizedLinearMixedModel(@formula(t ~ 1 + y + ch + (1|i) + (1|b) + (1|l)),
              dat[:grouseticks], Poisson()), fast=true)  # fails in pirls! with fast=false
    @test isapprox(deviance(gm4), 851.4046, atol=0.001)
    @test lowerbd(gm4) == vcat(fill(-Inf, 4), zeros(3))
    # these two values are not well defined at the optimum
    #@test isapprox(sum(x -> sum(abs2, x), gm4.u), 196.8695297987013, atol=0.1)
    #@test isapprox(sum(gm4.resp.devresid), 220.92685781326136, atol=0.1)
end
