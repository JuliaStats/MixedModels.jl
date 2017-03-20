using Base.Test, DataFrames, MixedModels

@testset "contra" begin
    contraception[:use01] = Array(float.(contraception[:use] .== "Y"))
    contraception[:a2] = abs2.(contraception[:a])
    contraception[:urbdist] = string.(contraception[:urb], contraception[:d])
    gm0 = fit!(glmm(@formula(use01 ~ 1 + a + a2 + urb + l + (1 | urbdist)), contraception,
        Bernoulli()), fast=true)
    @test LaplaceDeviance(gm0) ≈ 2361.6572 atol = 0.001
    gm1 = fit!(glmm(@formula(use01 ~ 1 + a + a2 + urb + l + (1 | urbdist)), contraception,
        Bernoulli()));
    @test gm1.θ[1] ≈ 0.573054 atol=0.0001
    @test lowerbd(gm1) == push!(fill(-Inf, 7), 0.)
    @test LaplaceDeviance(gm1) ≈ 2361.5457555 atol=0.0001
    @test loglikelihood(gm1) ≈ -1180.7729 atol=0.001
    @test logdet(gm1) ≈ 75.7275 atol=0.001
    @test sum(abs2, gm1.u[1]) ≈ 48.4747 atol=0.001
    @test sum(gm1.resp.devresid) ≈ 2237.344 atol=0.001
    show(IOBuffer(), gm1)
end

@testset "cbpp" begin
    cbpp[:prop] = cbpp[:i] ./ cbpp[:s]
    gm2 = fit!(glmm(@formula(prop ~ 1 + p + (1 | h)), cbpp, Binomial(), LogitLink(), wt = cbpp[:s]));

    @test LaplaceDeviance(gm2) ≈ 100.095856 atol=0.0001
    @test sum(abs2, gm2.u[1]) ≈ 9.72306601332534 atol=0.0001
    @test logdet(gm2) ≈ 16.901079633923366 atol=0.0001
    @test sum(gm2.resp.devresid) ≈ 73.47171054657996 atol=0.001
    @test loglikelihood(gm2) ≈ -92.02628 atol=0.001
    @test isnan(sdest(gm2))
    @test varest(gm2) == 1
end

@testset "verbagg" begin
    verbagg[:r201] = Array(float.(verbagg[:r2] .== "Y"))
    gm3 = fit!(glmm(@formula(r201 ~ 1 + a + g + b + s + (1 | id) + (1 | item)), verbagg, Bernoulli()));

    @test LaplaceDeviance(gm3) ≈ 8151.399720195352 atol=0.001
    @test lowerbd(gm3) == vcat(fill(-Inf, 6), zeros(2))
    @test sum(x -> sum(abs2, x), gm3.u) ≈ 273.2915019321093 atol=0.0001
    @test sum(gm3.resp.devresid) ≈ 7156.546153979343 atol=0.0001
end
#f = r201 ~ 1 + Anger + Gender + btype + situ + (1 | id)
