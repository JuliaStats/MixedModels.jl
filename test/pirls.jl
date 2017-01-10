using Base.Test, DataFrames, MixedModels

@testset "contra" begin
    contraception[:urban] = DataFrames.pool(convert(Vector{String}, contraception[:urban]))
    contraception[:livch] = DataFrames.pool(convert(Vector{String}, contraception[:livch]))
    gm0 = fit!(glmm(use01 ~ 1 + age + age2 + urban + livch + (1 | urbdist), contraception,
        Bernoulli()), fast=true)
    @test isapprox(LaplaceDeviance(gm0), 2361.6572; atol = 0.001)
    gm1 = fit!(glmm(use01 ~ 1 + age + age2 + urban + livch + (1 | urbdist), contraception,
        Bernoulli()));
    @test lowerbd(gm1) == push!(fill(-Inf, 7), 0.)
    @test isapprox(LaplaceDeviance(gm1), 2361.5457555; atol = 0.0001)
    @test isapprox(loglikelihood(gm1), -1180.7729; atol = 0.001)
# There may be multiple optima here.
#@show logdet(gm1), sumabs2(gm1.u[1]), sum(gm1.devresid)
#@test isapprox(logdet(gm1), 75.717; atol = 0.001)
#@test isapprox(sumabs2(gm1.u[1]), 48.475; atol = 0.001)
#@test isapprox(sum(gm1.devresid), 2237.354; atol = 0.001)
#@test isapprox(fixef(gm1), [-1.0592,0.00328263,-0.00447496,0.770853,0.834578,0.913848,0.927232];
#    atol = 0.001)
    show(IOBuffer(), gm1)
end

@testset "cbpp" begin
    for c in [:herd, :period]
        cbpp[c] = pool(convert(Vector{String}, cbpp[c]))
    end
    cbpp[:prop] = Array(cbpp[:incidence]) ./ Array(cbpp[:size])
    gm2 = fit!(glmm(prop ~ 1 + period + (1 | herd), cbpp, Binomial(), LogitLink(); wt = cbpp[:size]));

    @test isapprox(LaplaceDeviance(gm2), 100.095856; atol = 0.0001)
    @test isapprox(sumabs2(gm2.u[1]), 9.72306601332534; atol = 0.0001)
    @test isapprox(logdet(gm2), 16.901079633923366; atol = 0.0001)
    @test isapprox(sum(gm2.resp.devresid), 73.47171054657996; atol = 0.001)
    @test isapprox(loglikelihood(gm2), -92.02628; atol = 0.001)
    @test isnan(sdest(gm2))
    @test varest(gm2) == 1
end
#VerbAgg = rcopy("lme4::VerbAgg")
#VerbAgg[:r201] = [Int8(x == "N" ? 0 : 1) for x in VerbAgg[:r2]]
#m3 = glmm(r201 ~ 1 + Anger + Gender + btype + situ + (1 | id) + (1 | item), VerbAgg, Bernoulli());
#f = r201 ~ 1 + Anger + Gender + btype + situ + (1 | id)
