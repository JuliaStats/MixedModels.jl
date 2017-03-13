using Base.Test, DataFrames, MixedModels

@testset "contra" begin
    contraception[:use01] = [x == "N" ? 0 : 1 for x in contraception[:use]]
    contraception[:a2] = abs2.(contraception[:a])
    contraception[:urbdist] = pool(string.(Array(contraception[:urb]), Array(contraception[:d])))
    gm0 = fit!(glmm(@formula(use01 ~ 1 + a + a2 + urb + l + (1 | urbdist)), contraception,
        Bernoulli()), fast=true)
    @test isapprox(LaplaceDeviance(gm0), 2361.6572; atol = 0.001)
    gm1 = fit!(glmm(@formula(use01 ~ 1 + a + a2 + urb + l + (1 | urbdist)), contraception,
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
    cbpp[:prop] = cbpp[:i] ./ cbpp[:s]
    gm2 = fit!(glmm(@formula(prop ~ 1 + p + (1 | h)), cbpp, Binomial(), LogitLink(); wt = cbpp[:s]));

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
