using MixedModels
using Test

using MixedModels: dataset

const gfms = Dict(
    :cbpp => [@formula((incid/hsz) ~ 1 + period + (1|herd))],
    :contra => [@formula(use ~ 1+age+abs2(age)+urban+livch+(1|urbdist))],
    :grouseticks => [@formula(ticks ~ 1+year+ch+ (1|index) + (1|brood) + (1|location))],
    :verbagg => [@formula(r2 ~ 1+anger+gender+btype+situ+(1|subj)+(1|item))],
)

@testset "contra" begin
    contra = dataset(:contra)
    gm0 = fit(MixedModel, only(gfms[:contra]), contra, Bernoulli(), fast=true);
    @test gm0.lowerbd == zeros(1)
    @test isapprox(gm0.θ, [0.5720734451352923], atol=0.001)
    @test isapprox(deviance(gm0), 2361.657188518064, atol=0.001)
    gm1 = fit(MixedModel, only(gfms[:contra]), contra, Bernoulli());
    @test isapprox(gm1.θ, [0.573054], atol=0.005)
    @test lowerbd(gm1) == vcat(fill(-Inf, 7), 0.)
    @test isapprox(deviance(gm1), 2361.54575, rtol=0.00001)
    @test isapprox(loglikelihood(gm1), -1180.77288, rtol=0.00001)
    @test dof(gm0) == length(gm0.β) + length(gm0.θ)
    @test nobs(gm0) == 1934
    fit!(gm0, fast=true, nAGQ=7)
    @test isapprox(deviance(gm0), 2360.9838, atol=0.001)
    gm1 = fit(MixedModel, only(gfms[:contra]), contra, Bernoulli(), nAGQ=7)
    @test isapprox(deviance(gm1), 2360.8760, atol=0.001)
    @test gm1.β == gm1.beta
    @test gm1.θ == gm1.theta
    @test isnan(gm1.σ)
    @test length(gm1.y) == size(gm1.X, 1)
    @test :θ in propertynames(gm0)
    @testset "GLMM rePCA" begin
        @test length(MixedModels.PCA(gm0)) == 1
        @test length(MixedModels.rePCA(gm0)) == 1
        @test length(gm0.rePCA) == 1
    end
    # the next three values are not well defined in the optimization
    #@test isapprox(logdet(gm1), 75.7217, atol=0.1)
    #@test isapprox(sum(abs2, gm1.u[1]), 48.4747, atol=0.1)
    #@test isapprox(sum(gm1.resp.devresid), 2237.349, atol=0.1)
    show(IOBuffer(), gm1)
    show(IOBuffer(), BlockDescription(gm0))
end

@testset "cbpp" begin
    cbpp = dataset(:cbpp)
    gm2 = fit(MixedModel, only(gfms[:cbpp]), cbpp, Binomial(), wts=float(cbpp.hsz))
    @test deviance(gm2,true) ≈ 100.09585619892968 atol=0.0001
    @test sum(abs2, gm2.u[1]) ≈ 9.723054788538546 atol=0.0001
    @test logdet(gm2) ≈ 16.90105378801136 atol=0.0001
    @test isapprox(sum(gm2.resp.devresid), 73.47174762237978, atol=0.001)
    @test isapprox(loglikelihood(gm2), -92.02628186840045, atol=0.001)
    @test isnan(sdest(gm2))
    @test varest(gm2) == 1
end

@testset "verbagg" begin
    gm3 = fit(MixedModel, only(gfms[:verbagg]), dataset(:verbagg), Bernoulli())
    @test deviance(gm3) ≈ 8151.40 rtol=1e-5
    @test lowerbd(gm3) == vcat(fill(-Inf, 6), zeros(2))
    @test fitted(gm3) == predict(gm3)
    # these two values are not well defined at the optimum
    @test isapprox(sum(x -> sum(abs2, x), gm3.u), 273.29646346940785, rtol=1e-3)
    @test sum(gm3.resp.devresid) ≈ 7156.550941446312 rtol=1e-4
end

@testset "grouseticks" begin
    center(v::AbstractVector) = v .- (sum(v) / length(v))
    grouseticks = dataset(:grouseticks)
    grouseticks.ch = center(grouseticks.height)
    gm4 = fit(MixedModel, only(gfms[:grouseticks]), grouseticks, Poisson(), fast=true)  # fails in pirls! with fast=false
    @test isapprox(deviance(gm4), 851.4046, atol=0.001)
    # these two values are not well defined at the optimum
    #@test isapprox(sum(x -> sum(abs2, x), gm4.u), 196.8695297987013, atol=0.1)
    #@test isapprox(sum(gm4.resp.devresid), 220.92685781326136, atol=0.1)
end

@testset "goldstein" begin # from a 2020-04-22 msg by Ben Goldstein to R-SIG-Mixed-Models
    goldstein = 
        categorical!(
            DataFrame(
                group = repeat(1:10, outer=10), 
                y = [
                    83, 3, 8, 78, 901, 21, 4, 1, 1, 39,
                    82, 3, 2, 82, 874, 18, 5, 1, 3, 50,
                    87, 7, 3, 67, 914, 18, 0, 1, 1, 38,
                    86, 13, 5, 65, 913, 13, 2, 0, 0, 48,
                    90, 5, 5, 71, 886, 19, 3, 0, 2, 32,
                    96, 1, 1, 87, 860, 21, 3, 0, 1, 54,
                    83, 2, 4, 70, 874, 19, 5, 0, 4, 36,
                    100, 11, 3, 71, 950, 21, 6, 0, 1, 40,
                    89, 5, 5, 73, 859, 29, 3, 0, 2, 38,
                    78, 13, 6, 100, 852, 24, 5, 0, 1, 39
                    ],
                ),
            :group,
        )
    gform = @formula(y ~ 1 + (1|group))
    m1 = fit(MixedModel, gform, goldstein, Poisson())
    @test deviance(m1) ≈ 193.5587302384811 rtol=1.e-5
    @test only(m1.β) ≈ 4.192196439077657 atol=1.e-5
    @test only(m1.θ) ≈ 1.838245201739852 atol=1.e-5
    m11 = fit(MixedModel, gform, goldstein, Poisson(), nAGQ=11)
    @test deviance(m11) ≈ 193.51028088736842 rtol=1.e-5
    @test only(m11.β) ≈ 4.192196439077657 atol=1.e-5
    @test only(m11.θ) ≈ 1.838245201739852 atol=1.e-5
end