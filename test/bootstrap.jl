using DataFrames
using MixedModels
using Random
using Tables
using Test

using MixedModels: dataset

include("modelcache.jl")

@testset "simulate!" begin
    ds = dataset(:dyestuff)
    fm = only(models(:dyestuff))
    resp₀ = copy(response(fm))
    # type conversion of ints to floats
    simulate!(Random.MersenneTwister(1234321), fm, β=[1], σ=1)
    refit!(fm,resp₀)
    refit!(simulate!(Random.MersenneTwister(1234321), fm))
    @test deviance(fm) ≈ 339.0218639362958 atol=0.001
    refit!(fm, float(ds.yield))
    Random.seed!(1234321)
    refit!(simulate!(fm))
    @test deviance(fm) ≈ 339.0218639362958 atol=0.001
    simulate!(fm, θ = fm.θ)
    @test_throws DimensionMismatch refit!(fm, zeros(29))
end

@testset "bootstrap" begin
    fm = only(models(:dyestuff))
    # two implicit tests
    # 1. type conversion of ints to floats
    # 2. test method for default RNG
    parametricbootstrap(1, fm, β=[1], σ=1)

    bsamp = parametricbootstrap(MersenneTwister(1234321), 100, fm, use_threads=false)
    @test isa(propertynames(bsamp), Vector{Symbol})
    @test length(bsamp.objective) == 100
    @test keys(first(bsamp.bstr)) == (:objective, :σ, :β, :se, :θ)
    @test isa(bsamp.σs, Vector{<:NamedTuple})
    @test length(bsamp.σs) == 100
    allpars = DataFrame(bsamp.allpars)
    @test isa(allpars, DataFrame)
    cov = shortestcovint(shuffle(1.:100.))
    # there is no unique shortest coverage interval here, but the left-most one
    # is currently returned, so we take that. If this behavior changes, then
    # we'll have to change the test
    @test first(cov) == 1.
    @test last(cov) == 95.

    coefp = DataFrame(bsamp.coefpvalues)

    @test isa(coefp, DataFrame)
    @test coefp.iter == 1:100
    @test only(unique(coefp.coefname)) == Symbol("(Intercept)")
    @test propertynames(coefp) == [:iter, :coefname, :β, :se, :z, :p]

    @testset "threaded bootstrap" begin
        bsamp_threaded = parametricbootstrap(MersenneTwister(1234321), 100, fm, use_threads=true)
        # even though it's bad practice with floating point, exact equality should
        # be a valid test here -- if everything is working right, then it's the exact
        # same operations occuring within each bootstrap sample, which IEEE 754
        # guarantees will yield the same result
        @test sort(bsamp_threaded.σ) == sort(bsamp.σ)
        @test sort(bsamp_threaded.θ) == sort(bsamp.θ)
        @test sort(columntable(bsamp_threaded.β).β) == sort(columntable(bsamp.β).β)
        @test sum(issingular(bsamp)) == sum(issingular(bsamp_threaded))
    end
end
