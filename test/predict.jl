using DataFrames
using LinearAlgebra
using MixedModels
using Random
using StableRNGs
using Tables
using Test

using MixedModels: dataset

include("modelcache.jl")

@testset "simulate[!](::AbstractVector)" begin

end

@testset "predict" begin
    m = last(models(:sleepstudy))
    @test predict(m) == fitted(m)
    @test predict(m; use_re=false) == m.X * m.β
end