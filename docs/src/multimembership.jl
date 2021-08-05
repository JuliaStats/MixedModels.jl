using CairoMakie
using DataFrames
using Distributions
using LinearAlgebra
using StatsBase
using StatsModels
using MixedModels
using Random
using SparseArrays
using Tables

using MixedModels: MultimembershipReMat

nm = 20
nobs0 = 500
const RNG = MersenneTwister(42)
W = rand(RNG, Bernoulli(0.25), nobs0, nm)
b = randn(RNG, nm)
beta = [1, 2]
x = rand(RNG, nobs0)
X = [ones(size(x,1)) x]
y = X * beta + W * b + randn(RNG, nobs0)
df = DataFrame(; x, y, fake=repeat('A':('A'+nm-1); outer=(nobs0 ÷ nm)))


mtemp = LinearMixedModel(@formula(y ~ 1 + x + (1|fake)), df; contrasts=Dict(:fake => Grouping()))
rtemp = mtemp.reterms[1]
z = deepcopy(rtemp.z)
reterm = MultimembershipReMat{Float64,1}(rtemp.trm,
                              deepcopy(rtemp.levels),
                              deepcopy(rtemp.cnames),
                              z,
                              z,
                              deepcopy(rtemp.λ),
                              deepcopy(rtemp.inds),
                              SparseMatrixCSC{Float64, Int32}(W'),
                              deepcopy(rtemp.scratch))
# copyto!(reterm.adjA, SparseMatrixCSC{Float64, Int32}(W'))
model = LinearMixedModel(df.y, mtemp.feterm, AbstractReMat{Float64}[reterm], mtemp.formula)
BlockDescription(mtemp)
BlockDescription(model)

fit!(model)
model.reterms[1].adjA'

mtemp.L[1]
model.L[1]

Matrix(model.Xymat' * model.reterms[1].adjA')

Matrix(mtemp.Xymat' * mtemp.reterms[1].adjA')
