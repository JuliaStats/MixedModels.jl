using CategoricalArrays, Compat, DataFrames, MixedModels, RData
using Compat.Test

const dat = convert(Dict{Symbol,Any},load(joinpath(dirname(@__FILE__), "dat.rda")))

include("pls.jl")
include("UniformBlockDiagonal.jl")
include("matrixterm.jl")
include("FactorReTerm.jl")
include("pirls.jl")
