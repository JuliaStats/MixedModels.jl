using DataArrays, DataFrames, MixedModels, RData, Base.Test

const dat = convert(Dict{Symbol,Any},load(joinpath(dirname(@__FILE__), "dat.rda")))

include("UniformBlockDiagonal.jl")
include("matrixterm.jl")
include("FactorReTerm.jl")
include("pls.jl")
include("pirls.jl")
