using CategoricalArrays, DataFrames, LinearAlgebra, MixedModels, RData, Test

const dat = Dict(Symbol(k) => v for (k,v) in load(joinpath(dirname(@__FILE__), "dat.rda")))

include("UniformBlockDiagonal.jl")
include("linalg.jl")
include("matrixterm.jl")
include("FactorReTerm.jl")
include("pls.jl")
include("pirls.jl")
include("gausshermite.jl")
