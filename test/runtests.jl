using DataArrays, DataFrames, MixedModels, RData, Base.Test

const dat = convert(Dict{Symbol,Any},load(joinpath(dirname(@__FILE__), "dat.rda")))

include("maskedltri.jl")
include("pls.jl")
include("pirls.jl")
