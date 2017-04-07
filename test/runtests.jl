using DataArrays, DataFrames, MixedModels, RData, Base.Test

const dat = load(joinpath(@__DIR__, "dat.rda"))

include("maskedltri.jl")
#include("scalarReTerm.jl")
#include("vectorReTerm.jl")
include("pls.jl")
include("pirls.jl")
#include("throws.jl")
