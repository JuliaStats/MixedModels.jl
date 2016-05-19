using Compat, DataArrays, DataFrames, MixedModels, Base.Test

include(joinpath(dirname(@__FILE__),"data.jl"))

include("paramlowertriangular.jl")
include("scalarReTerm.jl")
include("vectorReTerm.jl")
include("pls.jl")
include("pirls.jl")
include("throws.jl")
include("inject.jl")
include("cfactor.jl")
