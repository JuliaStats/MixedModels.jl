using RData

const dat = Dict(Symbol(k) => v for (k,v) in load(joinpath(dirname(@__FILE__), "dat.rda")))

include("rcall.jl") # move this to last before merging
include("statschol.jl")
include("UniformBlockDiagonal.jl")
include("linalg.jl")
include("matrixterm.jl")
include("FactorReTerm.jl")
include("pls.jl")
include("pirls.jl")
include("gausshermite.jl")
include("fit.jl")
include("missing.jl")

using MixedModels
