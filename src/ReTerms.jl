#VERSION >= v"0.4.0-dev+6521" && __precompile__()

module ReTerms

using DataArrays, DataFrames, NLopt, StatsBase

export LMM,ReMat,VectorReMat,ColMajorLowerTriangular,DiagonalLowerTriangular

export AIC, BIC, fixef, lowerbd, objective, pwrss

using Base.LinAlg.BlasInt

import Base: ==

include("utils.jl")
include("blockmats.jl")
include("remat.jl")
include("paramlowertriangular.jl")
include("cfactor.jl")
include("inject.jl")
include("pls.jl")

end # module
