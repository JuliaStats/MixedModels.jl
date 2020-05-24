using MixedModels
import InteractiveUtils: versioninfo

# there seem to be processor-specific issues and knowing this is helpful 
println(versioninfo())

include("utilities.jl")
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
include("likelihoodratiotest.jl")
