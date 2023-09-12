using Aqua
using GLM
using MixedModels
using Test

import InteractiveUtils: versioninfo
import LinearAlgebra: BLAS

# there seem to be processor-specific issues and knowing this is helpful
@info sprint(versioninfo)
@info BLAS.get_config()

@testset "Aqua" begin
    # we can't check for unbound type parameters
    # because we actually need one at one point for _same_family()
    Aqua.test_all(MixedModels; ambiguities=false, unbound_args=false,
                  # XXX TODO: upstream this piracy
                  piracy=(;treat_as_own=[GLM.wrkresp!, Base.:|]))
end

include("utilities.jl")
include("misc.jl")
include("pivot.jl")
include("UniformBlockDiagonal.jl")
include("linalg.jl")
include("matrixterm.jl")
include("FactorReTerm.jl")
include("grouping.jl")
include("pls.jl")
include("pirls.jl")
include("gausshermite.jl")
include("fit.jl")
include("missing.jl")
include("likelihoodratiotest.jl")
include("bootstrap.jl")
include("mime.jl")
include("optsummary.jl")
include("predict.jl")
include("sigma.jl")
