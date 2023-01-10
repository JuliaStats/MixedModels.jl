using Aqua
using MixedModels
using Test

import InteractiveUtils: versioninfo
import LinearAlgebra: BLAS

# there seem to be processor-specific issues and knowing this is helpful
println(versioninfo())
@static if VERSION ≥ v"1.7.0-DEV.620"
    println(BLAS.get_config())
else
    @show BLAS.vendor()
    if startswith(string(BLAS.vendor()), "openblas")
        println(BLAS.openblas_get_config())
    end
end

@testset "Aqua" begin
    # we can't check for unbound type parameters
    # because we actually need one at one point for _same_family()
    Aqua.test_all(MixedModels; ambiguities=false, unbound_args=false)
end

include("utilities.jl")
include("misc.jl")
include("pivot.jl")
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
include("grouping.jl")
include("bootstrap.jl")
include("mime.jl")
include("optsummary.jl")
include("predict.jl")
include("sigma.jl")
