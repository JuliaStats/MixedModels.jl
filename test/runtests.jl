using MixedModels
import InteractiveUtils: versioninfo
import LinearAlgebra: BLAS

# there seem to be processor-specific issues and knowing this is helpful
println(versioninfo())
@static if VERSION ≥ v"1.7.0-DEV.620"
    @show getproperty.(BLAS.get_config().loaded_libs, :libname)
else
    @show BLAS.vendor()
    if startswith(string(BLAS.vendor()), "openblas")
        println(BLAS.openblas_get_config())
    end
end

include("utilities.jl")
include("pivot.jl")
include("UniformBlockDiagonal.jl")
include("linalg.jl")
include("matrixterm.jl")
include("FactorReTerm.jl")
include("pls.jl")
@static if VERSION < v"1.7.0-DEV.700"
    include("pirls.jl")
end
include("gausshermite.jl")
include("fit.jl")
include("missing.jl")
include("likelihoodratiotest.jl")
include("grouping.jl")
include("bootstrap.jl")
@static if VERSION < v"1.7.0-DEV.700"
    include("mime.jl")
end
