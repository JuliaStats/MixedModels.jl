__precompile__()

module MixedModels

using Compat, DataArrays, GLM, DataFrames, Distributions, NLopt, Showoff, StatsBase

export GeneralizedLinearMixedModel,
       LinearMixedModel,
       MixedModel,
       ReMat,
       ScalarReMat,
       VarCorr,
       VectorReMat,

       bootstrap,  # Create bootstrap replications of a model
       fixef,      # extract the fixed-effects parameter estimates
       glmm,       # define a GeneralizedLinearMixedModel
       lmm,        # create a LinearMixedModel from a formula/data specification
       lowerbd,    # lower bounds on the covariance parameters
       objective,  # the objective function in fitting a model
       pwrss,      # penalized, weighted residual sum-of-squares
       ranef,      # extract the conditional modes of the random effects
       refit!,     # install a response and refit the model
       remat,      # factory for construction of ReMat objects
       reml!,      # set the objective to be the REML criterion
       sdest,      # the estimate of the standard deviation of the per-observation noise
       simulate!,  # simulate a new response and refit the model
       varest      # estimate of the residual variance

abstract MixedModel <: RegressionModel # model with fixed and random effects

import Base: ==

include("densify.jl")
include("blockmats.jl")
include("linalg.jl")
include("inflate.jl")
include("remat.jl")
include("reweight.jl")
include("paramlowertriangular.jl")
include("cfactor.jl")
include("inject.jl")
include("pls.jl")
include("logdet.jl")
include("bootstrap.jl")
include("PIRLS.jl")
include("glm.jl")
include("mixedmodel.jl")

end # module
