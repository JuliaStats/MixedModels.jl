#VERSION >= v"0.4.0-dev+6521" && __precompile__()

module MixedModels

using Compat, DataArrays, DataFrames, Distributions, NLopt, Showoff, StatsBase

export ReMat, ScalarReMat,VectorReMat

export LinearMixedModel,
       MixedModel,
       VarCorr,

       AIC,        # Akaike's Information Criterion
       BIC,        # Schwatz's Bayesian Information Criterion
       bootstrap,  # Create bootstrap replications of a model
       fixef,      # extract the fixed-effects parameter estimates
       lmm,        # create a LinearMixedModel from a formula/data specification
       lowerbd,    # lower bounds on the covariance parameters
       npar,       # total number of parameters in the model
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
include("paramlowertriangular.jl")
include("cfactor.jl")
include("inject.jl")
include("pls.jl")
include("logdet.jl")
include("bootstrap.jl")
include("GLMM/glmtools.jl")
include("GLMM/PIRLS.jl")

end # module
