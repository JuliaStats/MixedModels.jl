#VERSION >= v"0.4.0-dev+6521" && __precompile__()

module MixedModels

using DataArrays, DataFrames, Distributions, NLopt, Showoff, StatsBase

export ScalarReMat,VectorReMat

export LinearMixedModel,
       MixedModel,
       VarCorr,

       AIC,        # Akaike's Information Criterion
       BIC,        # Schwatz's Bayesian Information Criterion
       fixef,      # extract the fixed-effects parameter estimates
       lmm,        # create a LinearMixedModel from a formula/data specification
       lowerbd,    # lower bounds on the covariance parameters
       objective,  # the objective function in fitting a model
       pwrss,      # penalized, weighted residual sum-of-squares
       ranef,      # extract the conditional modes of the random effects
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

end # module
