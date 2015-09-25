#VERSION >= v"0.4.0-dev+6521" && __precompile__()

module MixedModels

using DataArrays, DataFrames, NLopt, StatsBase

export ScalarReMat,VectorReMat,ColMajorLowerTriangular,DiagonalLowerTriangular

export LinearMixedModel,
       MixedModel,

       AIC,        # Akaike's Information Criterion
       BIC,        # Schwatz's Bayesian Information Criterion
       fixef,      # extract the fixed-effects parameter estimates
       grad!,      # install gradient of objective
       hasgrad,    # can the analytic gradient of the objective function be evaluated
       lmm,        # create a LinearMixedModel from a formula/data specification
       lowerbd,    # lower bounds on the covariance parameters
       objective,  # the objective function in fitting a model
       pwrss,      # penalized, weighted residual sum-of-squares
       ranef,      # extract the conditional modes of the random effects
       reml!       # set the objective to be the REML criterion

abstract MixedModel <: RegressionModel # model with fixed and random effects

using Base.LinAlg.BlasInt

import Base: ==

include("ReTerms/densify.jl")
include("ReTerms/blockmats.jl")
include("ReTerms/remat.jl")
include("ReTerms/paramlowertriangular.jl")
include("ReTerms/cfactor.jl")
include("ReTerms/inject.jl")
include("ReTerms/pls.jl")

end # module
