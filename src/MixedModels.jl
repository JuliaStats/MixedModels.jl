__precompile__()

module MixedModels

using CategoricalArrays, DataArrays, DataFrames, Distributions, GLM, NLopt, Showoff
using StaticArrays, StatsBase
using StatsFuns: log2π
using NamedArrays: NamedArray, setnames!

import StatsBase: coef, coeftable, dof, deviance, fit!, fitted, loglikelihood,
    model_response, nobs, vcov
import Base: cor, cond, convert, std
import Distributions: Bernoulli, Binomial, Poisson, Gamma
import GLM: LogitLink, LogLink, InverseLink
import DataFrames: @~

export
       @~,
       Bernoulli,
       Binomial,
       Poisson,
       Gamma,
       LogitLink,
       LogLink,
       InverseLink,
       GeneralizedLinearMixedModel,
       LinearMixedModel,
       MixedModel,
       OptSummary,
       ReMat,
       ScalarReMat,
       VarCorr,
       VectorReMat,

       bootstrap,
       bootstrap!,
       cfactor!,
       coef,
       coeftable,
       cond,
       condVar,
       dof,
       deviance,
       fit!,
       fitted,
       fixef,      # extract the fixed-effects parameter estimates
       getθ,
       glmm,       # define a GeneralizedLinearMixedModel
       LaplaceDeviance, # Laplace approximation to GLMM deviance
       lmm,        # create a LinearMixedModel from a formula/data specification
       loglikelihood,
       lowerbd,    # lower bounds on the covariance parameters
       model_response,
       nobs,
       objective,  # the objective function in fitting a model
       pwrss,      # penalized, weighted residual sum-of-squares
       pirls!,     # use Penalized Iteratively Reweighted Least Squares to obtain conditional modes of random effects
       ranef,      # extract the conditional modes of the random effects
       refit!,     # install a response and refit the model
       remat,      # factory for construction of ReMat objects
       sdest,      # the estimate of the standard deviation of the per-observation noise
       setθ!,
       simulate!,  # simulate a new response in place
       std,
       varest,     # estimate of the residual variance
       vcov

abstract MixedModel <: RegressionModel # model with fixed and random effects

import Base: ==, *

include("blockmats.jl")
include("linalg.jl")
include("cfactor.jl")
include("remat.jl")
include("paramlowertriangular.jl")
include("inject.jl")
include("pls.jl")
include("logdet.jl")
include("simulate.jl")
include("PIRLS.jl")
#include("glm.jl")
include("mixedmodel.jl")

end # module
