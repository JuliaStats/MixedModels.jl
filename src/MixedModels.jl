__precompile__()

module MixedModels

using ArgCheck, CategoricalArrays, Compat, DataArrays, DataFrames, Distributions, GLM
using NLopt, Showoff, StatsBase
using StatsFuns: log2π
using NamedArrays: NamedArray, setnames!
using Base.LinAlg: BlasFloat, BlasReal, HermOrSym, PosDefException, checksquare, copytri!

import Base: cor, cond, convert, full, logdet, std, A_mul_B!, Ac_mul_B!, A_mul_Bc!
import DataFrames: @formula
import Distributions: Bernoulli, Binomial, Poisson, Gamma
import GLM: LogitLink, LogLink, InverseLink
import NLopt: Opt
import Base.LinAlg: A_mul_B!, A_mul_Bc!, Ac_mul_B!, A_ldiv_B!, Ac_ldiv_B!, A_rdiv_B!, A_rdiv_Bc!
import StatsBase: coef, coeftable, dof, deviance, fit!, fitted, loglikelihood,
    model_response, nobs, vcov

export
       @formula,
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
#       ScalarReMat,
       VarCorr,
#       VectorReMat,

       bootstrap,
       bootstrap!,
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
       updateL!,   # update the lower-triangular, blocked matrix L to a new θ
       varest,     # estimate of the residual variance
       vcov

import Base: ==, *

include("types.jl")
include("linalg/cholUnblocked.jl")
include("linalg/rankUpdate.jl")
include("linalg/scaleInflate.jl")
include("linalg.jl")
include("blockmats.jl")
include("remat.jl")
include("pls.jl")
include("logdet.jl")
include("simulate.jl")
include("PIRLS.jl")
#include("VarCorr.jl")
include("mixedmodel.jl")

end # module
