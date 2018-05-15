__precompile__()

module MixedModels

using ArgCheck, BlockArrays, CategoricalArrays, Compat, DataFrames, Distributions
using GLM, NLopt, Showoff, StaticArrays, StatsBase, StatsModels
using StatsFuns: log2π
using NamedArrays: NamedArray, setnames!
using Compat.LinearAlgebra: BlasFloat, BlasReal, HermOrSym, PosDefException, checksquare, copytri!

import Base: cor, convert, eltype, full, logdet, std
import Compat.LinearAlgebra: A_mul_B!, A_mul_Bc!, Ac_mul_B!, A_ldiv_B!, Ac_ldiv_B!, A_rdiv_B!, A_rdiv_Bc!, cond
import NLopt: Opt
import StatsBase: coef, coeftable, dof, deviance, fit, fit!, fitted, loglikelihood,
    model_response, nobs, predict, stderror, vcov

export
       @formula,
       AbstractFactorReTerm,
       AbstractReTerm,
       Bernoulli,
       Binomial,
       Block,
       BlockedSparse,
       Gamma,
       LogitLink,
       LogLink,
       InverseGaussian,
       InverseLink,
       GeneralizedLinearMixedModel,
       LinearMixedModel,
       MatrixTerm,
       MixedModel,
       OptSummary,
       Poisson,
       ScalarFactorReTerm,
       UniformBlockDiagonal,
       VarCorr,
       VectorFactorReTerm,

       bootstrap,
       bootstrap!,
       coef,
       coeftable,
       cond,
       describeblocks,
       condVar,
       deviance,
       dof,
       fit,
       fit!,
       fitlmm,
       fitted,
       fixef,      # extract the fixed-effects parameter estimates
       fnames,
       getΛ,
       getθ,
       glmm,       # define a GeneralizedLinearMixedModel
       LaplaceDeviance, # Laplace approximation to GLMM deviance
       lmm,        # create a LinearMixedModel from a formula/data specification
       loglikelihood,
       lowerbd,    # lower bounds on the covariance parameters
       model_response,
       nblocks,
       nobs,
       objective,  # the objective function in fitting a model
       pwrss,      # penalized, weighted residual sum-of-squares
       pirls!,     # use Penalized Iteratively Reweighted Least Squares to obtain conditional modes of random effects
       predict,
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
include("modelterms.jl")
include("linalg/cholUnblocked.jl")
include("linalg/rankUpdate.jl")
include("linalg/scaleInflate.jl")
include("linalg/lambdaprods.jl")
include("linalg/logdet.jl")
include("linalg.jl")
include("pls.jl")
include("simulate.jl")
include("PIRLS.jl")
include("mixedmodel.jl")
include("deprecates.jl")

end # module
