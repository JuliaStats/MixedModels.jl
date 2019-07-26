module MixedModels

using BlockArrays, CategoricalArrays, Tables, Distributions, GLM, 
    LinearAlgebra, NLopt, Random, ProgressMeter, Showoff, SparseArrays, StaticArrays,
    Statistics, StatsBase, StatsModels, TypedTables

using LinearAlgebra: BlasFloat, BlasReal, HermOrSym, PosDefException, copytri!
#using NamedArrays: NamedArray, setnames!
using Printf: @printf, @sprintf

using StatsFuns: log2π

import Base: *
import NLopt: Opt

export
       @formula,
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
       MixedModel,
       OptSummary,
       Poisson,
       RaggedArray,
       RandomEffectsTerm,
       ReMat,
       UniformBlockDiagonal,
       VarCorr,

       aic,
       aicc,
       bic,
       coef,
       coeftable,
       cond,
       describeblocks,
       condVar,
       deviance,
       dof,
       fit,
       fit!,
       fitted,
       fixef,      # extract the fixed-effects parameter estimates
       fnames,
       GHnorm,
       loglikelihood,
       lowerbd,    # lower bounds on the covariance parameters
       nblocks,
       nobs,
       nocorr,
       objective,  # the objective function in fitting a model
       parametricbootstrap,
       pirls!,     # use Penalized Iteratively Reweighted Least Squares to obtain conditional modes of random effects
       predict,
       pwrss,      # penalized, weighted residual sum-of-squares
       ranef,      # extract the conditional modes of the random effects
       refit!,     # install a response and refit the model
       residuals,
       response,
       sdest,      # the estimate of the standard deviation of the per-observation noise
       setθ!,
       simulate!,  # simulate a new response in place
       sparse,
       statscholesky,
       std,
       updateL!,   # update the lower-triangular, blocked matrix L to a new θ
       varest,     # estimate of the residual variance
       vcov

import Base: ==, *

abstract type MixedModel{T} <: StatsModels.RegressionModel end # model with fixed and random effects

include("utilities.jl")
include("arraytypes.jl")
include("optsummary.jl")
include("varcorr.jl")
include("femat.jl")
include("remat.jl")
include("randomeffectsterm.jl")
include("linearmixedmodel.jl")
include("gausshermite.jl")
include("generalizedlinearmixedmodel.jl")
include("linalg/statschol.jl")
include("linalg/cholUnblocked.jl")
include("linalg/rankUpdate.jl")
include("linalg/logdet.jl")
include("linalg.jl")
include("simulate.jl")

end # module
