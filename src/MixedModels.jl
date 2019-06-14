module MixedModels

using ArgCheck, BlockArrays, CategoricalArrays, DataFrames, Distributions, GLM,
    LinearAlgebra, NLopt, Random, ProgressMeter, Showoff, SparseArrays, StaticArrays,
    Statistics, StatsBase, StatsModels

using LinearAlgebra: BlasFloat, BlasReal, HermOrSym, PosDefException, copytri!
using NamedArrays: NamedArray, setnames!
using Printf: @printf, @sprintf

using StatsFuns: log2π

import Base: *
import NLopt: Opt

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
       RaggedArray,
       RepeatedBlockDiagonal,
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
       fitted,
       fixef,      # extract the fixed-effects parameter estimates
       fnames,
       getΛ,
       getθ,
       GHnorm,
       Λ,
       Lambda,
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
       residuals,
       sdest,      # the estimate of the standard deviation of the per-observation noise
       setθ!,
       simulate!,  # simulate a new response in place
       std,
       updateL!,   # update the lower-triangular, blocked matrix L to a new θ
       varest,     # estimate of the residual variance
       vcov

import Base: ==, *

include("types.jl")
include("gausshermite.jl")
include("modelterms.jl")
include("linalg/cholUnblocked.jl")
include("linalg/rankUpdate.jl")
include("linalg/scaleInflate.jl")
include("linalg/logdet.jl")
include("linalg.jl")
include("pls.jl")
include("simulate.jl")
include("PIRLS.jl")
include("mixedmodel.jl")
include("deprecates.jl")

end # module
