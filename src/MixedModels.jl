module MixedModels

using Arrow
using DataAPI
using Distributions
using GLM
using JSON3
using LazyArtifacts
using LinearAlgebra
using Markdown
using NLopt
using Random
using PooledArrays
using ProgressMeter
using SparseArrays
using StaticArrays
using Statistics
using StatsBase
using StatsModels
using StructTypes
using Tables

using LinearAlgebra: BlasFloat, BlasReal, HermOrSym, PosDefException, copytri!
using Base: Ryu
using GLM: Link, canonicallink, linkfun, linkinv
using StatsModels: TableRegressionModel

using StatsFuns: log2π, normccdf

import Base: *
import DataAPI: levels, refpool, refarray, refvalue
import GLM: dispersion, dispersion_parameter
import NLopt: Opt
import StatsBase: fit, fit!

export @formula,
    AbstractReMat,
    Bernoulli,
    Binomial,
    BlockDescription,
    BlockedSparse,
    DummyCoding,
    EffectsCoding,
    Grouping,
    Gamma,
    GeneralizedLinearMixedModel,
    HelmertCoding,
    HypothesisCoding,
    IdentityLink,
    InverseGaussian,
    InverseLink,
    LinearMixedModel,
    LogitLink,
    LogLink,
    MixedModel,
    MixedModelBootstrap,
    Normal,
    OptSummary,
    Poisson,
    ProbitLink,
    RaggedArray,
    RandomEffectsTerm,
    ReMat,
    SeqDiffCoding,
    SqrtLink,
    UniformBlockDiagonal,
    VarCorr,
    aic,
    aicc,
    bic,
    coef,
    coefnames,
    coefpvalues,
    coeftable,
    columntable,
    cond,
    condVar,
    condVartables,
    describeblocks,
    deviance,
    dispersion,
    dispersion_parameter,
    dof,
    dof_residual,
    fit,
    fit!,
    fitted,
    fitted!,
    fixef,
    fixefnames,
    formula,
    fulldummy,
    fnames,
    GHnorm,
    isfitted,
    islinear,
    issingular,
    leverage,
    levels,
    logdet,
    loglikelihood,
    lowerbd,
    meanresponse,
    modelmatrix,
    model_response,
    nobs,
    objective,
    parametricbootstrap,
    pirls!,
    predict,
    pwrss,
    ranef,
    raneftables,
    rank,
    refarray,
    refit!,
    refpool,
    refvalue,
    replicate,
    residuals,
    response,
    responsename,
    restoreoptsum!,
    saveoptsum,
    shortestcovint,
    sdest,
    setθ!,
    simulate,
    simulate!,
    sparse,
    sparseL,
    std,
    stderror,
    updateL!,
    varest,
    vcov,
    weights,
    zerocorr

import Base: ==, *

"""
    MixedModel

Abstract type for mixed models.  MixedModels.jl implements two subtypes:
`LinearMixedModel` and `GeneralizedLinearMixedModel`.  See the documentation for
each for more details.

This type is primarily used for dispatch in `fit`.  Without a distribution and
link function specified, a `LinearMixedModel` will be fit.  When a
distribution/link function is provided, a `GeneralizedLinearModel` is fit,
unless that distribution is `Normal` and the link is `IdentityLink`, in which
case the resulting GLMM would be equivalent to a `LinearMixedModel` anyway and
so the simpler, equivalent `LinearMixedModel` will be fit instead.
"""
abstract type MixedModel{T} <: StatsModels.RegressionModel end # model with fixed and random effects

include("utilities.jl")
include("blocks.jl")
include("pca.jl")
include("datasets.jl")
include("arraytypes.jl")
include("varcorr.jl")
include("Xymat.jl")
include("remat.jl")
include("optsummary.jl")
include("schema.jl")
include("randomeffectsterm.jl")
include("linearmixedmodel.jl")
include("gausshermite.jl")
include("generalizedlinearmixedmodel.jl")
include("mixedmodel.jl")
include("likelihoodratiotest.jl")
include("linalg/pivot.jl")
include("linalg/cholUnblocked.jl")
include("linalg/rankUpdate.jl")
include("linalg/logdet.jl")
include("linalg.jl")
include("simulate.jl")
include("predict.jl")
include("bootstrap.jl")
include("blockdescription.jl")
include("grouping.jl")
include("mimeshow.jl")
include("serialization.jl")

end # module
