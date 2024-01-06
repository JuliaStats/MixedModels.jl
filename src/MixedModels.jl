module MixedModels

using Arrow
using BSplineKit
using DataAPI
using Distributions
using GLM
using JSON3
using LinearAlgebra
using Markdown
using NLopt
using Random
using PooledArrays
using ProgressMeter
using PRIMA
using SparseArrays
using StaticArrays
using Statistics
using StatsAPI
using StatsBase
using StatsModels
using StructTypes
using Tables
using TypedTables

using LinearAlgebra: BlasFloat, BlasReal, HermOrSym, PosDefException, copytri!
using Base: Ryu, require_one_based_indexing
using DataAPI: levels, refpool, refarray, refvalue
using GLM: Link, canonicallink, linkfun, linkinv, dispersion, dispersion_parameter
using MixedModelsDatasets: dataset, datasets
using NLopt: Opt
using StatsModels: TableRegressionModel
using StatsFuns: log2π, normccdf

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
    MixedModelProfile,
    Normal,
    OptSummary,
    Poisson,
    ProbitLink,
    RaggedArray,
    RandomEffectsTerm,
    ReMat,
    SeqDiffCoding,
    SqrtLink,
    Table,
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
    confint,
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
    objective!,
    parametricbootstrap,
    pirls!,
    predict,
    profile,
    profileσ,
    profilevc,
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
    stderror!,
    updateL!,
    varest,
    vcov,
    weights,
    zerocorr

# TODO: move this to the correct spot in list once we've decided on name
export savereplicates, restorereplicates

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
include("profile/profile.jl")

using PrecompileTools

@setup_workload begin
    # Putting some things in `setup` can reduce the size of the
    # precompile file and potentially make loading faster.
    sleepstudy = MixedModels.dataset(:sleepstudy)
    contra = MixedModels.dataset(:contra)
    progress = false
    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)

        # these are relatively small models and so shouldn't increase precompile times all that much
        # while still massively boosting load and TTFX times
        fit(MixedModel,
            @formula(reaction ~ 1 + days + (1 + days | subj)),
            sleepstudy; progress)
        fit(MixedModel,
            @formula(use ~ 1 + age + abs2(age) + urban + livch + (1 | urban & dist)),
            contra,
            Bernoulli();
            progress)
    end
end

end # module
