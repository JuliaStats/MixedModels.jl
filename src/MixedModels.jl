module MixedModels

using Arrow: Arrow
using Base: Ryu, require_one_based_indexing
using BSplineKit: BSplineKit, BSplineOrder, Natural, Derivative, SplineInterpolation
using BSplineKit: interpolate
using Compat: @compat
using DataAPI: DataAPI, levels, refpool, refarray, refvalue
using Distributions: Distributions, Bernoulli, Binomial, Chisq, Distribution, Gamma
using Distributions: InverseGaussian, Normal, Poisson, ccdf
using GLM: GLM, GeneralizedLinearModel, IdentityLink, InverseLink, LinearModel
using GLM: Link, LogLink, LogitLink, ProbitLink, SqrtLink
using GLM: canonicallink, glm, linkinv, dispersion, dispersion_parameter
using JSON3: JSON3
using LinearAlgebra: LinearAlgebra, Adjoint, BLAS, BlasFloat, ColumnNorm
using LinearAlgebra: Diagonal, Hermitian, HermOrSym, I, LAPACK, LowerTriangular
using LinearAlgebra: PosDefException, SVD, SymTridiagonal, Symmetric
using LinearAlgebra: UpperTriangular, cond, diag, diagind, dot, eigen, isdiag
using LinearAlgebra: ldiv!, lmul!, logdet, mul!, norm, normalize, normalize!, qr
using LinearAlgebra: rank, rdiv!, rmul!, svd, tril!
using Markdown: Markdown
using MixedModelsDatasets: dataset, datasets
using PooledArrays: PooledArrays, PooledArray
using NLopt: NLopt
using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload
using ProgressMeter: ProgressMeter, Progress, finish!, next!
using Random: Random, AbstractRNG, randn!
using SparseArrays: SparseArrays, SparseMatrixCSC, SparseVector, dropzeros!, nnz
using SparseArrays: nonzeros, nzrange, rowvals, sparse
using StaticArrays: StaticArrays, SVector
using Statistics: Statistics, mean, quantile, std
using StatsAPI: StatsAPI, aic, aicc, bic, coef, coefnames, coeftable, confint
using StatsAPI: cooksdistance, deviance
using StatsAPI: dof, dof_residual, fit, fit!, fitted, isfitted, islinear, leverage
using StatsAPI:
    loglikelihood, meanresponse, modelmatrix, nobs, pvalue, predict, r2, residuals
using StatsAPI: response, responsename, stderror, vcov, weights
using StatsBase: StatsBase, CoefTable, model_response, summarystats
using StatsFuns: log2π, normccdf
using StatsModels: StatsModels, AbstractContrasts, AbstractTerm, CategoricalTerm
using StatsModels: ConstantTerm, DummyCoding, EffectsCoding, FormulaTerm, FunctionTerm
using StatsModels: HelmertCoding, HypothesisCoding, InteractionTerm, InterceptTerm
using StatsModels: MatrixTerm, SeqDiffCoding, TableRegressionModel
using StatsModels: apply_schema, drop_term, formula, lrtest, modelcols, @formula
using StructTypes: StructTypes
using Tables: Tables, columntable
using TypedTables: TypedTables, DictTable, FlexTable, Table

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
    cooksdistance,
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
    glmm,
    isfitted,
    islinear,
    issingular,
    leverage,
    levels,
    lmm,
    logdet,
    loglikelihood,
    lowerbd,
    lrtest,
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
    profilesigma,
    profilevc,
    pvalue,
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
    settheta!,
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

@compat public rePCA, PCA, dataset, datasets

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
include("nlopt.jl")
using .MixedModelsNLoptExt

include("derivatives.jl")

# aliases with non-unicode function names
const settheta! = setθ!
const profilesigma = profileσ

# COV_EXCL_START
@setup_workload begin
    # Putting some things in `setup` can reduce the size of the
    # precompile file and potentially make loading faster.
    sleepstudy = dataset(:sleepstudy)
    contra = dataset(:contra)
    progress = false
    io = IOBuffer()
    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)

        # these are relatively small models and so shouldn't increase precompile times all that much
        # while still massively boosting load and TTFX times
        m = fit(MixedModel,
            @formula(reaction ~ 1 + days + (1 + days | subj)),
            sleepstudy; progress)
        show(io, m)
        show(io, m.PCA.subj)
        show(io, m.rePCA)
        fit(MixedModel,
            @formula(use ~ 1 + age + abs2(age) + urban + livch + (1 | urban & dist)),
            contra,
            Bernoulli();
            progress)
    end
end
# COV_EXCL_STOP

end # module
