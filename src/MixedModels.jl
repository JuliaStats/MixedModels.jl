using DataFrames  # should be externally available
module MixedModels

    using ArrayViews, DataArrays, DataFrames, Distributions
    using NLopt, NumericExtensions, NumericFuns, StatsBase
    using StatsBase: CoefTable
    using Base.LinAlg.CHOLMOD: CholmodFactor, CholmodSparse, CholmodSparse!,
          chm_scale, CHOLMOD_SYM, CHOLMOD_L, CHOLMOD_Lt, solve

    export
        LinearMixedModel,
        MixedModel,
        PLSDiag,               # multiple, scalar random-effects terms
        PLSGeneral,            # general random-effects structure
        PLSOne,                # solver for models with only one r.e. term
        PLSSolver,

        fixef,          # extract the fixed-effects parameter estimates
        grad!,          # install gradient of objective
        grplevels,      # number of levels per grouping factor in mixed-effects models
        isfit,          # predictate to check if a model has been fit
        isnested,       # check if vector f is nested in vector g
        isscalar,       # are all the random-effects terms in the model scalar?
        hasgrad,        # can the analytic gradient of the objective function be evaluated
        lmm,            # fit a linear mixed-effects model (LMM)
        lower,          # vector of lower bounds on parameters in mixed-effects models
        objective,      # the objective function in fitting a model
        pwrss,          # penalized, weighted residual sum-of-squares
        ranef,          # extract the conditional modes of the random effects
        reml!,          # set the objective to be the REML criterion
        isreml          # is the objective the REML criterion?

    abstract MixedModel          # model with fixed and random effects
    abstract PLSSolver           # type for solving the penalized least squares problem

    include("utils.jl")
    include("plsgeneral.jl")
    include("plsone.jl")
    include("plsdiag.jl")
    include("linearmixedmodels.jl")
end #module
