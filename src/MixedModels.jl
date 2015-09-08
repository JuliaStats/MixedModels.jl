using DataFrames  # should be externally available

# VERSION >= v"0.4.0-dev+6521" && __precompile__()

module MixedModels

    using ArrayViews, DataArrays, DataFrames, Distributions, NLopt, Showoff, StatsBase

    using Base.SparseMatrix.CHOLMOD
    using Base.LinAlg: Ac_ldiv_B!, A_rdiv_Bc!, chksquare

    export
        GeneralizedLinearMixedModel,
        LinearMixedModel,
        MixedModel,
        PLSDiag,    # multiple, scalar random-effects terms
#        PLSDiagWA,  # multiple, scalar random-effects terms using WSMP
#        PLSGeneral, # general random-effects structure
        PLSNested,  # solver for models whose grouping factors form a nested sequence
        PLSOne,     # solver for models with only one r.e. term
        PLSSolver,  # abstract type for a penalized least squares solver
        PLSTwo,     # solver for models with two crossed or nearly crossed r.e. terms

        fixef,      # extract the fixed-effects parameter estimates
        glmm,       # create a GeneralizedLinearMixedModel from a formula/data specification
        grad!,      # install gradient of objective
        grplevels,  # number of levels per grouping factor in mixed-effects models
        isfit,      # predictate to check if a model has been fit
        isnested,   # check if vector f is nested in vector g
        isscalar,   # are all the random-effects terms in the model scalar?
        hasgrad,    # can the analytic gradient of the objective function be evaluated
        lmm,        # create a LinearMixedModel from a formula/data specification
        lower,      # lower bounds on the covariance parameters
        objective,  # the objective function in fitting a model
        pwrss,      # penalized, weighted residual sum-of-squares
        ranef,      # extract the conditional modes of the random effects
        reml!       # set the objective to be the REML criterion

    abstract MixedModel <: RegressionModel # model with fixed and random effects

    include("utils.jl")
    include("pdmats.jl")
    include("plssolver.jl")
#    include("plsnested.jl")
    include("plsone.jl")
    include("plstwo.jl")
    include("plsdiag.jl")
    include("linearmixedmodels.jl")
#    include("glmtools.jl")
#    include("PIRLS.jl")
end #module
