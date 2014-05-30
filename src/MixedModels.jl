using DataFrames  # should be externally available
module MixedModels

    using DataArrays, DataFrames, NLopt, NumericExtensions, NumericFuns
    using StatsBase: CoefTable
    using Base.SparseMatrix: symperm
    using Base.LinAlg.CHOLMOD: CholmodDense, CholmodDense!, CholmodFactor,
          CholmodSparse, CholmodSparse!, chm_scale!, CHOLMOD_SYM,
          CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_P, CHOLMOD_Pt, solve

    export
        LinearMixedModel,
        LMMBase,
        LMMGeneral,
        LMMNested,
        LMMScalar1,
        LMMScalarNested,
        LMMScalarn,
        LMMVector1,
        MixedModel,

        fixef,          # extract the fixed-effects parameter estimates
        grad,           # gradient of objective
        grplevels,      # number of levels per grouping factor in mixed-effects models
        isfit,          # predictate to check if a model has been fit
        isnested,       # check if vector f is nested in vector g
        lmm,            # fit a linear mixed-effects model (LMM)
        lmmp,
        lower,          # vector of lower bounds on parameters in mixed-effects models
        objective,      # the objective function in fitting a model
        pwrss,          # penalized, weighted residual sum-of-squares
        ranef,          # extract the conditional modes of the random effects
        reml!,          # set the objective to be the REML criterion
        reml,           # is the objective the REML criterion?
        solve!,         # update the coefficients by solving the MME's
        theta!,         # set the value of the variance component parameters        
        theta           # extract the variance-component parameter vector

    abstract MixedModel                # model with fixed and random effects
    abstract LinearMixedModel <: MixedModel # Gaussian mixed model with identity link

    typealias VTypes Union(Float64,Complex128)
    typealias ITypes Union(Int32,Int64)

    include("utils.jl")     # utilities to deal with the model formula
    include("LMMBase.jl")   # information common to each type of LinearMixedModel
    include("delta.jl")
    include("linearmixedmodels.jl") # method definitions for the abstract class
#    include("general.jl") # general form of linear mixed-effects models
#    include("scalar1.jl") # models with a single, scalar random-effects term
#    include("scalarn.jl") # models with a single, scalar random-effects term
#    include("vector1.jl") # models with a single, vector-valued random-effects term
#    include("nested.jl")
#    include("lmmMUMPS.jl")              # fit models using MUMPS solver
    include("lmm.jl")    # fit and analyze linear mixed-effects models

end #module
