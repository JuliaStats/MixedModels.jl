using DataFrames, Distributions  # should be externally available
module MixedModels

    using DataFrames, NLopt, NumericExtensions
    using Base.SparseMatrix: symperm
    using Base.LinAlg.BLAS: gemm!, gemv!, syrk!, syrk, trmm!, trmm,
          trmv!, trsm!, trsv!
    using Base.LinAlg.CHOLMOD: CholmodDense, CholmodDense!, CholmodFactor,
          CholmodSparse, CholmodSparse!, chm_scale!, CHOLMOD_SYM,
          CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_P, CHOLMOD_Pt
    using Base.LinAlg.LAPACK:  potrf!, potrs!

    import Base: Ac_mul_B!, At_mul_B!, cor, cholfact, logdet, scale, show, size, solve, std
    import Distributions: fit
    import Stats: coef, coeftable, confint, stderr, vcov
    import GLM: deviance, df_residual, linpred

    export
        MixedModel,
        LinearMixedModel,
        LMMGeneral,
        LMMNested,
        LMMScalar1,
        LMMScalarn,
        LMMVector1,

        fixef,          # extract the fixed-effects parameter estimates
        grplevels,      # number of levels per grouping factor in mixed-effects models
        isfit,          # predictate to check if a model has been fit
        isnested,       # check if vector f is nested in vector g
        lmm,            # fit a linear mixed-effects model (LMM)
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
    include("linearmixedmodels.jl") # method definitions for the abstract class
    include("general.jl") # general form of linear mixed-effects models
    include("scalar1.jl") # models with a single, scalar random-effects term
    include("scalarn.jl") # models with a single, scalar random-effects term
    include("vector1.jl") # models with a single, vector-valued random-effects term
    include("nested.jl")
    include("lmm.jl")    # fit and analyze linear mixed-effects models

end #module
