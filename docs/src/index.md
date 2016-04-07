# MixedModels.jl Documentation

    {contents}

## Types

    {docs}
    LinearMixedModel
    MixedModels.OptSummary
    ScalarReMat
    VarCorr
    VectorReMat

## Public Functions

    {docs}
    bootstrap
    coef(::MixedModel)
    coeftable(::MixedModel)
    cond(::MixedModel)
    df(::LinearMixedModel)
    deviance(::LinearMixedModel)
    fit!(::LinearMixedModel)
    fitted(::LinearMixedModel)
    fixef
    LaplaceDeviance
    lmm
    lowerbd
    model_response(::LinearMixedModel)
    nobs(::LinearMixedModel)
    objective
    pwrss
    pirls!
    ranef
    refit!
    remat
    sdest
    simulate!
    varest
    vcov

## Internal Functions

    {docs}
    MixedModels.cfactor!
    MixedModels.chol2cor
    MixedModels.densify
    MixedModels.describeblocks
    MixedModels.downdate!
    MixedModels.feR,
    MixedModels.fixef!
    MixedModels.fnames
    MixedModels.grplevels
    MixedModels.inflate!
    MixedModels.inject!
    MixedModels.isfit
    MixedModels.lrt
    MixedModels.nlevs,
    MixedModels.pirls!
    MixedModels.ranef!
    MixedModels.reevaluateAend!
    MixedModels.resetθ!
    MixedModels.reterms
    MixedModels.reweight!
    MixedModels.sqrtpwrss
    MixedModels.tscale!
    MixedModels.unscaledre!
    MixedModels.vsize

## Index

    {index}
