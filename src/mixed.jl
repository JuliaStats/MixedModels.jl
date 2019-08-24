# LinearMixedModel
fit(::Type{MixedModel}, f::FormulaTerm, tbl;
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    verbose::Bool = false,
    REML::Bool = false) =
    fit(LinearMixedModel,
        f, tbl, wts = wts, contrasts = contrasts,
        verbose = verbose, REML = REML)
# GeneralizedLinearMixedModel
fit(::Type{MixedModel}, f::FormulaTerm, tbl,
    d::Distribution, l::Link = canonicallink(d);
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    offset = [],
    verbose::Bool = false,
    REML::Bool = false,
    fast::Bool = false,
    nAGQ::Integer = 1) =
    fit(GeneralizedLinearMixedModel,
        f, tbl, d, l, wts = wts, contrasts = contrasts, offset = offset,
        verbose = verbose, fast = fast, nAGQ = nAGQ)
# LinearMixedModel
fit(::Type{MixedModel}, f::FormulaTerm, tbl,
    d::Normal, l::IdentityLink;
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    verbose::Bool = false,
    REML::Bool = false,
    offset = [],
    fast::Bool = false,
    nAGQ::Integer = 1) =
    fit(LinearMixedModel, f, tbl,
        wts = wts, contrasts = contrasts,
        verbose = verbose, REML = REML)
