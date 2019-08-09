function StatsBase.fit(
    ::Type{MixedModel},
    f::FormulaTerm,
    tbl,
    d::Distribution = Normal();
    kw...
)
    if isa(d, Normal)
        fit!(
            LinearMixedModel(
                f,
                columntable(tbl),
                hints = get(kw, :hints, Dict{Symbol,Any}())
            ),
            verbose = get(kw, :verbose, false),
            REML = get(kw, :REML, false)
        )
    else
        fit!(
            GeneralizedLinearMixedModel(
                f,
                columntable(tbl),
                d,
                GLM.canonicallink(d),
                wt = get(kw, :wt, []),
                offset = get(kw, :offset, []),
                hints = get(kw, :hints, Dict{Symbol,Any}())
            ),
            verbose = get(kw, :verbose, false),
            fast = get(kw, :fast, false),
            nAGQ = get(kw, :nAGQ, 1)
        )
    end
end
