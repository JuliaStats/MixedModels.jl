struct MixedModelProfile{T<:AbstractFloat}
    m::LinearMixedModel{T}    # Model that has been profiled
    tbl::Table                # Table containing ζ, σ, β, and θ from each conditional fit
    fwd::Dict{Symbol}         # Interpolation splines for ζ as a function of a parameter
    rev::Dict{Symbol}         # Interpolation splines for a parameter as a function of ζ
end

include("utilities.jl")
include("fixefpr.jl")
include("sigmapr.jl")
include("thetapr.jl")
include("vcpr.jl")

"""
    profile(m::LinearMixedModel; threshold = 4)

Return a `MixedModelProfile` for the objective of `m` with respect to the fixed-effects coefficients.

Profiling starts at the parameter estimate and continues until reaching a parameter bound or the absolute
value of ζ exceeds `threshold`.
"""
function profile(m::LinearMixedModel{T}; threshold=4) where {T}
    final = copy(refit!(m).optsum.final)
    tc = TableColumns(m)
    val = profileσ(m, tc; threshold) # FIXME: defer creating the splines until the whole table is constructed
    objective!(m, final)   # restore the parameter estimates
    for s in filter(s -> startswith(string(s), 'β'), keys(first(val.tbl)))
        profileβj!(val, tc, s; threshold)
    end
    copyto!(m.optsum.final, final)
    m.optsum.fmin = objective!(m, final)
    for s in filter(s -> startswith(string(s), 'θ'), keys(first(val.tbl)))
        profileθj!(val, s, tc; threshold)
    end
    profileσs!(val, tc)
    objective!(m, final)   # restore the parameter estimates
    updateL!(setθ!(m, final))
    copyto!(m.optsum.final, final)
    m.optsum.fmin = objective(m)
    m.optsum.sigma = nothing
    return MixedModelProfile(m, Table(val.tbl), val.fwd, val.rev)
end

function StatsBase.confint(pr::MixedModelProfile; level::Real=0.95)
    cutoff = sqrt(quantile(Chisq(1), level))
    rev = pr.rev
    syms = sort!(collect(filter(k -> !startswith(string(k), 'θ'), keys(rev))))
    return DictTable(;
        par=syms,
        estimate=[rev[s](false) for s in syms],
        lower=[rev[s](-cutoff) for s in syms],
        upper=[rev[s](cutoff) for s in syms],
    )
end
