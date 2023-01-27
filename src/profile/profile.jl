struct MixedModelProfile{T<:AbstractFloat} 
    m::LinearMixedModel{T}    # Model that has been profiled
    tbl::Vector{<:NamedTuple} # Table containing ζ, σ, β, and θ from each conditional fit
    fwd::Dict{Symbol}         # Interpolation splines for ζ as a function of a parameter
    rev::Dict{Symbol}         # Interpolation splines for a parameter as a function of ζ
end

include("utilities.jl")
include("fixefpr.jl")
include("sigmapr.jl")
include("thetapr.jl")
include("vcpr.jl")

"""
    profile(m::LinearMixedModel)

Return a `MixedModelProfile` for the objective of `m` with respect to the fixed-effects coefficients.
"""
function profile(m::LinearMixedModel{T}; threshold = 4) where {T}
    final = copy(refit!(m).optsum.final)
    tc = TableColumns(m)
    val = profileσ(m, tc; threshold) # FIXME: defer creating the splines until the whole table is constructed
    copyto!(m.optsum.final, final)
    m.optsum.fmin = objective!(m, final)
    for s in filter(s -> startswith(string(s), 'θ'), keys(first(val.tbl)))
        profileθj!(val, s, tc; threshold)
    end
    objective!(m, final)   # restore the parameter estimates
    for s in filter(s -> startswith(string(s), 'β'), keys(first(val.tbl)))
        profileβj!(val, tc, s; threshold)
    end
    profileσs!(val, tc)
    objective!(m, final)   # restore the parameter estimates
    updateL!(setθ!(m, final))
    copyto!(m.optsum.final, final)
    m.optsum.fmin = objective(m)
    m.optsum.sigma = nothing
    return val
end

function StatsBase.confint(pr::MixedModelProfile; level::Real=0.95)
    cutoff = sqrt.(quantile(Chisq(1), level))
    rev = pr.rev
    syms = sort!(collect(keys(rev)))
    return DictTable(; 
        coef=syms,
        estimate=[rev[s](false) for s in syms],
        lower=[rev[s](-cutoff) for s in syms],
        upper=[rev[s](cutoff) for s in syms],
    )
end
