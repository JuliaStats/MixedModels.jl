"""
     MixedModelProfile{T<:AbstractFloat}

Type representing a likelihood profile of a [`LinearMixedModel`](@ref), including associated interpolation splines.

The function [`profile`](@ref) is used for computing profiles, while [`confint`](@ref) provides a useful method for constructing confidence intervals from a `MixedModelProfile`.

!!! note
    The exact fields and their representation are considered implementation details and are
    **not** part of the public API.
"""
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

`m` is `refit!` if `!isfitted(m)`.

Profiling starts at the parameter estimate and continues until reaching a parameter bound or the absolute
value of ζ exceeds `threshold`.
"""
function profile(m::LinearMixedModel; threshold=4)
    isfitted(m) || refit!(m)
    final = copy(m.optsum.final)
    profile = try
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
        MixedModelProfile(m, Table(val.tbl), val.fwd, val.rev)
    catch ex
        @error "Exception occurred in profiling; aborting..."
        rethrow(ex)
    finally
        objective!(m, final)   # restore the parameter estimates
        copyto!(m.optsum.final, final)
        m.optsum.fmin = objective(m)
        m.optsum.sigma = nothing
    end
    return profile
end

"""
    confint(pr::MixedModelProfile; level::Real=0.95)

Compute profile confidence intervals for coefficients and variance components, with confidence level level (by default 95%).

!!! note
    The API guarantee is for a Tables.jl compatible table. The exact return type is an
    implementation detail and may change in a future minor release without being considered
    breaking.

!!! note
    The "row names" indicating the associated parameter name are guaranteed to be unambiguous,
    but their precise naming scheme is not yet stable and may change in a future release
    without being considered breaking.
"""
function StatsAPI.confint(pr::MixedModelProfile; level::Real=0.95)
    cutoff = sqrt(quantile(Chisq(1), level))
    rev = pr.rev
    syms = sort!(collect(filter(k -> !startswith(string(k), 'θ'), keys(rev))))
    dt = DictTable(;
        par=syms,
        estimate=[rev[s](0) for s in syms],
        lower=[rev[s](-cutoff) for s in syms],
        upper=[rev[s](cutoff) for s in syms],
    )

    # XXX for reasons I don't understand, the reverse spline for REML-models
    # is flipped for the fixed effects, even though the table of interpolation
    # points isn't.
    for i in keys(dt.lower)
        if dt.lower[i] > dt.upper[i]
            dt.lower[i], dt.upper[i] = dt.upper[i], dt.lower[i]
        end
    end

    return dt
end

function Base.show(io::IO, mime::MIME"text/plain", pr::MixedModelProfile)
    print(io, "MixedModelProfile -- ")
    show(io, mime, pr.tbl)
    return nothing
end

Tables.columns(pr::MixedModelProfile) = Tables.columns(pr.tbl)
