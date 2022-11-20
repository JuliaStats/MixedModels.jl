struct FeProfile{T<:AbstractFloat}  # derived model with the j'th fixed-effects coefficient held constant
    m::LinearMixedModel{T}          # copy of original model after removing the j'th column from X
    y₀::Vector{T}                   # original response vector
    xⱼ::Vector{T}                   # the column that was removed from X
    j::Integer
end

"""
    Base.copy(ReMat{T,S})

Return a shallow copy of ReMat.

A shallow copy shares as much internal storage as possible with the original ReMat.
Only the vector `λ` and the `scratch` matrix are copied.
"""
function Base.copy(ret::ReMat{T,S}) where {T,S}
    return ReMat{T,S}(ret.trm,
        ret.refs,
        ret.levels,
        ret.cnames,
        ret.z,
        ret.wtz,
        copy(ret.λ),
        ret.inds,
        ret.adjA,
        copy(ret.scratch))
end

## FIXME: also create a shallow copy of a LinearMixedModel object that performs a shallow copy of the reterms and the optsum.
##   Probably don't bother to copy the components of L as we will always assume that an updateL! call precedes a call to
##   objective.

"""
    _copy_away_from_lowerbd!(sink, source, bd; incr=0.01)

Replace `sink[i]` by `max(source[i], bd[i] + incr)`.  When `bd[i] == -Inf` this simply copies `source[i]`.
"""
function _copy_away_from_lowerbd!(sink, source, bd; incr=0.01)
    for i in eachindex(sink, source, bd)
        @inbounds sink[i] = max(source[i], bd[i] + incr)
    end
    return sink
end

function σvals(m::LinearMixedModel{T}) where {T}
    @compat (; σ, reterms) = m
    isone(length(reterms)) && return σvals(only(reterms), σ)
    return (collect(Iterators.flatten(σvals.(reterms, σ)))...,)
end

function FeProfile(m::LinearMixedModel, j::Integer)
    Xy = m.Xymat.xy
    xcols = collect(axes(Xy, 2))
    ycol = pop!(xcols)
    notj = deleteat!(xcols, j) # indirectly check that j ∈ xcols
    y₀ = Xy[:, ycol]
    xⱼ = Xy[:, j]
    feterm = FeTerm(Xy[:, notj], m.feterm.cnames[notj])
    reterms = [copy(ret) for ret in m.reterms]
    mnew = fit!(LinearMixedModel(y₀ - xⱼ * m.β[j], feterm, reterms, m.formula); progress=false)
       # not sure this next call makes sense - should the second argument be m.optsum.final?
    _copy_away_from_lowerbd!(mnew.optsum.initial, mnew.optsum.final, mnew.lowerbd; incr=0.05)
    return FeProfile(mnew, y₀, xⱼ, j)
end

function betaprofile!(pr::FeProfile{T}, val, βⱼ::T, j::Integer, obj::T) where {T}
    prm = pr.m
    refit!(prm, mul!(copyto!(pr.m.y, pr.y₀), pr.xⱼ, βⱼ, -1, 1); progress=false)
    ζ = sqrt(prm.objective - obj)
    push!(val.ζ, ζ)
    push!(val.θ, (prm.θ...,))
    push!(val.σ, (prm.σ, σvals(prm)...))
    β = prm.β
    splice!(β, j:(j - 1), βⱼ)
    push!(val.β, (β...,)) # FIXME: do this in a loop rather than splice! followed by splatting
    return ζ
end

function profileβj(m::LinearMixedModel{T}, j::Integer; threshold=4) where {T}
    @compat (; β, θ, σ, stderror, objective) = m
    val = (; ζ = -T[0], β = [(β...,)], σ = [(σ, σvals(m)...)], θ = [(θ...,)])
    prj = FeProfile(m, j)
    st = stderror[j] * 0.5
    bb = β[j] - st
    while true
        betaprofile!(prj, val, bb, j, objective) < threshold || break
        bb -= st
    end
    rmul!(val.ζ, -1)
    bb = β[j] + st
    while true
        betaprofile!(prj, val, bb, j, objective) < threshold || break
        bb += st
    end
    m = length(val.ζ)
    return sort!(Table(merge((; p = fill(:β, m), j = fill(Int8(j), m)), val)), by = r -> r.ζ)
end

struct MixedModelProfile
    m::LinearMixedModel   # Model that has been profiled
    tbl::Table            # Table containing ζ, σ, β, and θ from each conditional fit
    fwd::Dict{Symbol}     # Interpolation splines for ζ as a function of a parameter
    rev::Dict{Symbol}     # Interpolation splines for a parameter as a function of ζ
end

"""
    profile(m::LinearMixedModel)

Return a `MixedModelProfile` for the objective of `m` with respect to the fixed-effects coefficients.
"""
function profile(m::LinearMixedModel{T}; threshold = 4) where {T}
    final = copy(refit!(m).optsum.final)
    ord, nat = BSplineOrder(4), Natural()
    tbl = profileσ(m; threshold)
    rev = Dict(:σ => interpolate(tbl.ζ, first.(tbl.σ), ord, nat))
    fwd = Dict(:σ => interpolate(first.(tbl.σ), tbl.ζ, ord, nat))
    updateL!(setθ!(m, final))
    for j in axes(m.β, 1)
        prbj = profileβj(m, j; threshold)
        betaj = getindex.(prbj.β, j)
        rev[Symbol("β$j")] = interpolate(prbj.ζ, betaj, ord, nat)
        fwd[Symbol("β$j")] = interpolate(betaj, prbj.ζ, ord, nat)
        append!(tbl, prbj)
    end
    updateL!(setθ!(m, final))
    copyto!(m.optsum.final, final)
    m.optsum.fmin = objective(m)
    for j in axes(final, 1)
        prbj = profileθj(m, j; threshold)
        thetaj = getindex.(prbj.θ, j)
        rev[Symbol("θ$j")] = interpolate(prbj.ζ, thetaj, ord, nat)
        fwd[Symbol("θ$j")] = interpolate(thetaj, prbj.ζ, ord, nat)
        append!(tbl, prbj)
    end
    return MixedModelProfile(m, tbl, fwd, rev)
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

function refitσ!(m::LinearMixedModel{T}, σ, val::NamedTuple, obj::T, neg::Bool) where {T}
    m.optsum.sigma = σ
    refit!(m; progress=false)
    push!(val.ζ, (neg ? -one(T) : one(T)) * sqrt(m.objective - obj))
    push!(val.σ, (σ, σvals(m)...))
    push!(val.β, (m.β...,))
    push!(val.θ, (m.θ...,))
    return m.objective
end

"""
    _facsz(m, σ, objective)

Return a factor so that refitting `m` with `σ` at its current value times this factor gives `ζ ≈ 0.5`
"""
function _facsz(m::LinearMixedModel, σ, objective)
    i64 = inv(64)
    m.optsum.sigma = σ * exp(i64)
    return exp(i64 / (2*sqrt(refit!(m; progress=false).objective - objective)))
end

"""
    profileσ(m; threshold=4)

Return a Table of the profile of `σ` for model `m`.  The profile extends to where the magnitude of ζ exceeds `threshold`.
"""
function profileσ(m::LinearMixedModel{T}; threshold = 4) where {T}
    isnothing(m.optsum.sigma) ||
        throw(ArgumentError("Can't profile σ, which is fixed at $(m.optsum.sigma)"))
    @compat (; β, σ, θ, objective, optsum) = m
    θinitial = copy(optsum.initial)
    _copy_away_from_lowerbd!(optsum.initial, optsum.final, optsum.lowerbd)
    val = (; ζ = T[0], β = [(β...,)], σ = [(σ, σvals(m)...)], θ = [(θ...,)])
    facsz = _facsz(m, σ, objective)
    σv = σ / facsz
    while true
        refitσ!(m, σv, val, objective, true)
        last(val.ζ) > -threshold || break
        σv /= facsz
    end
    σv = σ * facsz
    while true
        refitσ!(m, σv, val, objective, false)
        last(val.ζ) < threshold || break
        σv *= facsz
    end
    optsum.sigma = nothing
    optsum.initial = θinitial
    updateL!(setθ!(m, θ))
    m = length(val.ζ)
    return sort!(Table(merge((; p = fill(:σ, m), j = fill(zero(Int8), m)), val)); by = r -> r.σ)
end

"""
    optsumj(os::OptSummary, j::Integer)

Return an `OptSummary` with the `j`'th component of the parameter omitted.

`os.final` with its j'th component omitted is used as the initial parameter.
""" 
function optsumj(os::OptSummary, j::Integer)
    return OptSummary(
        deleteat!(copy(os.final), j),
        deleteat!(copy(os.lowerbd), j),
        os.optimizer
    )
end

function profileobjθj(m::LinearMixedModel{T}, θ::AbstractVector{T}, opt::Opt, osj::OptSummary) where {T}
    isone(length(θ)) && return objective(updateL!(setθ!(m, θ)))
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(osj.final, osj.initial))
    _check_nlopt_return(ret)
    return fmin
end

function profileθj(m::LinearMixedModel{T}, j::Integer; threshold=4) where {T}
    @compat (; β, σ, optsum) = m
    @compat (; final, fmin, lowerbd) = optsum
    updateL!(setθ!(m, final))
    θ = copy(final)
    θj = final[j]
    lbj = lowerbd[j]
    notj = deleteat!(collect(axes(final, 1)), j)
    osj = optsumj(optsum, j)
    opt = Opt(osj)
    function obj(x, g)
        isempty(g) || throw(ArgumentError("gradients are not evaluated by this objective"))
        for i in eachindex(notj, x)
            @inbounds θ[notj[i]] = x[i]
        end
        return objective(updateL!(setθ!(m, θ)))
    end
    NLopt.min_objective!(opt, obj)
    val = (; ζ = [zero(T)], β = [(β...,)], σ = [(σ, σvals(m)...)], θ = [(θ...,)])
    ζold = zero(T)
    δj = inv(T(64))
    while (abs(ζold) < threshold) && length(val.ζ) < 100  # increasing values of θ[j]
        θ[j] += δj
        ζ = sign(θ[j] - θj) * sqrt(profileobjθj(m, θ, opt, osj) - fmin)
        push!(val.ζ, ζ)
        push!(val.β, (m.β...,))
        push!(val.σ, (m.σ, σvals(m)...))
        push!(val.θ, (θ...,))
        δj /= 2(ζ - ζold)
        ζold = ζ
    end
    copyto!(θ, final)
    δj = -inv(T(32))
    ζold = zero(T)
    while (abs(ζold) < threshold) && (length(val.ζ) < 120) && (θ[j] > lbj)
        θ[j] += δj
        θ[j] = max(lbj, θ[j])
        ζ = sign(θ[j] - θj) * sqrt(profileobjθj(m, θ, opt, osj) - fmin)
        push!(val.ζ, ζ)
        push!(val.β, (m.β...,))
        push!(val.σ, (m.σ, σvals(m)...))
        push!(val.θ, (θ...,))
        δj /= (2 * abs(ζ - ζold))
        ζold = ζ
    end
    updateL!(setθ!(m, final))
    m = length(val.ζ)
    return sort!(Table(merge((; p = fill(:θ, m), j = fill(Int8(j), m)), val)); by = r -> r.ζ)
end

function profilevc(m::LinearMixedModel{T}, val, rowj::AbstractVector{T}) where {T}
    optsum = m.optsum
    sigma() = val / norm(rowj)   # a function to be re-evaluated when values in rowj change
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g must be empty"))
        updateL!(setθ!(m, x))
        optsum.sigma = sigma()
        return objective(m)
    end
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    _check_nlopt_return(ret)
    return fmin
end

function profileσs(m::LinearMixedModel{T}; threshold::Number=4) where {T}
    @compat (; λ, σ, β, optsum, parmap, reterms) = m
    isnothing(optsum.sigma) || throw(ArgumentError("Can't profile vc's when σ is fixed"))
    @compat (; initial, final, fmin) = optsum
    saveinitial = copy(initial)
    copyto!(initial, final)
    val = (;
        ζ = T[0],
        β = [(β...,)],
        σ = [(σ, σvals(m)...)],
        θ = [(final...,)],
    )
    for t in reterms
        for r in eachrow(t.λ.data)
            σij = σ * norm(r)
            δ = (iszero(σij) ? one(T) : σij) * T(inv(32))
            obj = profilevc(m, σij + δ, r)
            push!(val.ζ, sqrt(obj - fmin))
            push!(val.β, (m.β...,))
            push!(val.σ, (m.σ, σvals(m)...))
            push!(val.θ, (final...,))
            stepsz = δ / (2 * last(val.ζ))
            @info σij, δ, obj, last(val.ζ), stepsz
        end
    end
    copyto!(final, initial)
    copyto!(initial, saveinitial)
    optsum.sigma = nothing
    updateL!(setθ!(m, final))
    return Table(val)
end
