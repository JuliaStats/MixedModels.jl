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

function FeProfile(m::LinearMixedModel, j::Integer)
    Xy = m.Xymat.xy
    xcols = collect(axes(Xy, 2))
    ycol = pop!(xcols)
    notj = deleteat!(xcols, j) # indirectly check that j ∈ xcols
    y₀ = Xy[:, ycol]
    xⱼ = Xy[:, j]
    feterm = FeTerm(Xy[:, notj], m.feterm.cnames[notj])
    reterms = [copy(ret) for ret in m.reterms]
    m = fit!(LinearMixedModel(y₀ - xⱼ * m.β[j], feterm, reterms, m.formula); progress=false)
    @. m.optsum.initial = max(m.optsum.initial, m.lowerbd + 0.05)
    return FeProfile(m, y₀, xⱼ, j)
end

function betaprofile!(pr::FeProfile{T}, val, βⱼ::T, j::Integer, obj::T) where {T}
    prm = pr.m
    refit!(prm, mul!(copyto!(pr.m.y, pr.y₀), pr.xⱼ, βⱼ, -1, 1); progress=false)
    ζ = sqrt(prm.objective - obj)
    push!(val.ζ, ζ)
    push!(val.θ, prm.θ)
    push!(val.σ, prm.σ)
    β = prm.β
    splice!(β, j:(j - 1), βⱼ)
    push!(val.β, β)
    return ζ
end

function profileβj(m::LinearMixedModel{T}, j::Integer; threshold=4) where {T}
    @compat (; β, θ, σ, stderror, objective) = m
    val = (; ζ = -T[0], β = [SVector{length(β)}(β)], σ = [σ], θ = [SVector{length(θ)}(θ)])
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
    return sort!(Table(merge((; par = fill(Symbol("β$j"), length(val.σ))), val)), by = r -> r.ζ)
end

struct MixedModelProfile
    tbl::Table                       # Table containing ζ, σ, β, and θ from each conditional fit
    fecnames::Vector{String}         # Fixed-effects coefficient names
    facnames::Vector{Symbol}         # Names of grouping factors
    recnames::Vector{Vector{String}} # Vector of vectors of column names for random effects
    parmap::Vector{NTuple{3,Int}}    # parmap from the model (used to construct λ from θ)
    fwd::Dict{Symbol}                # Interpolation splines for ζ as a function of β
    rev::Dict{Symbol}                # Interpolation splines for β as a function of ζ
end

"""
    profile(m::LinearMixedModel)

Return a `MixedModelProfile` for the objective of `m` with respect to the fixed-effects coefficients.
"""
function profile(m::LinearMixedModel{T}; threshold = 4) where {T}
    ord, nat = BSplineOrder(4), Natural()
    tbl = profileσ(m; threshold)
    rev = Dict(:σ => interpolate(tbl.ζ, tbl.σ, ord, nat))
    fwd = Dict(:σ => interpolate(tbl.σ, tbl.ζ, ord, nat))
    for j in axes(m.β, 1)
        prbj = profileβj(m, j; threshold)
        betaj = getindex.(prbj.β, j)
        rev[Symbol("β$j")] = interpolate(prbj.ζ, betaj, ord, nat)
        fwd[Symbol("β$j")] = interpolate(betaj, prbj.ζ, ord, nat)
        append!(tbl, prbj)
    end
    return MixedModelProfile(
        tbl,
        m.feterm.cnames,
        fname.(m.reterms),
        getproperty.(m.reterms, :cnames),
        m.parmap,
        fwd,
        rev,
    )
end

function StatsBase.confint(pr::MixedModelProfile; level::Real=0.95)
    cutoff = sqrt.(quantile(Chisq(1), level))
    syms = circshift(unique(pr.tbl.par), -1)   # put σ last
    rev = pr.rev
    return DictTable(; 
        coef=syms,
        lower=[rev[s](-cutoff) for s in syms],
        upper=[rev[s](cutoff) for s in syms],
    )
end

function refitσ!(m::LinearMixedModel{T}, val::NamedTuple, obj::T, neg::Bool) where {T}
    m.optsum.sigma = last(val.σ)
    refit!(m; progress=false)
    push!(val.ζ, (neg ? -one(T) : one(T)) * sqrt(m.objective - obj))
    push!(val.β, m.β)
    push!(val.θ, m.θ)
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
    profileσ(m; threshold=3)

Return a Table of the profile of `σ` for model `m`.  The profile extends to where the magnitude of ζ exceeds `threshold`.
"""
function profileσ(m::LinearMixedModel{T}; threshold = 4) where {T}
    isnothing(m.optsum.sigma) ||
        throw(ArgumentError("Can't profile σ, which is fixed at $(m.optsum.sigma)"))
    @compat (; β, σ, θ, objective, optsum) = m
    θinitial = copy(optsum.initial)
        # copy optsum.final to optsum.initial with elements on the boundary set to 0.01
    map!((b, x) -> iszero(b) ? max(0.01, x) : x, optsum.initial, optsum.lowerbd, optsum.final)
    val = (; ζ = T[0], β = [SVector{length(β)}(β)], σ = [σ], θ = [SVector{length(θ)}(θ)])
    facsz = _facsz(m, σ, objective)
    push!(val.σ, σ / facsz)
    while true
        refitσ!(m, val, objective, true)
        last(val.ζ) > -threshold || break
        push!(val.σ, last(val.σ) / facsz)
    end
    push!(val.σ, σ * facsz)
    while true
        refitσ!(m, val, objective, false)
        last(val.ζ) < threshold || break
        push!(val.σ, last(val.σ) * facsz)
    end
    optsum.sigma = nothing
    optsum.initial = θinitial
    updateL!(setθ!(m, θ))
    return sort!(Table(merge((; par = fill(:σ, length(val.ζ))), val)); by = r -> r.σ)
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

function profileobjθj(m::LinearMixedModel{T}, j::Integer, θj::T) where {T}
    @compat (; θ, optsum) = m
    θ[j] = θj
    isone(length(θ)) && return objective(updateL!(setθ!(m, θ))), T[]
    inds = deleteat!(collect(axes(θ, 1)), j)
    function obj(x, g)
        isempty(g) || throw(ArgumentError("gradients are not evaluated by this objective"))
        for (i, xi) in zip(inds, x)
            θ[i] = xi
        end
        return objective(updateL!(setθ!(m, θ)))
    end
    osj = optsumj(optsum, j)
    opt = Opt(osj)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(osj.final, osj.initial))
    _check_nlopt_return(ret)
    return fmin, xmin
end

"""
    _stepszθj(m, j, θj, objective)

Return a step size for θ[j] to give `ζ ≈ 0.5`
"""
function _stepszθj(m::LinearMixedModel, j::Integer, θj, objective)
    step = min(inv(64), θj)
    return step / (2*sqrt(profileobjθj(m, j, θj + step) - objective))
end


function profileθj(m::LinearMixedModel{T}, j::Integer) where {T}
    @compat (; β, σ, θ, objective, optsum) = m
    θorig = copy(θ)
    θopt = θ[j]
    θjvals = if iszero(optsum.lowerbd[j])
        iszero(θopt) ? range(zero(T), T(2), 33) : range(zero(T), 3θopt, 33)
    else
        aθopt = abs(θopt)
        range(-5*aθopt, 5*aθopt, 33)
    end
    nrow = size(θjvals, 1)   # should always be 33 at present
    val = Table(
        (;
            par = repeat([Symbol("θ$j")], nrow),
            ζ = collect(θjvals),
            β = repeat([SVector{size(β, 1)}(β)], nrow),
            σ = similar(θjvals),
            θ = repeat([SVector{size(θ, 1)}(θ)], nrow),
        ),
    )
    @inbounds for (i, th) in enumerate(θjvals)
        if th ≈ θopt
            val.ζ[i] = zero(T)
            val.σ[i] = σ
            val.β[i] = β
            val.θ[i] = θ
        else
            obj, xmin = profileobjθj(m, j, th)
            val.ζ[i] = sign(th - θopt) * sqrt(obj - objective)
            val.σ[i] = m.σ
            val.β[i] = m.β
            splice!(xmin, j:(j-1), th)
            val.θ[i] = xmin
        end
    end
    updateL!(setθ!(m, θorig))
    return val
end

function profilevcij(m::LinearMixedModel{T}, opt::Opt, val::T, i, j) where {T}
        # Right now just punt on fixed sigma. This can probably be patched later
    @compat (; optsum, reterms, θ) = m
    isnothing(optsum.sigma) || throw(ArgumentError("Cannot profile vc on model with fixed σ"))
    rowj = view(reterms[i].λ.data, j, 1:j)
    sigma() = val / norm(rowj)
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g must be empty"))
        updateL!(setθ!(m, x))
        optsum.sigma = sigma()
        return objective(m)
    end
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    _check_nlopt_return(ret)
    optsum.sigma = nothing
    copyto!(optsum.final, θ)
    return fmin
end

function profileσs(m::LinearMixedModel{T}) where {T}
    isnothing(m.optsum.sigma) || throw(ArgumentError("Can't profile vc's when σ is fixed"))
    @compat (; λ, θ, σ, objective, optsum, parmap) = m
    opt = Opt(optsum)
    for (i, j, k) in parmap
        if j == k
            rowj = view(λ[i].data, j, 1:j)
            σij = σ * norm(rowj)
            δ = (iszero(σij) ? one(T) : σij) * T(inv(32))
            obj = profilevcij(m, opt, σij + δ, i, j)
            stepsz = (obj - objective) / 2δ
            @info i, j, k, σij, δ, obj, stepsz
        end
    end
    updateL!(setθ!(m, θ))
end
