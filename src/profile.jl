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
        copy(ret.scratch)) # likely don't need to copy scratch, as it is temporary storage.
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
    push!(val.ζ, (neg ? -1. : 1) * sqrt(m.objective - obj))
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
    @compat (; β, σ, θ, objective) = m
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
    m.optsum.sigma = nothing
    refit!(m)
    return sort!(Table(merge((; par = fill(:σ, length(val.ζ))), val)); by = r -> r.ζ)
end
