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
function Base.copy(ret::ReMat{T,S}) where {T, S}
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

function refit!(pr::FeProfile{T}, βⱼ) where {T}
    return refit!(pr.m, mul!(copyto!(pr.m.y, pr.y₀), pr.xⱼ, βⱼ, -1, 1); progress=false)
end

struct MixedModelProfile{T}
    prtbl::Table          # Table containing ζ, σ, β, and θ from each conditional fit
    δ::AbstractVector{T}  # values of fixed coefficient are `β[j] .+ δ .* stderror[j]`
    fecnames::Vector{String}  # Fixed-effects coefficient names
    facnames::Vector{Symbol}  # Names of grouping factors
    recnames::Vector{Vector{String}} # Vector of vectors of column names for random effects
    parmap::Vector{NTuple{3,Int}} # parmap from the model (used to construct λ from θ)
    fwd::Vector           # Interpolation splines for ζ as a function of β
    rev::Vector           # Interpolation splines for β as a function of ζ
end

"""
    profileβ(m::LinearMixedModel, δ=(-8:8) / 2)

Return a `MixedModelProfile` for the objective of `m` with respect to the fixed-effects coefficients.

`δ` is a vector of standardized steps at which values of each coefficient are fixed.
When β[i] is being profiled the values are fixed at `m.β[i] .+ δ .* m.stderror[i]`.
""" 
function profileβ(m::LinearMixedModel{T}, δ=(-8:8) / 2) where {T}
    (; β, θ, σ, stderror, objective) = m
    betamat = (stderror * δ' .+ β)'
    zsz = length(betamat)
    zeta = sizehint!(T[], zsz)
    sigma = sizehint!(T[], zsz)
    beta = sizehint!(SVector{length(β),T}[], zsz)
    theta = sizehint!(SVector{length(θ),T}[], zsz)
    @inbounds for (j, c) in enumerate(eachcol(betamat))
        prj = FeProfile(m, j)
        prm = prj.m
        j2jm1 = j:j-1
        for βj in c
            dev = βj - β[j]
            if dev ≈ 0
                push!(zeta, zero(T))
                push!(sigma, σ)
                push!(beta, β)
                push!(theta, θ)
            else
                refit!(prj, βj)
                βcopy = prm.β
                splice!(βcopy, j2jm1, βj)
                push!(beta, βcopy)
                push!(zeta, sign(dev) * sqrt(prm.objective - objective))
                push!(sigma, prm.σ)
                push!(theta, prm.θ)
            end
        end
    end
    updateL!(setθ!(m, θ))
    zetamat = reshape(zeta, length(δ), :)
    interp(x, y) = interpolate(x, y, BSplineOrder(4), Natural())
    fwdspl = [interp(b, z) for (b, z) in zip(eachcol(betamat), eachcol(zetamat))]
    revspl = [interp(z, b) for (b, z) in zip(eachcol(betamat), eachcol(zetamat))]
    return MixedModelProfile(
        Table(ζ = zeta, σ = sigma, β = beta, θ = theta),
        δ,
        copy(coefnames(m)),
        [fname(t) for t in m.reterms],
        [t.cnames for t in m.reterms],
        copy(m.parmap),
        fwdspl,
        revspl,
    )
end

function StatsBase.confint(pr::MixedModelProfile; level=0.95)
    cutoff = sqrt.(quantile(Chisq(1), level))
    (; fecnames, rev) = pr
    lower = [s(-cutoff) for s in rev]
    upper = [s(cutoff) for s in rev]
    return DictTable(coef = fecnames, lower = lower, upper = upper)
end

function refitlogσ!(m::LinearMixedModel{T}, stepsz, obj, logσ, zeta, beta, theta) where {T}
    push!(logσ, last(logσ) + stepsz)
    m.optsum.sigma = exp(last(logσ))
    refit!(m)
    push!(zeta, sign(stepsz[]) * sqrt(m.objective - obj))
    push!(beta, m.β)
    push!(theta, m.θ)
    return m.objective
end

function _logσstepsz(m::LinearMixedModel, σ, objective)
    i64 = inv(64)
    m.optsum.sigma = exp(log(σ) + i64)
    return i64 / sqrt(refit!(m).objective - objective)
end

function profilelogσ(m::LinearMixedModel{T}) where {T}
    isnothing(m.optsum.sigma) || throw(ArgumentError("Can't profile σ, which is fixed at $(m.optsum.sigma)"))
    (; β, σ, θ, objective) = m
    logσ = [log(σ)]
    beta = [SVector{length(β)}(β)]
    theta = [SVector{length(θ)}(θ)]
    zeta = [zero(T)]
    stepsz = -_logσstepsz(m, σ, objective)
    while abs(last(zeta)) < 4
        refitlogσ!(m, stepsz, objective, logσ, zeta, beta, theta)
    end
    reverse!(logσ); reverse!(zeta); reverse!(beta); reverse!(theta)
    stepsz = -stepsz
    while abs(last(zeta)) < 4
        refitlogσ!(m, stepsz, objective, logσ, zeta, beta, theta)
    end
    m.optsum.sigma = nothing
    refit!(m)
    return Table(logσ = logσ, ζ = zeta, β = beta, θ = theta)
end
