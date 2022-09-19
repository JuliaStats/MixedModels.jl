struct FeProfile{T<:AbstractFloat}
    m::LinearMixedModel{T}
    y₀::Vector{T}
    xⱼ::Vector{T}
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
    prtbl::Table
    δ::AbstractVector{T}
    fecnames::Vector{String}
    facnames::Vector{Symbol}
    recnames::Vector{Vector{String}}
    parmap::Vector{NTuple{3,Int}}
end

function profileβ(m::LinearMixedModel{T}, δ=(-8:8) / 2) where {T}
    β, θ, σ, std, obj = m.β, m.θ, m.σ, m.stderror, objective(m)
    zsz = length(δ) * length(β)
    zeta = sizehint!(T[], zsz)
    sigma = sizehint!(T[], zsz)
    beta = sizehint!(SVector{length(β),T}[], zsz)
    theta = sizehint!(SVector{length(θ),T}[], zsz)
    @inbounds for j in only(axes(β))
        prj = FeProfile(m, j)
        prm = prj.m
        estj, stdj = β[j], std[j]
        for (i, s) in enumerate(δ)
            if iszero(s)
                push!(zeta, zero(T))
                push!(sigma, σ)
                push!(beta, β)
                push!(theta, θ)
            else
                βj = estj + s * stdj
                refit!(prj, βj)
                βcopy = prm.β
                splice!(βcopy, j:j-1, βj)
                push!(beta, βcopy)
                push!(zeta, sign(s) * sqrt(prm.objective - obj))
                push!(sigma, prm.σ)
                push!(theta, prm.θ)
            end
        end
    end
    updateL!(setθ!(m, θ))
    return MixedModelProfile(
        Table(ζ = zeta, σ = sigma, β = beta, θ = theta),
        δ,
        copy(coefnames(m)),
        [fname(t) for t in m.reterms],
        [t.cnames for t in m.reterms],
        copy(m.parmap),
    )
end

function StatsBase.confint(pr::MixedModelProfile; level=0.95)
    cutoff = sqrt.(quantile(Chisq(1), level))
    (; prtbl, fecnames) = pr
    p = length(fecnames)
    zetamat = reshape(prtbl.ζ, length(pr.δ), p)
    betamat = reshape(prtbl.β, size(zetamat))
    lower = sizehint!(similar(zetamat, 0), p)
    upper = sizehint!(similar(zetamat, 0), p)
    for j in axes(zetamat, 2)
        invspl = interpolate(view(zetamat, :, j), getindex.(view(betamat, :, j), j), BSplineOrder(4))
        push!(lower, invspl(-cutoff))
        push!(upper, invspl(cutoff))
    end
    return DictTable(coef = fecnames, lower = lower, upper = upper)
end
