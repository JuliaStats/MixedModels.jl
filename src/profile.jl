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
    j ∈ xcols || throw(ArgumentError("j = $j must be in $xcols"))
    y₀ = Xy[:, ycol]
    xⱼ = Xy[:, j]
    notj = deleteat!(xcols, j)   # indirectly checks range of j
    feterm = FeTerm(Xy[:, notj], m.feterm.cnames[notj])
    reterms = [copy(ret) for ret in m.reterms]
    m = fit!(LinearMixedModel(y₀ - xⱼ * m.β[j], feterm, reterms, m.formula))
    @. m.optsum.initial = max(m.optsum.initial, m.lowerbd + 0.05)
    return FeProfile(m, y₀, xⱼ, j)
end

function refit!(pr::FeProfile{T}, βⱼ) where {T}
    return refit!(pr.m, mul!(copyto!(pr.m.y, pr.y₀), pr.xⱼ, βⱼ, -1, 1); progress=false)
end

function profileβ(m::LinearMixedModel{T}, steps=-5:5) where {T}
    β, θ, σ, std, obj = m.β, m.θ, m.σ, m.stderror, objective(m)
    k = length(θ)
    p = length(β)
    prlen = length(steps) * p
    i = sizehint!((p ≤ typemax(Int8) ? Int8 : Int16)[], prlen)
    zeta = sizehint!(T[], prlen)
    sigma = sizehint!(T[], prlen)
    beta = sizehint!(SVector{p,T}[], prlen)
    theta = sizehint!(SVector{k,T}[], prlen)
    for j in eachindex(β)
        prj = FeProfile(m, j)
        prm = prj.m
        estj, stdj = β[j], std[j]
        for s in steps
            push!(i, j)
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
    return Table(i = i, ζ = zeta, σ = sigma, β = beta, θ = theta)
end
