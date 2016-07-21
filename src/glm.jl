function updateμ!{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T})
    y, dist, link, η, μ = m.y, m.dist, m.link, m.η, m.μ
    wt, wrkresid, wrkwt, dres = m.wt, m.wrkresid, m.wrkwt, m.devresid

    priorwts = !isempty(wt)

    @inbounds Threads.@threads for i = eachindex(η)
        yi, ηi = y[i], η[i]
        μi = μ[i] = linkinv(link, ηi)
        dμdηi = mueta(link, ηi)
        vari = GLM.glmvar(dist, link, μi, ηi)
        wrkresid[i] = (yi - μi)/dμdηi
        wti = priorwts ? wt[i] : one(T)
        dres[i] = devresid(dist, yi, μi, wti)
        wrkwt[i] = wti * abs2(dμdηi) / vari
    end
    m
end

function updateweights{T <: AbstractFloat}(dres::Vector{T}, wrkwt::Vector{T}, wt::Vector{T})
    @inbounds @simd for i in eachindex(wt)
        wti = wt[i]
        dres[i] *= wti
        wrkwt[i] *= wti
    end
end

function updateμ!{T<:AbstractFloat, D<:Union{Bernoulli, Binomial}, L<:LogitLink}(m::GeneralizedLinearMixedModel{T,D,L})
    y, η, μ, wrkres, wrkwt, dres = m.y, m.η, m.μ, m.wrkresid, m.wrkwt, m.devresid

    @inbounds Threads.@threads for i in eachindex(η)
        ηi = clamp(η[i], -20.0, 20.0)
        ei = exp(-ηi)
        opei = 1 + ei
        μi = μ[i] = inv(opei)
        dμdη = wrkwt[i] = ei / abs2(opei)
        yi = y[i]
        wrkres[i] = (yi - μi) / dμdη
        dres[i] = -2 * (yi == 1 ? log(μi) : yi == 0 ? log1p(-μi) :
            (yi * (log(μi) - log(yi)) + (1 - yi) * (log1p(-μi) - log1p(-yi))))
    end

    if !isempty(m.wt)
        updateweights(dres, wrkwt, m.wt)
    end
end

function updateμ!{T<:AbstractFloat, D<:Poisson, L<:LogLink}(m::GeneralizedLinearMixedModel{T,D,L})
    y, η, μ, wrkres, wrkwt, dres = m.y, m.η, m.μ, m.wrkresid, m.wrkwt, m.devresid

    @inbounds Threads.@threads for i in eachindex(η)
        ηi = η[i]
        μi = μ[i] = exp(ηi)
        dμdη = wrkwt[i] = μi
        yi = y[i]
        wrkres[i] = (yi - μi) / dμdη
        dres[i] = 2 * (StatsFuns.xlogy(yi, yi / μi) - (yi - μi))
    end

    if !isempty(m.wt)
        updateweights(dres, wrkwt, m.wt)
    end
end

function wrkresp!{T <: AbstractFloat}(v::DenseVecOrMat{T}, m::GeneralizedLinearMixedModel{T})
    if isempty(m.offset)
        return broadcast!(+, v, m.η, m.wrkresid)
    end
    η, offset, wrkresid = m.η, m.offset, m.wrkresid
    @inbounds @simd for i in eachindex(η)
        v[i] = wrkresid[i] + η[i] - offset[i]
    end
    v
end
