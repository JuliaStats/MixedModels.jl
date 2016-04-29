function updateμ!{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T})
    y, dist, link, η, μ, dμdη, var = m.y, m.dist, m.link, m.η, m.μ, m.dμdη, m.var
    wt, wrkresid, wrkwt, dres = m.wt, m.wrkresid, m.wrkwt, m.devresid

    if !isempty(m.offset)
        broadcast!(+, η, η, m.offset)
    end
    priorwts = !isempty(wt)

    @inbounds for i = eachindex(η)
        yi, ηi = y[i], η[i]
        μi = μ[i] = linkinv(link, ηi)
        dμdηi = dμdη[i] = mueta(link, ηi)
        vari = var[i] = GLM.glmvar(dist, link, μi, ηi)
        wrkresid[i] = (yi - μi)/dμdηi
        wti = priorwts ? wt[i] : one(T)
        dres[i] = devresid(dist, yi, μi, wti)
        wrkwt[i] = wti * abs2(dμdηi) / vari
    end
    m
end

function updateμ!{T<:AbstractFloat, D<:Bernoulli, L<:LogitLink}(m::GeneralizedLinearMixedModel{T,D,L})
    y, η, μ, wrkres, wrkwt, dres = m.y, m.η, m.μ, m.wrkresid, m.wrkwt, m.devresid

    if !isempty(m.offset)
        off = m.offset
        @inbounds @simd for i in eachindex(off)
            η[i] += off[i]
        end
    end

    @inbounds @simd for i in eachindex(η)
        ηi = η[i]
        ei = exp(-ηi)
        opei = 1 + ei
        μi = μ[i] = inv(opei)
        dμdη = wrkwt[i] = ei / abs2(opei)
        yi = y[i]
        μi = μ[i]
        wrkres[i] = (yi - μi) / dμdη
        dres[i] = -2 * (yi == 1 ? log(μi) : log1p(-μi))
    end

    if !isempty(m.wt)
        wt = m.wt
        @inbounds @simd for i in eachindex(wt)
            wti = wt[i]
            dres[i] *= wti
            wrkwt[i] *= wti
        end
    end
end

function wrkresp!{T <: AbstractFloat}(v::DenseVecOrMat{T}, m::GeneralizedLinearMixedModel)
    if isempty(m.offset)
        return broadcast!(+, v, m.η, m.wrkresid)
    end
    η, offset, wrkresid = m.η, m.offset, m.wrkresid
    for i in eachindex(η)
        v[i] = wrkresid[i] + η[i] - offset[i]
    end
    v
end
