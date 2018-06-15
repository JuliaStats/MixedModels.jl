agqvals(m::GeneralizedLinearMixedModel, maxk=31) = [deviance(m,k) for k in 1:2:maxk]

function unscaledcond(m::GeneralizedLinearMixedModel{T}, zv::AbstractVector{T}) where T
    length(m.u[1]) == length(m.devc) || throw(ArgumentError("m must have a single scalar random-effect term"))
    u = vec(m.u[1])
    u₀ = vec(m.u₀[1])
    Compat.copyto!(u₀, u)
    ra = RaggedArray(m.resp.devresid, m.LMM.trms[1].f.refs)
    devc0 = sum!(broadcast!(abs2, m.devc0, u), ra)  # the deviance components at z = 0
    sd = broadcast!(inv, m.sd, m.LMM.L.data[Block(1,1)].diag)
    devc = m.devc
    res = zeros(T, (length(devc), length(zv)))
    for (j, z) in enumerate(zv)
        u .= u₀ .+ z .* sd
        updateη!(m)
        Compat.copyto!(view(res, :, j), -(sum!(broadcast!(abs2, devc, u), ra) .- devc0) ./ 2)
    end
    Compat.copyto!(u, u₀)
    updateη!(m)
    zv, res'
end

function ugrid(m::GeneralizedLinearMixedModel{T}, uv::AbstractVector{T}) where T
    length(m.u[1]) == length(m.devc) || throw(ArgumentError("m must have a single scalar random-effect term"))
    u = vec(m.u[1])
    u₀ = Compat.copyto!(vec(m.u₀[1]), u)
    ra = RaggedArray(m.resp.devresid, m.LMM.trms[1].f.refs)
    res = zeros(T, (length(u), length(uv)))
    devc = m.devc
    for (j, uu) in enumerate(uv)
        fill!(u, uu)
        updateη!(m)
        Compat.copyto!(view(res, :, j), sum!(fill!(devc, abs2.(uu)), ra))
    end
    Compat.copyto!(u, u₀)
    updateη!(m)
    uv, res'
end