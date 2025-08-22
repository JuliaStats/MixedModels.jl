struct FeProfile{T<:AbstractFloat}  # derived model with the j'th fixed-effects coefficient held constant
    m::LinearMixedModel{T}          # copy of original model after removing the j'th column from X
    tc::TableColumns{T}
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

function FeProfile(m::LinearMixedModel, tc::TableColumns, j::Integer)
    Xy = m.Xymat.xy
    xcols = collect(axes(Xy, 2))
    ycol = pop!(xcols)
    notj = deleteat!(xcols, j) # indirectly check that j ∈ xcols
    y₀ = Xy[:, ycol]
    xⱼ = Xy[:, j]
    feterm = FeTerm(Xy[:, notj], m.feterm.cnames[notj])
    reterms = [copy(ret) for ret in m.reterms]
    mnew = fit!(
        LinearMixedModel(y₀ - xⱼ * m.β[j], feterm, reterms, m.formula); progress=false
    )
    # not sure this next call makes sense - should the second argument be m.optsum.final?
    _copy_away_from_lowerbd!(
        mnew.optsum.initial, mnew.optsum.final, mnew.lowerbd; incr=0.05
    )
    return FeProfile(mnew, tc, y₀, xⱼ, j)
end

function betaprofile!(
    pr::FeProfile{T}, tc::TableColumns{T}, βⱼ::T, j::Integer, obj::T, neg::Bool
) where {T}
    prm = pr.m
    refit!(prm, mul!(copyto!(prm.y, pr.y₀), pr.xⱼ, βⱼ, -1, 1); progress=false)
    (; positions, v) = tc
    v[1] = (-1)^neg * sqrt(prm.objective - obj)
    getθ!(view(v, positions[:θ]), prm)
    v[first(positions[:σ])] = prm.σ
    σvals!(view(v, positions[:σs]), prm)
    β = prm.β
    bpos = 0
    for (i, p) in enumerate(positions[:β])
        v[p] = (i == j) ? βⱼ : β[(bpos += 1)]
    end
    return first(v)
end

function profileβj!(
    val::NamedTuple, tc::TableColumns{T,N}, sym::Symbol; threshold=4
) where {T,N}
    m = val.m
    (; β, θ, σ, stderror, objective) = m
    (; cnames, v) = tc
    pnm = (; p=sym)
    j = parsej(sym)
    prj = FeProfile(m, tc, j)
    st = stderror[j] * 0.5
    bb = β[j] - st
    tbl = [merge(pnm, mkrow!(tc, m, zero(T)))]
    while true
        ζ = betaprofile!(prj, tc, bb, j, objective, true)
        push!(tbl, merge(pnm, NamedTuple{cnames,NTuple{N,T}}((v...,))))
        if abs(ζ) > threshold
            break
        end
        bb -= st
    end
    reverse!(tbl)
    bb = β[j] + st
    while true
        ζ = betaprofile!(prj, tc, bb, j, objective, false)
        push!(tbl, merge(pnm, NamedTuple{cnames,NTuple{N,T}}((v...,))))
        if abs(ζ) > threshold
            break
        end
        bb += st
    end
    append!(val.tbl, tbl)
    ζv = getproperty.(tbl, :ζ)
    βv = getproperty.(tbl, sym)
    val.fwd[sym] = interpolate(βv, ζv, BSplineOrder(4), Natural())
    val.rev[sym] = interpolate(ζv, βv, BSplineOrder(4), Natural())
    return val
end
