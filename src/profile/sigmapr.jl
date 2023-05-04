"""
    refitσ!(m::LinearMixedModel{T}, σ::T, tc::TableColumns{T}, obj::T, neg::Bool)

Refit the model `m` with the given value of `σ` and return a NamedTuple of information about the fit.

`obj` and `neg` allow for conversion of the objective to the `ζ` scale and `tc` is used to return a NamedTuple

!!! note
    This method is internal and may change or disappear in a future release
    without being considered breaking.
"""
function refitσ!(
    m::LinearMixedModel{T}, σ, tc::TableColumns{T}, obj::T, neg::Bool
) where {T}
    m.optsum.sigma = σ
    refit!(m; progress=false)
    return mkrow!(tc, m, (neg ? -one(T) : one(T)) * sqrt(m.objective - obj))
end

"""
    _facsz(m, σ, objective)

Return a factor such that refitting `m` with `σ` at its current value times this factor gives `ζ ≈ 0.5`
"""
function _facsz(m::LinearMixedModel{T}, σ::T, obj::T) where {T}
    i64 = T(inv(64))
    expi64 = exp(i64)     # help the compiler infer it is a constant
    m.optsum.sigma = σ * expi64
    return exp(i64 / (2 * sqrt(refit!(m; progress=false).objective - obj)))
end

"""
    profileσ(m::LinearMixedModel, tc::TableColumns; threshold=4)

Return a Table of the profile of `σ` for model `m`.  The profile extends to where the magnitude of ζ exceeds `threshold`.
!!! note
    This method is called by `profile` and currently considered internal.
    As such, it may change or disappear in a future release without being considered breaking.
"""
function profileσ(m::LinearMixedModel{T}, tc::TableColumns{T}; threshold=4) where {T}
    (; σ, optsum) = m
    isnothing(optsum.sigma) ||
        throw(ArgumentError("Can't profile σ, which is fixed at $(optsum.sigma)"))
    θ = copy(optsum.final)
    θinitial = copy(optsum.initial)
    _copy_away_from_lowerbd!(optsum.initial, optsum.final, optsum.lowerbd)
    obj = optsum.fmin
    σ = m.σ
    pnm = (p=:σ,)
    tbl = [merge(pnm, mkrow!(tc, m, zero(T)))]
    facsz = _facsz(m, σ, obj)
    σv = σ / facsz
    while true
        newrow = merge(pnm, refitσ!(m, σv, tc, obj, true))
        push!(tbl, newrow)
        newrow.ζ > -threshold || break
        σv /= facsz
    end
    reverse!(tbl)
    σv = σ * facsz
    while true
        newrow = merge(pnm, refitσ!(m, σv, tc, obj, false))
        push!(tbl, newrow)
        newrow.ζ < threshold || break
        σv *= facsz
    end
    optsum.sigma = nothing
    optsum.initial = θinitial
    updateL!(setθ!(m, θ))
    σv = [r.σ for r in tbl]
    ζv = [r.ζ for r in tbl]
    fwd = Dict(:σ => interpolate(σv, ζv, BSplineOrder(4), Natural()))
    rev = Dict(:σ => interpolate(ζv, σv, BSplineOrder(4), Natural()))
    return (; m, tbl, fwd, rev)
end
