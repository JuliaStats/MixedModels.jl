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
    refit!(m; progress=false, warm_start=true)
    return mkrow!(tc, m, _ζ(m.objective, obj, neg, :σ, σ))
end

"""
    _facsz(m, σ, objective)

Return a factor such that refitting `m` with `σ` at its current value times this factor gives `ζ ≈ 0.5`
"""
function _facsz(m::LinearMixedModel{T}, σ::T, obj::T) where {T}
    i64 = T(inv(64))
    expi64 = exp(i64)     # help the compiler infer it is a constant
    σv = σ * expi64
    m.optsum.sigma = σv
    ζ = _ζ(refit!(m; progress=false, warm_start=true).objective, obj, false, :σ, σv)
    iszero(ζ) && throw(
        ArgumentError(
            "cannot determine the step size for profiling σ: " *
            "the objective does not change between σ = $(σ) and σ = $(σv)",
        ),
    )
    return exp(i64 / (2 * ζ))
end

"""
    profileσ(m::LinearMixedModel, tc::TableColumns; threshold=4)

Return a Table of the profile of `σ` for model `m`.  The profile extends to where the magnitude of ζ exceeds `threshold`.

!!! note
    This method is called by `profile` and currently considered internal.
    As such, it may change or disappear in a future release without being considered breaking.
"""
function profileσ(m::LinearMixedModel{T}, tc::TableColumns{T}; threshold=4) where {T}
    optsum = m.optsum
    isnothing(optsum.sigma) ||
        throw(ArgumentError("Can't profile σ, which is fixed at $(optsum.sigma)"))
    θ = copy(optsum.final)
    θinitial = copy(optsum.initial)   # overwritten by the warm starts in refit!
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
    copyto!(optsum.final, θ)   # warm start the increasing values of σ from θ̂
    σv = σ * facsz
    while true
        newrow = merge(pnm, refitσ!(m, σv, tc, obj, false))
        push!(tbl, newrow)
        newrow.ζ < threshold || break
        σv *= facsz
    end
    optsum.sigma = nothing
    optsum.initial = θinitial
    copyto!(optsum.final, θ)
    updateL!(setθ!(m, θ))
    σv = [r.σ for r in tbl]
    ζv = [r.ζ for r in tbl]
    local fwd, rev
    try
        fwd = Dict(:σ => interpolate(σv, ζv, BSplineOrder(4), Natural()))
        rev = Dict(:σ => interpolate(ζv, σv, BSplineOrder(4), Natural()))
    catch
        @error "An error occurred while fitting the profile splines for σ. Try adjusting the threshold."
        rethrow()
    end
    return (; m, tbl, fwd, rev)
end
