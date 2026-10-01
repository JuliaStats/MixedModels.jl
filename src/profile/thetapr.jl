
"""
    optsumj(os::OptSummary, j::Integer)

Return an `OptSummary` with the `j`'th component of the parameter omitted.

`os.final` with its j'th component omitted is used as the initial parameter.
"""
function optsumj(os::OptSummary, j::Integer)
    return OptSummary(
        deleteat!(copy(os.final), j),
        os.optimizer;
        os.backend,
    )
end

"""
    _nextstep(δ, Δζ, target)

Return the next step in a θ profile after a step of `δ` changed ζ by `Δζ`, signed so that
the expected change is positive.

The new step aims for a change of `target` in ζ but grows by at most a factor of 16. If ζ did
not change in the expected direction, which happens when the profile is flat or the
conditional optimization is inexact, the step stays the same.

!!! note
    This method is internal.
"""
function _nextstep(δ::T, Δζ::T, target::T) where {T}
    return Δζ > 0 ? δ * min(target / Δζ, T(16)) : δ
end

function profileobj!(obj,
    m::LinearMixedModel{T}, θ::AbstractVector{T}, osj::OptSummary) where {T}
    isone(length(θ)) && return objective!(m, θ)
    return profileobj!(obj, m, θ, osj, Val(osj.backend))
end

function profileθj!(
    val::NamedTuple, sym::Symbol, tc::TableColumns{T}; threshold=4
) where {T}
    (; m, fwd, rev) = val
    optsum = m.optsum
    (; final, fmin) = optsum
    j = parsej(sym)
    θ = copy(final)
    osj = optsum
    pmj = m.parmap[j]
    lbj = pmj[2] == pmj[3] ? zero(T) : T(-Inf)
    if length(θ) > 1      # set up the conditional optimization problem
        notj = deleteat!(collect(axes(final, 1)), j)
        # With θ[j] held fixed, flipping the sign of a column of λ is no longer a symmetry
        # of the objective, so the usual unconstrained optimization followed by `rectify!`
        # could move to the mirror image of the model at -θ[j]. Constraining the diagonal
        # elements to be non-negative through `abs` prevents this.
        isdiagj = [(pm = m.parmap[i]; pm[2] == pm[3]) for i in notj]
        osj = optsumj(optsum, j)
        function obj(x, g=T[])
            isempty(g) ||
                throw(ArgumentError("gradients are not evaluated by this objective"))
            for i in eachindex(notj, x)
                @inbounds θ[notj[i]] = isdiagj[i] ? abs(x[i]) : x[i]
            end
            return objective!(m, θ)
        end
    else
        obj = nothing
    end
    pnm = (; p=sym)
    ζold = zero(T)
    tbl = [merge(pnm, mkrow!(tc, m, ζold))]    # start with the row for ζ = 0
    δj = inv(T(64))
    θj = final[j]
    θ[j] = max(lbj, θj - δj)    # an estimate close to the bound gets a point on the bound
    # decreasing values of θ[j], unless the estimate is on the bound
    while θj > lbj && (abs(ζold) < threshold) && length(tbl) < 100
        ζ = _ζ(profileobj!(obj, m, θ, osj), fmin, θ[j] < θj, sym, θ[j])
        push!(tbl, merge(pnm, mkrow!(tc, m, ζ)))
        θ[j] == lbj && break
        δj = _nextstep(δj, ζold - ζ, inv(T(4)))  # smaller steps for negative ζ
        ζold = ζ
        θ[j] = max(lbj, θ[j] - δj)
    end
    reverse!(tbl)               # reorder the new part of the table by increasing ζ
    sv = getproperty(sym).(tbl)
    δj = inv(T(32))             # used when the slope at the estimate cannot be determined
    if _splineable(sv)          # need to handle the case of convergence on the boundary
        slope = (
            Derivative(1) *
            interpolate(sv, getproperty(:ζ).(tbl), BSplineOrder(4), Natural())
        )(
            last(sv)
        )
        # approximate step for an increase of 0.5
        slope > 0 && isfinite(slope) && (δj = inv(T(2) * slope))
    end
    ζold = zero(T)
    copyto!(θ, final)
    θ[j] += δj
    while (ζold < threshold) && (length(tbl) < 120)
        ζ = _ζ(profileobj!(obj, m, θ, osj), fmin, false, sym, θ[j])
        push!(tbl, merge(pnm, mkrow!(tc, m, ζ)))
        δj = _nextstep(δj, ζ - ζold, inv(T(2)))
        ζold = ζ
        θ[j] += δj
    end
    append!(val.tbl, tbl)
    updateL!(setθ!(m, final))
    sv = getproperty(sym).(tbl)
    ζv = getproperty(:ζ).(tbl)
    # A flat profile, e.g. near a boundary, or an inexact conditional optimization can
    # produce values that cannot be interpolated. Skip the spline rather than abort.
    if _splineable(sv)
        fwd[sym] = interpolate(sv, ζv, BSplineOrder(4), Natural())
        isnondecreasing(fwd[sym]) || @warn "Forward spline for $sym is not monotone."
    else
        @warn "The values of $sym in its profile are not strictly increasing, " *
            "so no forward spline was constructed."
    end
    if _splineable(ζv)
        rev[sym] = interpolate(ζv, sv, BSplineOrder(4), Natural())
        isnondecreasing(rev[sym]) || @warn "Reverse spline for $sym is not monotone."
    else
        @warn "ζ is not strictly increasing in the profile of $sym, " *
            "so no reverse spline was constructed."
    end
    return val
end

# a cubic spline needs at least four distinct, increasing values to interpolate
_splineable(x::AbstractVector) = length(x) > 3 && issorted(x; lt=≤)
