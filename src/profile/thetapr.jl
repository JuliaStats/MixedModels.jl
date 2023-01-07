
"""
    optsumj(os::OptSummary, j::Integer)

Return an `OptSummary` with the `j`'th component of the parameter omitted.

`os.final` with its j'th component omitted is used as the initial parameter.
""" 
function optsumj(os::OptSummary, j::Integer)
    return OptSummary(
        deleteat!(copy(os.final), j),
        deleteat!(copy(os.lowerbd), j),
        os.optimizer
    )
end

function profileobj!(m::LinearMixedModel{T}, θ::AbstractVector{T}, opt::Opt, osj::OptSummary) where {T}
    isone(length(θ)) && return objective!(m, θ)
    fmin, xmin, ret = NLopt.optimize(opt, copyto!(osj.final, osj.initial))
    _check_nlopt_return(ret)
    return fmin
end

function profileθj!(pr::MixedModelProfile{T}, sym::Symbol, tc::TableColumns{T}; threshold=4) where {T}
    @compat (; m, fwd, rev) = pr
    optsum = m.optsum
    @compat (; final, fmin, lowerbd) = optsum
    j = parsej(sym)
    θ = copy(final)
    lbj = lowerbd[j]
    osj = optsum
    opt = Opt(osj)
    if length(θ) > 1      # set up the conditional optimization problem
        notj = deleteat!(collect(axes(final, 1)), j)
        osj = optsumj(optsum, j)
        opt = Opt(osj)               # create an NLopt optimizer object for the reduced problem
        function obj(x, g)
            isempty(g) || throw(ArgumentError("gradients are not evaluated by this objective"))
            for i in eachindex(notj, x)
                @inbounds θ[notj[i]] = x[i]
            end
            return objective!(m, θ)
        end
        NLopt.min_objective!(opt, obj)
    end
    pnm = (; p = sym)
    ζold = zero(T)
    tbl = [merge(pnm, mkrow!(tc, m, ζold))]    # start with the row for ζ = 0
    δj = inv(T(32))
    θj = final[j]
    θ[j] = θj - δj
    while (abs(ζold) < threshold) && θ[j] ≥ lbj && length(tbl) < 100  # decreasing values of θ[j]
        ζ = sign(θ[j] - θj) * sqrt(profileobj!(m, θ, opt, osj) - fmin)
        push!(tbl, merge(pnm, mkrow!(tc, m, ζ)))
        θ[j] == lbj && break
        δj /= (2 * abs(ζ - ζold))
        ζold = ζ
        θ[j] = max(lbj, (θ[j] -= δj))
    end
    reverse!(tbl)                      # reorder the new part of the table by increasing ζ
    sv = getproperty(sym).(tbl)
    slope = (Derivative(1) * interpolate(sv, getproperty(:ζ).(tbl), BSplineOrder(4), Natural()))(last(sv))
    δj = inv(T(2) * slope)  # approximate step for increase of 0.5
    ζold = zero(T)
    copyto!(θ, final)
    θ[j] += δj
    while (ζold < threshold) && (length(tbl) < 120)
        ζ = sqrt(profileobj!(m, θ, opt, osj) - fmin)
        push!(tbl, merge(pnm, mkrow!(tc, m, ζ)))
        δj /= (2 * abs(ζ - ζold))
        ζold = ζ
        θ[j] += δj
    end
    append!(pr.tbl, tbl)
    updateL!(setθ!(m, final))
    sv = getproperty(sym).(tbl)
    ζv = getproperty(:ζ).(tbl)
    pr.fwd[sym] = interpolate(sv, ζv, BSplineOrder(4), Natural())
    pr.rev[sym] = interpolate(ζv, sv, BSplineOrder(4), Natural())
    return pr
end
