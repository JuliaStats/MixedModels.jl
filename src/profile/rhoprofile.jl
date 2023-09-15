"""
     profilerho12(m::LinearMixedModel{T}, mult::T) where {T}

Profile ρ₁₂ where `mult` is `ρ/sqrt(1-ρ²)`

!!! note
    This method is called by `profile` and currently considered internal.
    As such, it may change or disappear in a future release without being considered breaking.
"""
function profilerho12(m::LinearMixedModel{T}, mult::T) where {T}
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g must be empty"))
        θ = m.θ
        θ[1] = x[1]
        θ[2] = mult * x[2]
        θ[3] = x[2]
        return objective(updateL!(setθ!(m, θ)))
    end
    optsum = optsumj(m.optsum, 2)
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    _check_nlopt_return(ret)
    return fmin, xmin
end

"""
    profileσ(m::LinearMixedModel, tc::TableColumns; threshold=4)

Return a Table of the profile of `σ` for model `m`.  The profile extends to where the magnitude of ζ exceeds `threshold`.
!!! note
    This method is called by `profile` and currently considered internal.
    As such, it may change or disappear in a future release without being considered breaking.
"""
function profileρ(m::LinearMixedModel{T}) where {T}
    minobj = m.objective
    ub = 1. - inv(256)
    ρvals = -ub:inv(128):ub
    nρ = length(ρvals)
    ζvec = sizehint!(T[], length(ρvals))
    for ρ in ρvals
        mult = ρ / sqrt(one(ρ) - abs2(ρ))
        obj, θshort = profilerho12(m, mult)
        push!(ζvec, sqrt(obj - minobj))
    end
    return ρvals, ζvec
end
