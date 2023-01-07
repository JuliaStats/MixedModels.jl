
function profilevc(m::LinearMixedModel{T}, val, rowj::AbstractVector{T}) where {T}
    optsum = m.optsum
    sigma() = val / norm(rowj)   # a function to be re-evaluated when values in rowj change
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g must be empty"))
        updateL!(setθ!(m, x))
        optsum.sigma = sigma()
        return objective(m)
    end
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    _check_nlopt_return(ret)
    return fmin
end

function profileσs(m::LinearMixedModel{T}; threshold::Number=4) where {T}
    @compat (; λ, σ, β, optsum, parmap, reterms) = m
    isnothing(optsum.sigma) || throw(ArgumentError("Can't profile vc's when σ is fixed"))
    @compat (; initial, final, fmin) = optsum
    saveinitial = copy(initial)
    copyto!(initial, final)
    val = (; ζ = T[0], β = [(β...,)], σ = [(σ, σvals(m)...)], θ = [(final...,)])
    for t in reterms
        for r in eachrow(t.λ.data)
            σij = σ * norm(r)
            δ = (iszero(σij) ? one(T) : σij) * T(inv(32))
            obj = profilevc(m, σij + δ, r)
            push!(val.ζ, sqrt(obj - fmin))
            push!(val.β, (m.β...,))
            push!(val.σ, (m.σ, σvals(m)...))
            push!(val.θ, (final...,))
            stepsz = δ / (2 * last(val.ζ))
            @info σij, δ, obj, last(val.ζ), stepsz
        end
    end
    copyto!(final, initial)
    copyto!(initial, saveinitial)
    optsum.sigma = nothing
    updateL!(setθ!(m, final))
    return Table(val)
end
