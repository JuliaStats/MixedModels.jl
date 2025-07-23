module MixedModelsFiniteDiffExt

using MixedModels: LinearMixedModel, objective!, updateL!, setθ!
using FiniteDiff: FiniteDiff, finite_difference_gradient, finite_difference_hessian

const FINITEDIFF = """
!!! warning "FiniteDiff.jl support is experimental."
    Compatibility with FiniteDiff.jl is experimental. The precise structure,
    including function names and method definitions, is subject to
    change without being considered a breaking change. In particular,
    the exact set of parameters included is subject to change. The
    θ parameter is always included, but whether σ and/or the fixed effects
    should be included is currently still being decided.
"""

"""
    FiniteDiff.finite_difference_gradient(model::LinearMixedModel, args...; kwargs...)

Evaluate the gradient of the objective function at the currently fitted parameter
values.

$(FINITEDIFF)
"""
function FiniteDiff.finite_difference_gradient(
    model::LinearMixedModel{T}, θ::Vector{T}=model.θ, args...; kwargs...
) where {T}
    local grad
    try
        grad = finite_difference_gradient(x -> objective!(model, x), θ, args...; kwargs...)
    finally
        updateL!(setθ!(model, θ))
    end

    return grad
end

"""
    FiniteDiff.finite_difference_hessian(model::LinearMixedModel, args...; kwargs...)

Evaluate the Hessian of the objective function at the currently fitted parameter
values.

$(FINITEDIFF)
"""
function FiniteDiff.finite_difference_hessian(
    model::LinearMixedModel{T}, θ::Vector{T}=model.θ, args...; kwargs...
) where {T}
    local hess
    try
        hess = finite_difference_hessian(x -> objective!(model, x), θ, args...; kwargs...)
    finally
        updateL!(setθ!(model, θ))
    end

    return hess
end


end # module
