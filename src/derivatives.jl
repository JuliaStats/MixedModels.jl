const FORWARDDIFF = """
!!! warning "ForwardDiff.jl support is experimental."
    Compatibility with ForwardDiff.jl is experimental. The precise structure,
    including function names and method definitions, is subject to
    change without being considered a breaking change. In particular,
    the whole set of `fd_` functions should be considered private implementation
    details.
"""

"""
    fd_deviance

ForwardDiff.jl compatible [`deviance`](@ref).

$(FORWARDDIFF)
"""
function fd_deviance end

"""
    fd_gradient_workspace(m::LinearMixedModel)

Create a reusable workspace for [`fd_objective_gradient!`](@ref), caching the
promoted (dual-valued) copies of the model's numerical fields along with the
ForwardDiff configuration.

$(FORWARDDIFF)
"""
function fd_gradient_workspace end

"""
    fd_objective_gradient!(fdws, g, m::LinearMixedModel, θ)

ForwardDiff.jl based analogue of [`objective_gradient!`](@ref): overwrite `g`
with the gradient of the objective at `θ` and return the objective value,
evaluated by forward-mode automatic differentiation using the workspace `fdws`
from [`fd_gradient_workspace`](@ref).

Unlike `objective_gradient!`, this does not update `m` itself to `θ`.

$(FORWARDDIFF)
"""
function fd_objective_gradient! end
