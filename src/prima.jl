
"""
    prfit!(m::LinearMixedModel; kwargs...)

Fit a mixed model using the [PRIMA](https://github.com/libprima/PRIMA.jl) implementation
of the BOBYQA optimizer.

!!! warning "Experimental feature"
    This function is an experimental feature that will go away in the future.
    Do **not** rely on it, unless you are willing to pin the precise MixedModels.jl
    version. The purpose of the function is to provide the MixedModels developers
    a chance to explore the performance of the PRIMA implementation without the large
    and potentially breaking changes it would take to fully replace the current NLopt
    backend with a PRIMA backend or a backend supporting a range of optimizers.

!!! note "Package extension"
    In order to reduce the dependency burden, all methods of this function are
    implemented in a package extension and are only defined when PRIMA.jl is loaded
    by the user.
"""
function prfit! end
