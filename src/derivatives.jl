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
    fd_setθ!

ForwardDiff.jl compatible [`setθ!`](@ref).

$(FORWARDDIFF)
"""
function fd_setθ! end

"""
    fd_updateL!

ForwardDiff.jl compatible [`updateL!`](@ref).

$(FORWARDDIFF)
"""
function fd_updateL! end

"""
    fd_pwrss

ForwardDiff.jl compatible [`pwrss`](@ref).

$(FORWARDDIFF)
"""
function fd_pwrss end

"""
    fd_logdet

ForwardDiff.jl compatible [`logdet`](@ref).

$(FORWARDDIFF)
"""
function fd_logdet end

"""
    fd_cholUnblocked!

ForwardDiff.jl compatible [`cholUnblocked!`](@ref).

$(FORWARDDIFF)
"""
function fd_cholUnblocked! end

"""
    fd_rankUpdate!

ForwardDiff.jl compatible [`rankUpdate!`](@ref).

$(FORWARDDIFF)
"""
function fd_rankUpdate! end
