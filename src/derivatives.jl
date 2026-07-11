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
