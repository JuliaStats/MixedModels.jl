module MixedModelsForwardDiffExt

using MixedModels
using MixedModels:
    _logdet,
    fd_deviance,
    log2π,
    pwrss,
    setθ!,
    ssqdenom,
    TriangularRFP,
    updateL!
using LinearAlgebra: LowerTriangular, copy_oftype
using ForwardDiff: ForwardDiff,
    Chunk,
    DiffResults,
    GradientConfig,
    HessianConfig

const FORWARDDIFF = """
!!! warning "Large allocations"
    Most of MixedModels.jl relies strongly on in-place methods in order to minimize
    the amount of memory allocated. In addition to reducing the memory burden
    (especially for large models), this practice generally speeds up evaluation
    of the objective. In-place methods, however, generally do not play well with
    automatic differentiation, which requires out-of-place copies promoted to the
    dual number type on every evaluation. These will generally be slower and much
    more memory intensive, so use of this functionality is **not** recommended
    for large models.

!!! warning "ForwardDiff.jl support is experimental."
    Compatibility with ForwardDiff.jl is experimental. The precise structure,
    including function names and method definitions, is subject to
    change without being considered a breaking change. In particular,
    the exact set of parameters included is subject to change. The
    θ parameter is always included; σ is profiled out (or held at its fixed
    value for models fitted with a fixed `sigma`), matching [`objective`](@ref),
    but whether the fixed effects should be included is still being decided.
"""

#####
##### Gradients
#####

function ForwardDiff.GradientConfig(
    model::LinearMixedModel{T}, x::AbstractVector{T}=model.θ, chunk::Chunk=Chunk(x)
) where {T}
    return GradientConfig(fd_deviance(model), x, chunk)
end

"""
    ForwardDiff.gradient(model::LinearMixedModel)

Evaluate the gradient of the objective function at the currently fitted parameter
values.

$(FORWARDDIFF)
"""
function ForwardDiff.gradient(
    model::LinearMixedModel{T}, θ::Vector{T}=model.θ,
    cfg::GradientConfig=GradientConfig(model, θ),
    check::Val{CHK}=Val(true),
) where {T,CHK}
    return ForwardDiff.gradient!(similar(model.θ), model, θ, cfg, check)
end

function ForwardDiff.gradient!(result::AbstractArray,
    model::LinearMixedModel{T}, θ::Vector{T}=model.θ,
    cfg::GradientConfig=GradientConfig(model, θ),
    check::Val{CHK}=Val(true),
) where {T,CHK}
    return ForwardDiff.gradient!(result, fd_deviance(model), θ, cfg, check)
end

#####
##### Hessians
#####

function ForwardDiff.HessianConfig(
    model::LinearMixedModel{T}, x::AbstractVector{T}=model.θ, chunk::Chunk=Chunk(x)
) where {T}
    return HessianConfig(fd_deviance(model), x, chunk)
end

"""
    ForwardDiff.hessian(model::LinearMixedModel)

Evaluate the Hessian of the objective function at the currently fitted parameter
values.

$(FORWARDDIFF)
"""
function ForwardDiff.hessian(
    model::LinearMixedModel{T}, θ::Vector{T}=model.θ,
    cfg::HessianConfig=HessianConfig(model, θ),
    check::Val{CHK}=Val(true),
) where {T,CHK}
    n = length(θ)
    return ForwardDiff.hessian!(Matrix{T}(undef, n, n), model, θ, cfg, check)
end

function ForwardDiff.hessian!(result::AbstractArray,
    model::LinearMixedModel{T}, θ::Vector{T}=model.θ,
    cfg::HessianConfig=HessianConfig(model, θ),
    check::Val{CHK}=Val(true),
) where {T,CHK}
    return ForwardDiff.hessian!(result, fd_deviance(model), θ, cfg, check)
end

#####
##### Evaluation of objective
#####

MixedModels.fd_deviance(model) = Base.Fix1(fd_deviance, model)

# promote a block of L to the dual number type, preserving the structural wrappers
# that updateL! dispatches on.  A TriangularRFP block cannot hold dual numbers (its
# element type is restricted to BlasFloat), so it is expanded to a dense lower triangle
_fdcopy(A::AbstractMatrix, ::Type{T}) where {T} = copy_oftype(A, T)
function _fdcopy(A::LowerTriangular, ::Type{T}) where {T}
    return LowerTriangular(copy_oftype(parent(A), T))
end
_fdcopy(A::TriangularRFP, ::Type{T}) where {T} = LowerTriangular(copy_oftype(Array(A), T))

function MixedModels.fd_deviance(model::LinearMixedModel, θ::AbstractVector{T}) where {T}
    # extract and promote to the dual number type, then run the same
    # pipeline as objective!(model, θ) on the promoted copies
    AA = [copy_oftype(Ai, T) for Ai in model.A]
    LL = [_fdcopy(Li, T) for Li in model.L]
    RR = [copy_oftype(Ri, T) for Ri in model.reterms]

    return _fd_objective(model, AA, LL, RR, θ)
end

function _fd_objective(model::LinearMixedModel, AA::Vector, LL::Vector, RR::Vector, θ)
    setθ!(RR, model.parmap, θ)
    updateL!(AA, LL, RR)

    r² = pwrss(LL)
    ld = _logdet(LL, RR, model.optsum.REML)
    dof = ssqdenom(model)

    σ = model.optsum.sigma
    val = if isnothing(σ)
        ld + dof * (1 + log2π + log(r² / dof))
    else
        muladd(dof, muladd(2, log(σ), log2π), ld + r² / σ^2)
    end
    wts = model.sqrtwts
    return isempty(wts) ? val : val - 2 * sum(log, wts)
end

#####
##### Reusable workspace for gradient-based optimization
#####

# callable evaluating the objective from cached promoted copies of the model's
# numerical fields, so that repeated gradient evaluations (e.g. within a
# gradient-based optimizer) do not reallocate the promoted copies
struct FDCachedObjective{Md<:LinearMixedModel,VA<:Vector,VL<:Vector,VR<:Vector}
    model::Md
    AA::VA
    LL::VL
    RR::VR
end

(f::FDCachedObjective)(θ::AbstractVector) = _fd_objective(f.model, f.AA, f.LL, f.RR, θ)

function MixedModels.fd_gradient_workspace(model::LinearMixedModel{T}) where {T}
    x = Vector{T}(model.θ)
    # the tag is constructed from the closure fd_deviance(model) rather than the
    # FDCachedObjective, whose type mentions the dual type itself; the tag check
    # is therefore disabled in fd_objective_gradient!
    tag = ForwardDiff.Tag(fd_deviance(model), T)
    cfg = GradientConfig(nothing, x, Chunk(x), tag)
    D = eltype(cfg.duals)
    f = FDCachedObjective(model,
        [copy_oftype(Ai, D) for Ai in model.A],
        [_fdcopy(Li, D) for Li in model.L],
        [copy_oftype(Ri, D) for Ri in model.reterms])
    return (; f, cfg, result=DiffResults.GradientResult(x))
end

function MixedModels.fd_objective_gradient!(
    fdws::NamedTuple, g::AbstractVector, m::LinearMixedModel, θ::AbstractVector
)
    fdws.f.model === m ||
        throw(ArgumentError("the workspace was created for a different model"))
    ForwardDiff.gradient!(fdws.result, fdws.f, θ, fdws.cfg, Val(false))
    copyto!(g, DiffResults.gradient(fdws.result))
    return DiffResults.value(fdws.result)
end

end # module
