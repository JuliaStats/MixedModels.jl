"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

# Fields
- `LMM`: a [`LinearMixedModel`](@ref) - the local approximation to the GLMM.
- `β`: the fixed-effects vector
- `β₀`: similar to `β`. Used in the PIRLS algorithm if step-halving is needed.
- `θ`: covariance parameter vector
- `b`: similar to `u`, equivalent to `broadcast!(*, b, LMM.Λ, u)`
- `u`: a vector of matrices of random effects
- `u₀`: similar to `u`.  Used in the PIRLS algorithm if step-halving is needed.
- `resp`: a `GlmResp` object
- `η`: the linear predictor
- `wt`: vector of prior case weights, a value of `T[]` indicates equal weights.
The following fields are used in adaptive Gauss-Hermite quadrature, which applies
only to models with a single random-effects term, in which case their lengths are
the number of levels in the grouping factor for that term.  Otherwise they are
zero-length vectors.
- `devc`: vector of deviance components
- `devc0`: vector of deviance components at offset of zero
- `sd`: approximate standard deviation of the conditional density
- `mult`: multiplier

# Properties

In addition to the fieldnames, the following names are also accessible through the `.` extractor

- `theta`: synonym for `θ`
- `beta`: synonym for `β`
- `σ` or `sigma`: common scale parameter (value is `NaN` for distributions without a scale parameter)
- `lowerbd`: vector of lower bounds on the combined elements of `β` and `θ`
- `formula`, `trms`, `A`, `L`, and `optsum`: fields of the `LMM` field
- `X`: fixed-effects model matrix
- `y`: response vector

"""
struct GeneralizedLinearMixedModel{T <: AbstractFloat} <: MixedModel{T}
    LMM::LinearMixedModel{T}
    β::Vector{T}
    β₀::Vector{T}
    θ::Vector{T}
    b::Vector{Matrix{T}}
    u::Vector{Matrix{T}}
    u₀::Vector{Matrix{T}}
    resp::GlmResp
    η::Vector{T}
    wt::Vector{T}
    devc::Vector{T}
    devc0::Vector{T}
    sd::Vector{T}
    mult::Vector{T}
end


function Base.getproperty(m::GeneralizedLinearMixedModel, s::Symbol)
    if s == :theta
        m.θ
    elseif s == :beta
        m.β
    elseif s ∈ (:λ, :lambda)
        getΛ(m)
    elseif s ∈ (:σ, :sigma)
        sdest(m)
    elseif s == :lowerbd
        m.LMM.optsum.lowerbd
    elseif s ∈ (:formula, :trms, :A, :L, :optsum)
        getfield(m.LMM, s)
    elseif s == :X
        m.LMM.trms[end - 1].x
    elseif s == :y
        vec(m.LMM.trms[end].x)
    else
        getfield(m, s)
    end
end

function Base.setproperty!(m::GeneralizedLinearMixedModel, s::Symbol, y)
    if s ∈ (:θ, :theta)
        setθ!(m, y)
    elseif s ∈ (:β, :beta)
        setβ!(m, y)
    elseif s ∈ (:βθ, :betatheta)
        setβθ!(m, y)
    else
        setfield!(m, s, y)
    end
end

Base.propertynames(m::GeneralizedLinearMixedModel, private=false) =
    (:theta, :beta, :λ, :lambda, :σ, :sigma, :X, :y, :lowerbd, fieldnames(typeof(m))..., fieldnames(typeof(m.LMM))...)
