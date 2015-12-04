## Penalized iteratively reweighted least squares algorithm for determining the
## conditional modes of the random effects in a GLMM

type GeneralizedLinearMixedModel{T} <: MixedModel
    LMM::LinearMixedModel{T}
    d::Distribution
    l::Link
    wt::Vector{T}
    η::Vector{T}
    dμdη::Vector{T}
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector, l::Link)
    LMM = lmm(f,fr)
    y = model_response(LMM)
    n = length(y)
    T = eltype(y)
    if length(wt) == 0
        wt = ones(y)
    elseif length(wt) != n
        throw(DimensionMismatch("length(wt) should be 0 or length(y)"))
    end
    if eltype(wt) ≠ T
        throw(ArgumentError("eltype(wt) must be eltype(y)"))
    end
    μ = similar(y)
    η = similar(y)
    for i in eachindex(y)
        μ[i] = mu = mustart(d,y[i],wt[i])
        η[i] = link(l,mu)
    end
    GeneralizedLinearMixedModel(LMM,d,l,wt,η,T[μη(l,eta) for eta in η])
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::DataVector, l::Link)
    glmm(f,fr,d,convert(Vector,wt),l)
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector) = glmm(f,fr,d,wt,canonical(d))

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f,fr,d,Float64[])
