## Penalized iteratively reweighted least squares algorithm for determining the
## conditional modes of the random effects in a GLMM

type GeneralizedLinearMixedModel{S<:PLSSolver} <: MixedModel
    LMM::LinearMixedModel{S}
    d::Distribution
    l::Link
    wt::Vector{Float64}
    η::Vector{Float64}
    dμdη::Vector{Float64}
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector, l::Link)
    LMM = lmm(f,fr)
    y = LMM.y
    n = length(y)
    if length(wt) == 0
        wt = ones(n)
    elseif length(wt) != n
        throw(DimensionMismatch(""))
    end
    eltype(y) == eltype(wt) || error("wt Vector must be same type as y")
    η = [link(l,μ) for μ in mustart!(LMM.μ,d,y,wt)]
    GeneralizedLinearMixedModel(LMM,d,l,wt,η,[μη(l,eta) for eta in η])
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector) = glmm(f,fr,d,wt,canonical(d))

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f,fr,d,Array(Float64,(0,)))
