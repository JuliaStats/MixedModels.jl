## Penalized iteratively reweighted least squares algorithm for determining the
## conditional modes of the random effects in a GLMM

type GeneralizedLinearMixedModel{T <: AbstractFloat, D <: UnivariateDistribution, L <: Link} <: MixedModel
    LMM::LinearMixedModel{T}
    r::GLM.GlmResp{Vector{T},D,L}
    β₀::DenseVector{T}
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector, l::Link)
    LMM = lmm(f, fr)
    y = copy(model_response(LMM))
    if isempty(wt)
        wt = ones(y)
    end
    X = LMM.trms[end][:,1:end-1]
    gl = glm(X, y, d, l; wts=wt, offset=zeros(y))
    β₀ = coef(gl)
    r = gl.rr
    Base.A_mul_B!(r.offset, X, β₀)
    GeneralizedLinearMixedModel(LMM, r, β₀)
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::DataVector, l::Link)
    glmm(f,fr,d,convert(Vector,wt),l)
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector) = glmm(f, fr, d, wt, GLM.canonicallink(d))

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f,fr,d,Float64[])
