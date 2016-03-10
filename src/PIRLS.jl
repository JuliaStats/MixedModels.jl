## Penalized iteratively reweighted least squares algorithm for determining the
## conditional modes of the random effects in a GLMM

type GeneralizedLinearMixedModel{T <: AbstractFloat, D <: UnivariateDistribution, L <: Link} <: MixedModel
    LMM::LinearMixedModel{T}
    r::GLM.GlmResp{Vector{T},D,L}
    β₀::DenseVector{T}
    u₀::Vector
    δ::Vector
    fe::DenseMatrix{T}
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector, l::Link)
    LMM = lmm(f, fr)
    y = copy(model_response(LMM))
    if isempty(wt)
        wt = ones(y)
    end
    A, R, trms = LMM.A, LMM.R, LMM.trms
    fe = copy(trms[end])
    X = fe[:,1:end-1]
                    # fit a glm pm the fixed-effects only
    gl = glm(X, y, d, l; wts=wt, offset=zeros(y))
    β₀ = coef(gl)
    r = gl.rr
    Base.A_mul_B!(r.offset, X, β₀)
    updatemu!(r, zeros(y))
    T = eltype(y)
    trms[end] = reshape(copy(r.wrkresid), (length(y), 1))
    sz = convert(Vector{Int}, map(x -> size(x,2), LMM.trms))
    pp1 = length(sz)
    for i in eachindex(sz)
        A[i, pp1] = Array(T, (sz[i], 1))
        R[i, pp1] = Array(T, (sz[i], 1))
    end
    LMM.weights = copy(r.wrkwts)
    reweight!(LMM)
    LMM[:θ] = LMM[:θ]        # forces an update of R
    δ = ranef(LMM, true)
    GeneralizedLinearMixedModel(LMM, r, β₀, map(zeros, δ), δ, fe)
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::DataVector, l::Link)
    glmm(f,fr,d,convert(Vector,wt),l)
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector) = glmm(f, fr, d, wt, GLM.canonicallink(d))

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f, fr, d, Float64[])

lmm(m::GeneralizedLinearMixedModel) = m.LMM

function usolve!(m::GeneralizedLinearMixedModel)
    m
end
