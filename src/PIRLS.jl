## Penalized iteratively reweighted least squares algorithm for determining the
## conditional modes of the random effects in a GLMM

type GeneralizedLinearMixedModel{T <: AbstractFloat, D <: UnivariateDistribution, L <: Link} <: MixedModel
    LMM::LinearMixedModel{T}
    r::GLM.GlmResp{Vector{T}, D, L}
    β₀::DenseVector{T}
    u₀::Vector
    δ::Vector
    X::DenseMatrix{T}
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector, l::Link)
    LMM = lmm(f, fr)
    y = copy(model_response(LMM))
    trms = LMM.trms
    kp1 = length(trms) - 1
    X = copy(trms[kp1])
    trms[kp1] = trms[end]
    resize!(trms, kp1)
    LMM.A, LMM.R = generateAR(trms, kp1 - 1)
                    # fit a glm pm the fixed-effects only
    gl = glm(X, y, d, l; wts = isempty(wt) ? ones(y) : wt, offset = zeros(y))
    β₀ = coef(gl)
    r = gl.rr
    Base.A_mul_B!(r.offset, X, β₀)
    updatemu!(r, zeros(y))
    copy!(trms[end], r.wrkresid)
    ## FIXME  When using prior weights this will need to be modified.
    LMM.weights = copy(r.wrkwts)
    reweight!(LMM)
    fit!(LMM)
    δ = ranef(LMM, true)
    GeneralizedLinearMixedModel(LMM, r, β₀, map(zeros, δ), δ, X)
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::DataVector, l::Link)
    glmm(f, fr, d, convert(Vector, wt), l)
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector) = glmm(f, fr, d, wt, GLM.canonicallink(d))

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f, fr, d, Float64[])

lmm(m::GeneralizedLinearMixedModel) = m.LMM

function ustep(m::GeneralizedLinearMixedModel, fac)
    ## FIXME change the action of GLM.updatemu! so that the copying of linpr to r.eta happens externally
    lm = lmm(m)
    u₀, δ, Λ, trms = m.u₀, m.δ, lm.Λ, lm.trms
    trialeta = zeros(size(m.X, 1))
    for i in eachindex(δ)
        ui = u₀[i]
        ti = copy(δ[i])
        for j in eachindex(ti)
            ti[j] *= fac
            ti[j] += ui[j]
        end
        unscaledre!(trialeta, trms[i], Λ[i], ti)
    end
    updatemu!(m.r, trialeta)
end
