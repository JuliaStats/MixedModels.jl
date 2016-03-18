## Penalized iteratively reweighted least squares algorithm for determining the
## conditional modes of the random effects in a GLMM

type GeneralizedLinearMixedModel{T <: AbstractFloat, D <: UnivariateDistribution, L <: Link} <: MixedModel
    LMM::LinearMixedModel{T}
    dist::D
    link::L
    β₀::DenseVector{T}
    u::Vector
    u₀::Vector
    X::DenseMatrix{T}
    y::DenseVector{T}
    μ::DenseVector{T}
    η::DenseVector{T}
    dμdη::DenseVector{T}
    devresid::DenseVector{T}
    offset::DenseVector{T}
    var::DenseVector{T}
    wrkresid::DenseVector{T}
    wrkwt::DenseVector{T}
    wt::DenseVector{T}
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector, l::Link)
    LMM = lmm(f, fr)
    A, R, trms, u, y = LMM.A, LMM.R, LMM.trms, ranef(LMM, true), copy(model_response(LMM))
    kp1 = length(LMM.Λ) + 1
    X = copy(trms[kp1])         # the copy may be unnecessary
            # zero the dimension of the fixed-effects in trms, A and R
    trms[kp1] = zeros((length(y), 0))
    for i in 1:kp1
        qi = size(trms[i], 2)
        A[i, kp1] = zeros((qi, 0))
        R[i, kp1] = zeros((qi, 0))
    end
    qend = size(trms[end], 2)  # should always be 1 but no harm in extracting it
    A[kp1, end] = zeros((0, qend))
    R[kp1, end] = zeros((0, qend))
            # fit a glm pm the fixed-effects only
    gl = glm(X, y, d, l; wts = isempty(wt) ? ones(y) : wt)
    r = gl.rr
    β₀ = coef(gl)
    res = GeneralizedLinearMixedModel(LMM, d, l, β₀, u, map(copy, u), X, y, r.mu,
        r.eta, r.mueta, r.devresid, X * β₀, r.var, r.wrkresid, r.wrkwts, r.wts)
    updateμ!(res)
    wrkresp!(trms[end], res)
    LMM.weights = copy(res.wrkwt)
    reweight!(LMM)
    fit!(LMM)
    map!(copy, res.u, ranef!(res.u₀, LMM, true))
    pwrss!(res)
    res
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::DataVector, l::Link)
    glmm(f, fr, d, convert(Vector, wt), l)
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector) = glmm(f, fr, d, wt, GLM.canonicallink(d))

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f, fr, d, Float64[])

lmm(m::GeneralizedLinearMixedModel) = m.LMM

pwrss(m::GeneralizedLinearMixedModel) = pwrss(lmm(m))

Base.logdet(m::GeneralizedLinearMixedModel) = logdet(lmm(m))

function LaplaceDeviance(m::GeneralizedLinearMixedModel)
    sum(m.devresid) + mapreduce(sumabs2, +, m.u) + logdet(m)
end

function updateη!{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T})
    lm = lmm(m)
    η, u, Λ, trms = m.η, m.u, lm.Λ, lm.trms
    fill!(η, zero(T))
    for i in eachindex(u)
        unscaledre!(η, trms[i], Λ[i], u[i])
    end
    updateμ!(m)
end

function pwrss!(m::GeneralizedLinearMixedModel)
    lm = m.LMM
    updateη!(m)
    wrkresp!(lm.trms[end], m)
    reevaluateAend!(lm)
    lm[:θ] = lm[:θ]
    pwrss(lm)
end

function pirls!{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T})
    u₀, u, iter, maxiter = m.u₀, m.u, 0, 100
    obj₀ = pwrss(m)
    while iter < maxiter
        iter += 1
        nhalf = 0
        obj = pwrss!(m)
        @show iter, obj
        while obj >= obj₀
            nhalf += 1
            if nhalf > 10
                throw(ErrorException("number of averaging steps > 10"))
            end
            for i in eachindex(u)
                ui = u[i]
                ui₀ = u₀[i]
                for j in eachindex(ui)
                    ui[j] += ui₀[j]
                    ui[j] *= 0.5
                end
            end
            obj = pwrss!(m)
            @show nhalf, obj
        end
        if isapprox(obj, obj₀; rtol = 0.001)
            break
        end
        obj₀ = obj
    end
    m
end
