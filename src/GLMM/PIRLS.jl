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

function reweightZ{T <: AbstractFloat}(x::LinearMixedModel{T}, wts)
    trms = x.trms
    A = x.A
    if length(wts) ≠ size(trms[1], 1)
        throw(DimensionMismatch("$(length(wts)) = length(wts) ≠ size(x.trms[1], 1)"))
    end
    for j in 1:length(x.Λ), i in 1:j
        wtprod!(A[i,j],trms[i],trms[j],wts)
    end
    x
end

function wtprod!{T<:Real}(A::Diagonal{T}, ti::ScalarReMat{T}, tj::ScalarReMat{T}, wt::Vector{T})
    n, q = size(ti)
    if ti === tj
        ad = fill!(A.diag,0)
        z = ti.z
        if length(ad) ≠ q || length(wt) ≠ n || length(z) != n
            throw(DimensionMismatch("size(A) should be $q and length(wt) should be $n"))
        end
        tir = ti.f.refs
        for i in eachindex(tir,wt, )
            ad[tir[i]] += abs2(z[i]) * wt[i]
        end
        return A
    end
    error("Shouldn't happen?")
end
