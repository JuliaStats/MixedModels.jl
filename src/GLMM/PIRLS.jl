## Penalized iteratively reweighted least squares algorithm for determining the
## conditional modes of the random effects in a GLMM

type GeneralizedLinearMixedModel{T <: AbstractFloat, D <: UnivariateDistribution, L <: Link} <: MixedModel
    LMM::LinearMixedModel{T}
    r::GLM.GlmResp{Vector{T},D,L}
    β₀::DenseVector{T}
    fe::DenseMatrix{T}
    App::DenseMatrix{T}
    Rpp::DenseMatrix{T}
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector, l::Link)
    LMM = lmm(f, fr)
    y = copy(model_response(LMM))
    if isempty(wt)
        wt = ones(y)
    end
    X = LMM.trms[end][:,1:end-1]
    # fit a glm pm the fixed-effects only
    gl = glm(X, y, d, l; wts=wt, offset=zeros(y))
    β₀ = coef(gl)
    r = gl.rr
    Base.A_mul_B!(r.offset, X, β₀)
    A = LMM.A
    R = LMM.R
    trms = LMM.trms
    updatemu!(r, zeros(y))
    fe = copy(trms[end])
    App = copy(A[end,end])
    Rpp = copy(R[end,end])
    T = eltype(y)
    trms[end] = reshape(copy(r.wrkresid), (length(y), 1))
    sz = convert(Vector{Int}, map(x -> size(x,2), LMM.trms))
    pp1 = length(sz)
    for i in eachindex(sz)
        A[i, pp1] = Array(T, (sz[i], 1))
        R[i, pp1] = Array(T, (sz[i], 1))
    end
    reweight!(LMM, r.wrkwts)
    GeneralizedLinearMixedModel(LMM, r, β₀, fe, App, Rpp)
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::DataVector, l::Link)
    glmm(f,fr,d,convert(Vector,wt),l)
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector) = glmm(f, fr, d, wt, GLM.canonicallink(d))

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f,fr,d,Float64[])

function reweight!{T <: AbstractFloat}(x::LinearMixedModel{T}, wts)
    trms = x.trms
    A = x.A
    if length(wts) ≠ size(trms[1], 1)
        throw(DimensionMismatch("$(length(wts)) = length(wts) ≠ size(x.trms[1], 1)"))
    end
    for j in 1:size(A,2), i in 1:j
        wtprod!(A[i,j], trms[i], trms[j], wts)
    end
    x
end

function wtprod!{T <: AbstractFloat}(A::Diagonal{T}, ti::ScalarReMat{T}, tj::ScalarReMat{T}, wt::Vector{T})
    n, q = size(ti)
    if ti === tj
        ad = fill!(A.diag,0)
        z = ti.z
        if length(ad) ≠ q || length(wt) ≠ n || length(z) != n
            throw(DimensionMismatch("size(A) should be $q and length(wt) should be $n"))
        end
        tir = ti.f.refs
        for i in eachindex(tir, wt)
            ad[tir[i]] += abs2(z[i]) * wt[i]
        end
        return A
    end
    error("Shouldn't happen?")
end

function wtprod!{T <: AbstractFloat}(A::Matrix{T}, ti::ScalarReMat{T}, tj::Matrix{T}, wt::Vector{T})
    r, m = size(ti)
    q, n = size(tj)
    if r ≠ q || length(wt) ≠ r || size(A, 1) ≠ m || size(A, 2) ≠ n
        throw(DimensionMismatch("matrix product dimensions"))
    end
    fill!(A, 0)
    z = ti.z
    tir = ti.f.refs
    for j in 1:n
        for k in eachindex(wt)
            A[tir[k], j] += wt[k] * z[k] * tj[k, j]
        end
    end
    A
end

function wtprod!{T <: AbstractFloat}(A::Matrix{T}, ti::Matrix{T}, tj::Matrix{T}, wt::Vector{T})
    Ac_mul_B!(A, ti, scale(wt, tj))
end
