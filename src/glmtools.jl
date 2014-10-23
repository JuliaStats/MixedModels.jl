using Distributions, ArrayViews
abstract Link

type CauchitLink <: Link end
type CloglogLink  <: Link end
type IdentityLink <: Link end
type InverseLink  <: Link end
type LogitLink <: Link end
type LogLink <: Link end
type ProbitLink <: Link end
type SqrtLink <: Link end

two(x::FloatingPoint) = one(x) + one(x)
half(x::FloatingPoint) = inv(two(x))

xlogx(x::FloatingPoint) = x > zero(x) ? x * log(x) : zero(x)
xlogx(x::Real) = xlogx(float(x))

xlogy{T<:FloatingPoint}(x::T, y::T) = x > zero(T) ? x * log(y) : zero(x)
xlogy{T<:Real}(x::T, y::T) = xlogy(float(x), float(y))
xlogy(x::Real, y::Real) = xlogy(promote(x, y)...)

link(::Type{CauchitLink},μ::FloatingPoint) = tan(pi*(μ - half(μ)))
linkinv(::Type{CauchitLink},η::FloatingPoint) = half(η) + atan(η)/π
dμdη(::Type{CauchitLink},η::FloatingPoint) = inv(π*(one(η) + abs2(η)))

link(::Type{CloglogLink},μ::FloatingPoint) = log(-log(one(μ) - μ))
linkinv(::Type{CloglogLink},η::FloatingPoint) = -expm1(-exp(η))
dμdη(::Type{CloglogLink},η::FloatingPoint) = exp(η)*exp(-exp(η))

link(::Type{IdentityLink},μ::FloatingPoint) = μ
linkinv(::Type{IdentityLink},η::FloatingPoint) = η
dμdη(::Type{IdentityLink},μ::FloatingPoint) = one(μ)

link(::Type{InverseLink},μ::FloatingPoint) = inv(μ)
linkinv(::Type{InverseLink},η::FloatingPoint) = inv(η)
dμdη(::Type{InverseLink},η::FloatingPoint) = -inv(abs2(η))

link(::Type{LogitLink},μ::FloatingPoint) = log(μ/(one(μ)-μ))
linkinv(::Type{LogitLink},η::FloatingPoint) = inv(one(η) + exp(-η))
dμdη(::Type{LogitLink},η::FloatingPoint) = (e = exp(-abs(η)); e/abs2(one(e)+e))

link(::Type{LogLink},μ::FloatingPoint) = log(μ)
linkinv(::Type{LogLink},η::FloatingPoint) = exp(η)
dμdη(::Type{LogLink},η::FloatingPoint) = exp(η)

link(::Type{ProbitLink},μ::FloatingPoint) = sqrt2*erfinv((two(μ)*μ - one(μ)))
linkinv(::Type{ProbitLink},η::FloatingPoint) = (one(η) + erf(η/sqrt2))/two(η)
dμdη(::Type{ProbitLink},η::FloatingPoint) = exp(-abs2(η)/two(η))/sqrt2π

link(::Type{SqrtLink},μ::FloatingPoint) = sqrt(μ)
linkinv(::Type{SqrtLink},η::FloatingPoint) = abs2(η)
dμdη(::Type{SqrtLink},η::FloatingPoint) = η + η

function link!{L<:Link,T<:FloatingPoint}(::Type{L},η::SharedVector{T},μ::SharedVector{T})
    (n = length(η)) == length(μ) || throw(DimensionMismatch(""))
    @parallel for i in 1:n
        @inbounds η[i] = link(L,μ[i])
    end
    η
end

function link!{L<:Link,T<:FloatingPoint}(::Type{L},η::DenseVector{T},μ::DenseVector{T})
    (n = length(η)) == length(μ) || throw(DimensionMismatch(""))
    @simd for i in 1:n
        @inbounds η[i] = link(L,μ[i])
    end
    η
end

abstract ModResp

type GLMResp{V<:DenseArray{Float64,1}} <: ModResp
    canonical::Bool
    d::UnivariateDistribution
    devresid::V
    l::DataType
#    offset::V
    var::V
    wrkresid::V
    wrkwts::V
    wts::V
    y::V
    η::V
    μ::V
    μη::V
end

Base.var(::Type{Bernoulli},μ::FloatingPoint) = μ * (one(μ) - μ)
Base.var(::Type{Binomial},μ::FloatingPoint) =  μ * (one(μ) - μ)
Base.var(::Type{Gamma},μ::FloatingPoint) = abs2(μ)
Base.var(::Type{Normal},μ::FloatingPoint) = one(μ)
Base.var(::Type{Poisson},μ::FloatingPoint) = μ

function devresid2{T<:FloatingPoint}(::Type{Bernoulli},y::T,μ::T,wt::T)
    omy = one(T) - y
    two(y)*wt*(xlogy(y,y/μ) + xlogy(omy,omy/(one(T)-μ)))
end
devresid2{T<:FloatingPoint}(::Type{Binomial},y::T,μ::T,wt::T) = devresid2(Bernoulli,y,μ,wt)
devresid2{T<:FloatingPoint}(::Type{Gamma},y::T,μ::T,wt::T) = -two(y)*wt*(log(y/μ)-(y-μ)/μ)
devresid2{T<:FloatingPoint}(::Type{Normal},y::T,μ::T,wt::T) = wt * abs2(y-μ)
devresid2{T<:FloatingPoint}(::Type{Poisson},y::T,μ::T,wt::T) = two(y)*wt*(xlogy(y,y/μ) - (y-μ))

function updateμ!(r::GLMResp)
    @parallel for i in 1:length(r.y)
        @inbounds begin
            eta = r.η[i]
            y = r.y[i]
            r.μ[i] = (mu = linkinv(r.l,eta))
            r.μη[i] = (mueta = dμdη(r.l,eta))
            r.var[i] = var(r.d,mu)
            r.wrkresid[i] = (y - mu)/mueta
            r.devresid[i] = devresid(r.d,y,mu,r.wts[i])
        end
    end
end

## Patterned on code by Madeleine Udell in her ParallelSparseMatMul package
## function Base.At_mul_B!{T<:FloatingPoint}(y::SharedVector{T},A::SharedMatrix{T},x::SharedVector{T})
##     m,n = size(A)
##     m == length(x) && n == length(y) || throw(DimensionMismatch(""))
##     function f(y,A,x)
##         At_mul_B!(y.loc_subarr_1d,
##                   reshape(A.loc_subarr_1d,(m,div(length(A.loc_subarr_1d),m))),x)
##         1
##     end
##     @sync for p in procs(y)
##         @async remotecall_wait(p, f, y, A, x)
##     end
##     y
## end
                           
function GLMResp{V<:DenseVector}(y::V,d::UnivariateDistribution,c::Bool,η::V,μ::V,#off::V,
                                 wts::V)
    if isa(d,Binomial)
        for yy in y
            0. ≤ yy ≤ 1. || error("Binomial responses should be proportions, in [0,1]")
        end
    else
        for yy in y
            insupport(d,yy) || error("y must be in the support of d")
        end
    end
    (n = length(y)) == length(μ) == length(η) == length(wts) || throw(DimensionMismatch(""))
#    length(off) ∈ [0,n] || error("offset must have length $n or length 0")
    res = GLMResp(c,d,similar(y),canonicallink(d),#off,
                  similar(y),similar(y),similar(y),wts,y,η,μ,similar(μ))
    updateμ!(res)
    res
end

GLMResp{V<:DenseVector}(y::V,d::UnivariateDistribution,η::V) = GLMResp(y,d,true,η,similar(η),#zeros(η),
                                                                       ones(η))

canonicallink(::Type{Bernoulli}) = LogitLink
canonicallink(::Type{Binomial}) = LogitLink
canonicallink(::Type{Gamma}) = InverseLink
canonicallink(::Type{Normal}) = IdentityLink
canonicallink(::Type{Poisson}) = LogLink

mustart{T<:FloatingPoint}(::Type{Bernoulli},y::T,wt::T) = (wt*y + half(y))/(wt + one(y))
mustart{T<:FloatingPoint}(::Type{Binomial},y::T,wt::T) = mustart(Bernoulli,y,wt)
mustart{T<:FloatingPoint}(::Type{Gamma},y::T,::T) = y
mustart{T<:FloatingPoint}(::Type{Normal},y::T,::T) = y
mustart{T<:FloatingPoint}(::Type{Poisson},y::T,::T) = y + inv(convert(typeof(y),10.))

type GLMmodel{T<:FloatingPoint,D<:Distribution,L<:Link}
    Xt::DenseMatrix{T}       # transposed model matrix
    wTt::DenseMatrix{T}      # weighted transposed model matrix
    L::Cholesky{T,Matrix{T},:L} # Cholesky factor of X'WX
    vv::DenseMatrix{T}       # rows are y,o,wt,μ,η,μη,dr,wrsd,wrsp,v,wwt
    β::DenseVector{T}        # coefficient vector
    δβ::DenseVector{T}       # increment
end

const NROW = 11

function initvv(vv::SharedMatrix,D,L)
    i1d = localindexes(vv)
    for j in (1+div(first(i1d),NROW)):div(last(i1d,NROW))
        y = vv[1,j]
        o = vv[2,j] = zero(eltype(vv))     # offset
        wt = vv[3,j] = one(eltype(vv))     # prior weight
        μ = vv[4,j] = mustart(D,yy, wt)    # initial value for μ
        η = vv[5,j] = link(L,μ)            # initial η
        μη = vv[6,j] = dμdη(L,η)           # derivative
        dr = vv[7,j] = devresid2(D,y,μ,wt) # squared deviance residual
        wrsd = vv[8,j] = (y-μ)/μη          # working residual
        wrsp = vv[9,j] = wrsd + η - o      # working response
        v = vv[10,j] = var(D,μ)            # variance
        vv[11,j] = wt*abs2(μη/v)           # working weight
    end
    vv
end

function GLMmodel{T<:FloatingPoint}(X::DenseMatrix{T},y::DenseVector{T},D,L;shared=true)
    D <: Distribution && L <: Link || error("D must be a distribution and L a link")
    n,p = size(X)
    length(y) == n || throw(DimensionMismatch(""))
    vv = SharedArray(T,(NROW,n))
    copy!(view(vv,1,:),y)
    @sync for p in procs(vv)
        @async remotecall_wait(p,initvv,vv,D,L)
    end
    Xt = X'
    GLMModel{T,D,L}(Xt,similar(Xt),cholfact(Xt*Xt',:L),vv,zeros(T,p),zeros(T,p))
end
