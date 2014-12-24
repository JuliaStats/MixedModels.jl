## Definition of Link types and methods for link, invlink and μη, the derivative of μ w.r.t. η

abstract Link

immutable CauchitLink <: Link end
immutable CloglogLink  <: Link end
immutable IdentityLink <: Link end
immutable InverseLink  <: Link end
immutable LogitLink <: Link end
immutable LogLink <: Link end
immutable ProbitLink <: Link end
immutable SqrtLink <: Link end

const cc = Cauchy()
const nn = Normal()

link(::CauchitLink,μ) = quantile(cc,μ)
linkinv(::CauchitLink,η) = cdf(cc,η)
μη(::CauchitLink,η) = pdf(cc,η)

link(::CloglogLink,μ) = log(-log(one(μ) - μ))
linkinv(::CloglogLink,η) = -expm1(-exp(η))
μη(::CloglogLink,η) = exp(η)*exp(-exp(η))

link(::IdentityLink,μ) = μ
invlink(::IdentityLink,η) = η
μη(::IdentityLink,η) = one(η)

link(::InverseLink,μ) = inv(μ)
invlink(::InverseLink,η) = inv(η)
μη(::InverseLink,η) = -inv(abs2(η))

link(::LogitLink,μ) = log(μ/(one(μ)-μ))
invlink(::LogitLink,η) = inv(one(η) + exp(-η))
μη(::LogitLink,η) = (ee = exp(-η); ee/abs2(one(η)+ee))

link(::LogLink,μ) = log(μ)
invlink(::LogLink,η) = exp(η)
μη(::LogLink,η) = exp(η)

link(::ProbitLink,μ) = quantile(nn,μ)
linkinv(::ProbitLink,η) = cdf(nn,η)
μη(l::ProbitLink,η) = pdf(nn,η)

link(::SqrtLink,μ) = √μ
linkinv(::SqrtLink,η) = abs2(η)
μη(::SqrtLink,η) = η + η

@doc """
An instance of the canonical Link type for a distribution in the exponential family
""" ->
canonical(::Bernoulli) = LogitLink()
canonical(::Binomial) = LogitLink()
canonical(::Gamma) = InverseLink()
canonical(::Normal) = IdentityLink()
canonical(::Poisson) = LogLink()

varfunc(::Bernoulli,μ) = μ*(one(μ)-μ)
varfunc(::Binomial,μ) = μ*(one(μ)-μ)
varfunc(::Gamma,μ) = abs2(μ)
varfunc(::Normal,μ) = one(μ)
varfunc(::Poisson,μ) = μ

@doc """
Evaluate `y*log(y/μ)` with the correct limit as `y` approaches zero from above
"""->
ylogydμ{T<:FloatingPoint}(y::T,μ::T) = y > zero(T) ? y*log(y/μ) : zero(T)

two(y) = one(y) + one(y)                # equivalent to convert(typeof(y),2)

@doc """
Evaluate the squared deviance residual for a distribution instance and values of `y` and `μ`
"""->
devresid2(::Bernoulli,y,μ) = two(y)*(ylogydμ(y,μ) + ylogydμ(one(y)-y,one(μ)-μ))
devresid2(::Binomial,y,μ) = devresid2(Bernoulli(),y,μ)
devresid2(::Gamma,y,μ) =  two(y)*((y-μ)/μ - (y == zero(y) ? y : log(y/μ)))
devresid2(::Normal,y,μ) = abs2(y-μ)
devresid2(::Poisson,y,μ) = two(y)*(ylogydμ(y,μ)-(y-μ))

@doc """
Initial μ value from the response and the weight
""" ->
mustart{T<:FloatingPoint}(::Bernoulli,y::T,wt::T) = (wt*y + convert(T,0.5))/(wt + one(T))
mustart{T<:FloatingPoint}(::Binomial,y::T,wt::T) = (wt*y + convert(T,0.5))/(wt + one(T))
mustart(::Gamma,y,wt) = y
mustart(::Normal,y,wt) = y
mustart{T<:FloatingPoint}(::Poisson,y::T,wt::T) = convert(T,1.1)*y

@doc """
In-place modification of μ to starting values from d, y and wt
"""
function mustart!{T}(μ::Vector{T},d::Distribution,y::Vector{T},wt::Vector{T})
    (n = length(μ)) == length(y) == length(wt) || throw(DimensionMismatch(""))
    @inbounds for i in 1:n
        μ[i] = mustart(d,y[i],wt[i])
    end
    μ
end
