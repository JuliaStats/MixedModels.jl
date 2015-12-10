# API-INDEX


## MODULE: MixedModels

---

## Methods [Exported]

[AIC(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__aic.1)  Akaike's Information Criterion

[BIC(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__bic.1)  Schwartz's Bayesian Information Criterion

[bootstrap(m::MixedModels.LinearMixedModel{T},  N::Integer,  saveresults::Function)](MixedModels.md#method__bootstrap.1)  Simulate `N` response vectors from `m`, refitting the model.  The function saveresults

[fixef(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__fixef.2)  The fixed-effects parameter estimates

[lmm(f::DataFrames.Formula,  fr::DataFrames.AbstractDataFrame)](MixedModels.md#method__lmm.1)  Create a `LinearMixedModel` object from a formula and data frame

[lowerbd{T}(A::LowerTriangular{T, Array{T, 2}})](MixedModels.md#method__lowerbd.1)  lower bounds on the parameters (elements in the lower triangle)

[npar(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__npar.1)  Number of parameters in the model.

[objective(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__objective.1)  `objective(m)` -> Negative twice the log-likelihood

[pwrss(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__pwrss.1)  returns the penalized residual sum-of-squares

[ranef{T}(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__ranef.1)  `ranef(m)` -> vector of matrices of random effects on the original scale

[ranef{T}(m::MixedModels.LinearMixedModel{T},  uscale)](MixedModels.md#method__ranef.2)  `ranef(m)` -> vector of matrices of random effects on the original scale

[refit!(m::MixedModels.LinearMixedModel{T},  y)](MixedModels.md#method__refit.1)  refit the model `m` with response `y`

[remat(e::Expr,  df::DataFrames.DataFrame)](MixedModels.md#method__remat.1)  `remat(e,df)` -> `ReMat`

[reml!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__reml.1)  `reml!(m,v=true)` -> m : Set m.REML to v.  If m.REML is modified, unset m.fit

[reml!(m::MixedModels.LinearMixedModel{T},  v::Bool)](MixedModels.md#method__reml.2)  `reml!(m,v=true)` -> m : Set m.REML to v.  If m.REML is modified, unset m.fit

[sdest(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__sdest.1)  `sdest(m) -> s`

[simulate!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__simulate.1)  Simulate a response vector from model `m`, and refit `m`.

[varest(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__varest.1)  returns s², the estimate of σ², the variance of the conditional distribution of Y given B

---

## Types [Exported]

[MixedModels.LinearMixedModel{T}](MixedModels.md#type__linearmixedmodel.1)  Linear mixed-effects model representation

[MixedModels.ReMat](MixedModels.md#type__remat.1)  `ReMat` - model matrix for a random-effects term

[MixedModels.ScalarReMat{T}](MixedModels.md#type__scalarremat.1)  `ScalarReMat` - a model matrix for scalar random effects

[MixedModels.VarCorr](MixedModels.md#type__varcorr.1)  `VarCorr` a type to encapsulate the information on the fitted random-effects

[MixedModels.VectorReMat{T}](MixedModels.md#type__vectorremat.1)  `VectorReMat` - a representation of a model matrix for vector-valued random effects

---

## Methods [Internal]

[LD{T}(d::Diagonal{T})](MixedModels.md#method__ld.1)  `LD(A) -> log(det(triu(A)))` for `A` diagonal, HBlkDiag, or UpperTriangular

[LT(A::MixedModels.ScalarReMat{T})](MixedModels.md#method__lt.1)  `LT(A) -> LowerTriangular`

[canonical(::Distributions.Bernoulli)](MixedModels.md#method__canonical.1)  An instance of the canonical Link type for a distribution in the exponential family

[cfactor!(A::AbstractArray{T, 2})](MixedModels.md#method__cfactor.1)  Slightly modified version of `chol!` from `julia/base/linalg/cholesky.jl`

[cfactor!(R::Array{Float64, 2})](MixedModels.md#method__cfactor.2)  `cfactor!` method for dense matrices calls `LAPACK.potrf!` directly to avoid

[chol2cor(L::LowerTriangular{T, S<:AbstractArray{T, 2}})](MixedModels.md#method__chol2cor.1)  Convert a lower Cholesky factor to a correlation matrix

[cond(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__cond.1)  Condition numbers for blocks of Λ

[densify(S)](MixedModels.md#method__densify.1)  `densify(S[,threshold])`

[densify(S,  threshold)](MixedModels.md#method__densify.2)  `densify(S[,threshold])`

[describeblocks(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__describeblocks.1)  describe the blocks of the A and R matrices

[devresid2(::Distributions.Bernoulli,  y,  μ)](MixedModels.md#method__devresid2.1)  Evaluate the squared deviance residual for a distribution instance and values of `y` and `μ`

[downdate!{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}}(C::DenseArray{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}, 2},  A::DenseArray{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}, 2})](MixedModels.md#method__downdate.1)  Subtract, in place, A'A or A'B from C

[fit!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__fit.1)  `fit!(m)` -> `m`

[fit!(m::MixedModels.LinearMixedModel{T},  verbose::Bool)](MixedModels.md#method__fit.2)  `fit!(m)` -> `m`

[fit!(m::MixedModels.LinearMixedModel{T},  verbose::Bool,  optimizer::Symbol)](MixedModels.md#method__fit.3)  `fit!(m)` -> `m`

[fixef!(v,  m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__fixef.1)  Overwrite `v` with the fixed-effects coefficients of model `m`

[fnames(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__fnames.1)  `fnames(m)` -> vector of names of grouping factors

[getindex{T}(A::LowerTriangular{T, Array{T, 2}},  s::Symbol)](MixedModels.md#method__getindex.1)  return the lower triangle as a vector (column-major ordering)

[grplevels(v::Array{T, 1})](MixedModels.md#method__grplevels.1)  `grplevels(m)` -> Vector{Int} : number of levels in each term's grouping factor

[inflate!(A::MixedModels.HBlkDiag{T})](MixedModels.md#method__inflate.1)  `inflate!(A)` is equivalent to `A += I`, without making a copy of A

[inject!(d,  s)](MixedModels.md#method__inject.1)  like `copy!` but allowing for heterogeneous matrix types

[isfit(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__isfit.1)  Predicate - whether or not the model has been fit.

[logdet(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__logdet.1)  returns `log(det(Λ'Z'ZΛ + I))`

[lrt(mods::MixedModels.LinearMixedModel{T}...)](MixedModels.md#method__lrt.1)  Likelihood ratio test of one or more models

[model_response(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__model_response.1)  extract the response (as a reference)

[mustart!{T}(μ::Array{T, 1},  d::Distributions.Distribution{F<:Distributions.VariateForm, S<:Distributions.ValueSupport},  y::Array{T, 1},  wt::Array{T, 1})](MixedModels.md#method__mustart.1)  In-place modification of μ to starting values from d, y and wt

[mustart{T<:AbstractFloat}(::Distributions.Bernoulli,  y::T<:AbstractFloat,  wt::T<:AbstractFloat)](MixedModels.md#method__mustart.2)  Initial μ value from the response and the weight

[regenerateAend!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__regenerateaend.1)  Regenerate the last column of `m.A` from `m.trms`

[resetθ!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__reset952.1)  Reset the value of `m.θ` to the initial values

[rowlengths(L::LowerTriangular{T, S<:AbstractArray{T, 2}})](MixedModels.md#method__rowlengths.1)  `rowlengths(L)` -> a vector of the Euclidean lengths of the rows of `L`

[setindex!{T}(A::LowerTriangular{T, Array{T, 2}},  v::AbstractArray{T, 1},  s::Symbol)](MixedModels.md#method__setindex.1)  set the lower triangle of A to v using column-major ordering

[sqrtpwrss(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__sqrtpwrss.1)  returns the square root of the penalized residual sum-of-squares

[std(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__std.1)  `std(m) -> Vector{Vector{Float64}}` estimated standard deviations of variance components

[tscale!(A::LowerTriangular{T, S<:AbstractArray{T, 2}},  B::MixedModels.HBlkDiag{T})](MixedModels.md#method__tscale.1)  scale B using the implicit expansion of A to a homogeneous block diagonal

[unscaledre!(y::AbstractArray{T, 1},  M::MixedModels.ScalarReMat{T},  L::LowerTriangular{T, S<:AbstractArray{T, 2}})](MixedModels.md#method__unscaledre.1)  Add unscaled random effects to y

[vcov(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__vcov.1)  returns the estimated variance-covariance matrix of the fixed-effects estimator

[ylogydμ{T<:AbstractFloat}(y::T<:AbstractFloat,  μ::T<:AbstractFloat)](MixedModels.md#method__ylogyd956.1)  Evaluate `y*log(y/μ)` with the correct limit as `y` approaches zero from above

---

## Types [Internal]

[MixedModels.OptSummary](MixedModels.md#type__optsummary.1)  Summary of an NLopt optimization

