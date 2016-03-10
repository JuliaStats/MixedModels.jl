# API-INDEX


## MODULE: MixedModels

---

## Methods [Exported]

[bootstrap(m::MixedModels.LinearMixedModel{T},  N::Integer,  saveresults::Function)](MixedModels.md#method__bootstrap.1)  Simulate `N` response vectors from `m`, refitting the model.  The function saveresults

[fixef(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__fixef.2)      fixef(m)

[lmm(f::DataFrames.Formula,  fr::DataFrames.AbstractDataFrame)](MixedModels.md#method__lmm.1)      lmm(form, frm)

[lmm(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__lmm.2)      lmm(m::MixedModel)

[lowerbd(m::MixedModels.MixedModel)](MixedModels.md#method__lowerbd.1)      lowerbd(m::MixedModel)

[lowerbd{T}(A::LowerTriangular{T, Array{T, 2}})](MixedModels.md#method__lowerbd.2)  lower bounds on the parameters (elements in the lower triangle)

[objective(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__objective.1)      objective(m)

[pwrss(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__pwrss.1)      pwrss(m::LinearMixedModel)

[ranef(m::MixedModels.MixedModel)](MixedModels.md#method__ranef.2)      ranef(m)

[ranef(m::MixedModels.MixedModel,  uscale)](MixedModels.md#method__ranef.3)      ranef(m)

[refit!(m::MixedModels.LinearMixedModel{T},  y)](MixedModels.md#method__refit.1)  refit the model `m` with response `y`

[remat(e::Expr,  df::DataFrames.DataFrame)](MixedModels.md#method__remat.1)  `remat(e,df)` -> `ReMat`

[reml!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__reml.1)  `reml!(m,v=true)` -> m : Set m.REML to v.  If m.REML is modified, unset m.fit

[reml!(m::MixedModels.LinearMixedModel{T},  v::Bool)](MixedModels.md#method__reml.2)  `reml!(m,v=true)` -> m : Set m.REML to v.  If m.REML is modified, unset m.fit

[sdest(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__sdest.1)      sdest(m)

[simulate!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__simulate.1)  Simulate a response vector from model `m`, and refit `m`.

[varest(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__varest.1)      varest(m::LinearMixedModel)

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

[cfactor!(A::AbstractArray{T, 2})](MixedModels.md#method__cfactor.1)  Slightly modified version of `chol!` from `julia/base/linalg/cholesky.jl`

[cfactor!(R::Array{Float64, 2})](MixedModels.md#method__cfactor.2)  `cfactor!` method for dense matrices calls `LAPACK.potrf!` directly to avoid

[chol2cor(L::LowerTriangular{T, S<:AbstractArray{T, 2}})](MixedModels.md#method__chol2cor.1)  Convert a lower Cholesky factor to a correlation matrix

[cond(m::MixedModels.MixedModel)](MixedModels.md#method__cond.1)      cond(m::MixedModel)

[densify(S)](MixedModels.md#method__densify.1)  `densify(S[,threshold])`

[densify(S,  threshold)](MixedModels.md#method__densify.2)  `densify(S[,threshold])`

[describeblocks(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__describeblocks.1)  describe the blocks of the A and R matrices

[df(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__df.1)  Number of parameters in the model.

[downdate!{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}}(C::DenseArray{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}, 2},  A::DenseArray{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}, 2})](MixedModels.md#method__downdate.1)  Subtract, in place, A'A or A'B from C

[fit!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__fit.1)  `fit!(m)` -> `m`

[fit!(m::MixedModels.LinearMixedModel{T},  verbose::Bool)](MixedModels.md#method__fit.2)  `fit!(m)` -> `m`

[fit!(m::MixedModels.LinearMixedModel{T},  verbose::Bool,  optimizer::Symbol)](MixedModels.md#method__fit.3)  `fit!(m)` -> `m`

[fixef!(v,  m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__fixef.1)      fixef!(v, m)

[fnames(m::MixedModels.MixedModel)](MixedModels.md#method__fnames.1)      fnames(m::MixedModel)

[getindex{T}(A::LowerTriangular{T, Array{T, 2}},  s::Symbol)](MixedModels.md#method__getindex.1)  return the lower triangle as a vector (column-major ordering)

[grplevels(m::MixedModels.MixedModel)](MixedModels.md#method__grplevels.1)  `grplevels(m)` -> Vector{Int} : number of levels in each term's grouping factor

[inflate!(A::MixedModels.HBlkDiag{T})](MixedModels.md#method__inflate.1)  `inflate!(A)` is equivalent to `A += I`, without making a copy of A

[inject!(d,  s)](MixedModels.md#method__inject.1)  like `copy!` but allowing for heterogeneous matrix types

[isfit(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__isfit.1)      isfit(m)

[logdet(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__logdet.1)  returns `log(det(Λ'Z'ZΛ + I))`

[lrt(mods::MixedModels.LinearMixedModel{T}...)](MixedModels.md#method__lrt.1)  Likelihood ratio test of one or more models

[model_response(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__model_response.1)  extract the response (as a reference)

[ranef!(v::Array{T, 1},  m::MixedModels.MixedModel,  uscale)](MixedModels.md#method__ranef.1)      ranef!(v, m, uscale)

[reevaluateAend!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__reevaluateaend.1)      reevaluateAend!(m)

[resetθ!(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__reset952.1)      resetθ!(m)

[reterms(m::MixedModels.MixedModel)](MixedModels.md#method__reterms.1)      reterms(m)

[rowlengths(L::LowerTriangular{T, S<:AbstractArray{T, 2}})](MixedModels.md#method__rowlengths.1)  `rowlengths(L)` -> a vector of the Euclidean lengths of the rows of `L`

[setindex!{T}(A::LowerTriangular{T, Array{T, 2}},  v::AbstractArray{T, 1},  s::Symbol)](MixedModels.md#method__setindex.1)  set the lower triangle of A to v using column-major ordering

[sqrtpwrss(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__sqrtpwrss.1)  returns the square root of the penalized residual sum-of-squares

[std(m::MixedModels.MixedModel)](MixedModels.md#method__std.1)      std(m)

[tscale!(A::LowerTriangular{T, S<:AbstractArray{T, 2}},  B::MixedModels.HBlkDiag{T})](MixedModels.md#method__tscale.1)  scale B using the implicit expansion of A to a homogeneous block diagonal

[unscaledre!(y::AbstractArray{T, 1},  M::MixedModels.ScalarReMat{T},  L::LowerTriangular{T, S<:AbstractArray{T, 2}},  u::DenseArray{T, 2})](MixedModels.md#method__unscaledre.1)      unscaledre!(y, M, L, u)

[vcov(m::MixedModels.LinearMixedModel{T})](MixedModels.md#method__vcov.1)  returns the estimated variance-covariance matrix of the fixed-effects estimator

---

## Types [Internal]

[MixedModels.OptSummary](MixedModels.md#type__optsummary.1)  Summary of an NLopt optimization

