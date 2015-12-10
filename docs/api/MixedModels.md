# MixedModels


## Methods [Exported]

---

<a id="method__aic.1" class="lexicon_definition"></a>
#### AIC(m::MixedModels.LinearMixedModel{T}) [¶](#method__aic.1)
Akaike's Information Criterion


*source:*
[MixedModels/src/pls.jl:215](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L215)

---

<a id="method__bic.1" class="lexicon_definition"></a>
#### BIC(m::MixedModels.LinearMixedModel{T}) [¶](#method__bic.1)
Schwartz's Bayesian Information Criterion


*source:*
[MixedModels/src/pls.jl:220](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L220)

---

<a id="method__bootstrap.1" class="lexicon_definition"></a>
#### bootstrap(m::MixedModels.LinearMixedModel{T},  N::Integer,  saveresults::Function) [¶](#method__bootstrap.1)
Simulate `N` response vectors from `m`, refitting the model.  The function saveresults
is called after each refit.

To save space the last column of `m.trms[end]`, which is the response vector, is overwritten
by each simulation.  The original response is restored before returning.


*source:*
[MixedModels/src/bootstrap.jl:8](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/bootstrap.jl#L8)

---

<a id="method__fixef.2" class="lexicon_definition"></a>
#### fixef(m::MixedModels.LinearMixedModel{T}) [¶](#method__fixef.2)
The fixed-effects parameter estimates


*source:*
[MixedModels/src/pls.jl:238](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L238)

---

<a id="method__lmm.1" class="lexicon_definition"></a>
#### lmm(f::DataFrames.Formula,  fr::DataFrames.AbstractDataFrame) [¶](#method__lmm.1)
Create a `LinearMixedModel` object from a formula and data frame


*source:*
[MixedModels/src/pls.jl:95](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L95)

---

<a id="method__lowerbd.1" class="lexicon_definition"></a>
#### lowerbd{T}(A::LowerTriangular{T, Array{T, 2}}) [¶](#method__lowerbd.1)
lower bounds on the parameters (elements in the lower triangle)


*source:*
[MixedModels/src/paramlowertriangular.jl:45](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/paramlowertriangular.jl#L45)

---

<a id="method__npar.1" class="lexicon_definition"></a>
#### npar(m::MixedModels.LinearMixedModel{T}) [¶](#method__npar.1)
Number of parameters in the model.

Note that `size(m.trms[end],2)` is `length(coef(m)) + 1`, thereby accounting
for the scale parameter, σ, that is profiled out.


*source:*
[MixedModels/src/pls.jl:250](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L250)

---

<a id="method__objective.1" class="lexicon_definition"></a>
#### objective(m::MixedModels.LinearMixedModel{T}) [¶](#method__objective.1)
`objective(m)` -> Negative twice the log-likelihood


*source:*
[MixedModels/src/pls.jl:210](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L210)

---

<a id="method__pwrss.1" class="lexicon_definition"></a>
#### pwrss(m::MixedModels.LinearMixedModel{T}) [¶](#method__pwrss.1)
returns the penalized residual sum-of-squares


*source:*
[MixedModels/src/pls.jl:280](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L280)

---

<a id="method__ranef.1" class="lexicon_definition"></a>
#### ranef{T}(m::MixedModels.LinearMixedModel{T}) [¶](#method__ranef.1)
`ranef(m)` -> vector of matrices of random effects on the original scale
`ranef(m,true)` -> vector of matrices of random effects on the U scale


*source:*
[MixedModels/src/pls.jl:355](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L355)

---

<a id="method__ranef.2" class="lexicon_definition"></a>
#### ranef{T}(m::MixedModels.LinearMixedModel{T},  uscale) [¶](#method__ranef.2)
`ranef(m)` -> vector of matrices of random effects on the original scale
`ranef(m,true)` -> vector of matrices of random effects on the U scale


*source:*
[MixedModels/src/pls.jl:355](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L355)

---

<a id="method__refit.1" class="lexicon_definition"></a>
#### refit!(m::MixedModels.LinearMixedModel{T},  y) [¶](#method__refit.1)
refit the model `m` with response `y`


*source:*
[MixedModels/src/bootstrap.jl:93](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/bootstrap.jl#L93)

---

<a id="method__remat.1" class="lexicon_definition"></a>
#### remat(e::Expr,  df::DataFrames.DataFrame) [¶](#method__remat.1)
`remat(e,df)` -> `ReMat`

A factory for `ReMat` objects constructed from a random-effects term and a
`DataFrame`


*source:*
[MixedModels/src/remat.jl:37](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/remat.jl#L37)

---

<a id="method__reml.1" class="lexicon_definition"></a>
#### reml!(m::MixedModels.LinearMixedModel{T}) [¶](#method__reml.1)
`reml!(m,v=true)` -> m : Set m.REML to v.  If m.REML is modified, unset m.fit


*source:*
[MixedModels/src/pls.jl:395](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L395)

---

<a id="method__reml.2" class="lexicon_definition"></a>
#### reml!(m::MixedModels.LinearMixedModel{T},  v::Bool) [¶](#method__reml.2)
`reml!(m,v=true)` -> m : Set m.REML to v.  If m.REML is modified, unset m.fit


*source:*
[MixedModels/src/pls.jl:395](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L395)

---

<a id="method__sdest.1" class="lexicon_definition"></a>
#### sdest(m::MixedModels.LinearMixedModel{T}) [¶](#method__sdest.1)
`sdest(m) -> s`

returns `s`, the estimate of σ, the standard deviation of the per-observation noise


*source:*
[MixedModels/src/pls.jl:263](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L263)

---

<a id="method__simulate.1" class="lexicon_definition"></a>
#### simulate!(m::MixedModels.LinearMixedModel{T}) [¶](#method__simulate.1)
Simulate a response vector from model `m`, and refit `m`.

- m, LinearMixedModel.
- β, fixed effects parameter vector
- σ, standard deviation of the per-observation random noise term
- σv, vector of standard deviations for the scalar random effects.


*source:*
[MixedModels/src/bootstrap.jl:76](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/bootstrap.jl#L76)

---

<a id="method__varest.1" class="lexicon_definition"></a>
#### varest(m::MixedModels.LinearMixedModel{T}) [¶](#method__varest.1)
returns s², the estimate of σ², the variance of the conditional distribution of Y given B


*source:*
[MixedModels/src/pls.jl:275](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L275)

## Types [Exported]

---

<a id="type__linearmixedmodel.1" class="lexicon_definition"></a>
#### MixedModels.LinearMixedModel{T} [¶](#type__linearmixedmodel.1)
Linear mixed-effects model representation

- `mf` the model frame, mostly used to get the `terms` component for labelling fixed effects
- `trms` is a length `nt` vector of model matrices. Its last element is `hcat(X,y)`
- `Λ` is a length `nt - 1` vector of lower triangular matrices
- `weights` a vector of weights
- `A` is an `nt × nt` symmetric matrix of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
- `R`, also a `nt × nt` matrix of matrices, is the upper Cholesky factor of `Λ'AΛ+I`
- `opt`, an OptSummary object


*source:*
[MixedModels/src/pls.jl:26](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L26)

---

<a id="type__remat.1" class="lexicon_definition"></a>
#### MixedModels.ReMat [¶](#type__remat.1)
`ReMat` - model matrix for a random-effects term


*source:*
[MixedModels/src/remat.jl:4](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/remat.jl#L4)

---

<a id="type__scalarremat.1" class="lexicon_definition"></a>
#### MixedModels.ScalarReMat{T} [¶](#type__scalarremat.1)
`ScalarReMat` - a model matrix for scalar random effects

The matrix is represented by the grouping factor, `f`, and a vector `z`.


*source:*
[MixedModels/src/remat.jl:11](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/remat.jl#L11)

---

<a id="type__varcorr.1" class="lexicon_definition"></a>
#### MixedModels.VarCorr [¶](#type__varcorr.1)
`VarCorr` a type to encapsulate the information on the fitted random-effects
variance-covariance matrices.

The main purpose is to isolate the logic in the show method.


*source:*
[MixedModels/src/pls.jl:442](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L442)

---

<a id="type__vectorremat.1" class="lexicon_definition"></a>
#### MixedModels.VectorReMat{T} [¶](#type__vectorremat.1)
`VectorReMat` - a representation of a model matrix for vector-valued random effects

The matrix is represented by the grouping factor, `f`, and the transposed raw
model matrix, `z`.


*source:*
[MixedModels/src/remat.jl:24](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/remat.jl#L24)


## Methods [Internal]

---

<a id="method__ld.1" class="lexicon_definition"></a>
#### LD{T}(d::Diagonal{T}) [¶](#method__ld.1)
`LD(A) -> log(det(triu(A)))` for `A` diagonal, HBlkDiag, or UpperTriangular


*source:*
[MixedModels/src/logdet.jl:4](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/logdet.jl#L4)

---

<a id="method__lt.1" class="lexicon_definition"></a>
#### LT(A::MixedModels.ScalarReMat{T}) [¶](#method__lt.1)
`LT(A) -> LowerTriangular`

Create as a lower triangular matrix compatible with the blocks of `A`
and initialized to the identity.


*source:*
[MixedModels/src/paramlowertriangular.jl:96](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/paramlowertriangular.jl#L96)

---

<a id="method__canonical.1" class="lexicon_definition"></a>
#### canonical(::Distributions.Bernoulli) [¶](#method__canonical.1)
An instance of the canonical Link type for a distribution in the exponential family


*source:*
[MixedModels/src/GLMM/glmtools.jl:52](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/GLMM/glmtools.jl#L52)

---

<a id="method__cfactor.1" class="lexicon_definition"></a>
#### cfactor!(A::AbstractArray{T, 2}) [¶](#method__cfactor.1)
Slightly modified version of `chol!` from `julia/base/linalg/cholesky.jl`

Uses `inject!` (as opposed to `copy!`), `downdate!` (as opposed to `syrk!`
    or `gemm!`) and recursive calls to `cfactor!`,


*source:*
[MixedModels/src/cfactor.jl:7](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/cfactor.jl#L7)

---

<a id="method__cfactor.2" class="lexicon_definition"></a>
#### cfactor!(R::Array{Float64, 2}) [¶](#method__cfactor.2)
`cfactor!` method for dense matrices calls `LAPACK.potrf!` directly to avoid
errors being thrown when `R` is computationally singular


*source:*
[MixedModels/src/cfactor.jl:34](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/cfactor.jl#L34)

---

<a id="method__chol2cor.1" class="lexicon_definition"></a>
#### chol2cor(L::LowerTriangular{T, S<:AbstractArray{T, 2}}) [¶](#method__chol2cor.1)
Convert a lower Cholesky factor to a correlation matrix


*source:*
[MixedModels/src/pls.jl:290](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L290)

---

<a id="method__cond.1" class="lexicon_definition"></a>
#### cond(m::MixedModels.LinearMixedModel{T}) [¶](#method__cond.1)
Condition numbers for blocks of Λ


*source:*
[MixedModels/src/pls.jl:285](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L285)

---

<a id="method__densify.1" class="lexicon_definition"></a>
#### densify(S) [¶](#method__densify.1)
`densify(S[,threshold])`

Convert sparse `S` to `Diagonal` if S is diagonal
Convert sparse `S` to dense if the proportion of nonzeros exceeds `threshold`.
A no-op for other matrix types.


*source:*
[MixedModels/src/densify.jl:8](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/densify.jl#L8)

---

<a id="method__densify.2" class="lexicon_definition"></a>
#### densify(S,  threshold) [¶](#method__densify.2)
`densify(S[,threshold])`

Convert sparse `S` to `Diagonal` if S is diagonal
Convert sparse `S` to dense if the proportion of nonzeros exceeds `threshold`.
A no-op for other matrix types.


*source:*
[MixedModels/src/densify.jl:8](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/densify.jl#L8)

---

<a id="method__describeblocks.1" class="lexicon_definition"></a>
#### describeblocks(m::MixedModels.LinearMixedModel{T}) [¶](#method__describeblocks.1)
describe the blocks of the A and R matrices


*source:*
[MixedModels/src/pls.jl:509](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L509)

---

<a id="method__devresid2.1" class="lexicon_definition"></a>
#### devresid2(::Distributions.Bernoulli,  y,  μ) [¶](#method__devresid2.1)
Evaluate the squared deviance residual for a distribution instance and values of `y` and `μ`


*source:*
[MixedModels/src/GLMM/glmtools.jl:74](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/GLMM/glmtools.jl#L74)

---

<a id="method__downdate.1" class="lexicon_definition"></a>
#### downdate!{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}}(C::DenseArray{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}, 2},  A::DenseArray{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}, 2}) [¶](#method__downdate.1)
Subtract, in place, A'A or A'B from C


*source:*
[MixedModels/src/cfactor.jl:48](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/cfactor.jl#L48)

---

<a id="method__fit.1" class="lexicon_definition"></a>
#### fit!(m::MixedModels.LinearMixedModel{T}) [¶](#method__fit.1)
`fit!(m)` -> `m`

Optimize the objective using an NLopt optimizer.


*source:*
[MixedModels/src/pls.jl:148](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L148)

---

<a id="method__fit.2" class="lexicon_definition"></a>
#### fit!(m::MixedModels.LinearMixedModel{T},  verbose::Bool) [¶](#method__fit.2)
`fit!(m)` -> `m`

Optimize the objective using an NLopt optimizer.


*source:*
[MixedModels/src/pls.jl:148](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L148)

---

<a id="method__fit.3" class="lexicon_definition"></a>
#### fit!(m::MixedModels.LinearMixedModel{T},  verbose::Bool,  optimizer::Symbol) [¶](#method__fit.3)
`fit!(m)` -> `m`

Optimize the objective using an NLopt optimizer.


*source:*
[MixedModels/src/pls.jl:148](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L148)

---

<a id="method__fixef.1" class="lexicon_definition"></a>
#### fixef!(v,  m::MixedModels.LinearMixedModel{T}) [¶](#method__fixef.1)
Overwrite `v` with the fixed-effects coefficients of model `m`


*source:*
[MixedModels/src/pls.jl:230](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L230)

---

<a id="method__fnames.1" class="lexicon_definition"></a>
#### fnames(m::MixedModels.LinearMixedModel{T}) [¶](#method__fnames.1)
`fnames(m)` -> vector of names of grouping factors


*source:*
[MixedModels/src/pls.jl:313](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L313)

---

<a id="method__getindex.1" class="lexicon_definition"></a>
#### getindex{T}(A::LowerTriangular{T, Array{T, 2}},  s::Symbol) [¶](#method__getindex.1)
return the lower triangle as a vector (column-major ordering)


*source:*
[MixedModels/src/paramlowertriangular.jl:7](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/paramlowertriangular.jl#L7)

---

<a id="method__grplevels.1" class="lexicon_definition"></a>
#### grplevels(v::Array{T, 1}) [¶](#method__grplevels.1)
`grplevels(m)` -> Vector{Int} : number of levels in each term's grouping factor


*source:*
[MixedModels/src/pls.jl:318](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L318)

---

<a id="method__inflate.1" class="lexicon_definition"></a>
#### inflate!(A::MixedModels.HBlkDiag{T}) [¶](#method__inflate.1)
`inflate!(A)` is equivalent to `A += I`, without making a copy of A

Even if `A += I` did not make a copy, this function is needed for the special
behavior on the `HBlkDiag` type.


*source:*
[MixedModels/src/inflate.jl:7](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/inflate.jl#L7)

---

<a id="method__inject.1" class="lexicon_definition"></a>
#### inject!(d,  s) [¶](#method__inject.1)
like `copy!` but allowing for heterogeneous matrix types


*source:*
[MixedModels/src/inject.jl:4](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/inject.jl#L4)

---

<a id="method__isfit.1" class="lexicon_definition"></a>
#### isfit(m::MixedModels.LinearMixedModel{T}) [¶](#method__isfit.1)
Predicate - whether or not the model has been fit.


*source:*
[MixedModels/src/pls.jl:324](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L324)

---

<a id="method__logdet.1" class="lexicon_definition"></a>
#### logdet(m::MixedModels.LinearMixedModel{T}) [¶](#method__logdet.1)
returns `log(det(Λ'Z'ZΛ + I))`


*source:*
[MixedModels/src/logdet.jl:35](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/logdet.jl#L35)

---

<a id="method__lrt.1" class="lexicon_definition"></a>
#### lrt(mods::MixedModels.LinearMixedModel{T}...) [¶](#method__lrt.1)
Likelihood ratio test of one or more models


*source:*
[MixedModels/src/pls.jl:331](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L331)

---

<a id="method__model_response.1" class="lexicon_definition"></a>
#### model_response(m::MixedModels.LinearMixedModel{T}) [¶](#method__model_response.1)
extract the response (as a reference)

In Julia 0.5 this can be a one-liner `m.trms[end][:,end]`


*source:*
[MixedModels/src/bootstrap.jl:103](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/bootstrap.jl#L103)

---

<a id="method__mustart.1" class="lexicon_definition"></a>
#### mustart!{T}(μ::Array{T, 1},  d::Distributions.Distribution{F<:Distributions.VariateForm, S<:Distributions.ValueSupport},  y::Array{T, 1},  wt::Array{T, 1}) [¶](#method__mustart.1)
In-place modification of μ to starting values from d, y and wt


*source:*
[MixedModels/src/GLMM/glmtools.jl:92](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/GLMM/glmtools.jl#L92)

---

<a id="method__mustart.2" class="lexicon_definition"></a>
#### mustart{T<:AbstractFloat}(::Distributions.Bernoulli,  y::T<:AbstractFloat,  wt::T<:AbstractFloat) [¶](#method__mustart.2)
Initial μ value from the response and the weight


*source:*
[MixedModels/src/GLMM/glmtools.jl:83](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/GLMM/glmtools.jl#L83)

---

<a id="method__regenerateaend.1" class="lexicon_definition"></a>
#### regenerateAend!(m::MixedModels.LinearMixedModel{T}) [¶](#method__regenerateaend.1)
Regenerate the last column of `m.A` from `m.trms`

This should be called after updating parts of `m.trms[end]`, typically the response.


*source:*
[MixedModels/src/bootstrap.jl:25](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/bootstrap.jl#L25)

---

<a id="method__reset952.1" class="lexicon_definition"></a>
#### resetθ!(m::MixedModels.LinearMixedModel{T}) [¶](#method__reset952.1)
Reset the value of `m.θ` to the initial values


*source:*
[MixedModels/src/bootstrap.jl:37](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/bootstrap.jl#L37)

---

<a id="method__rowlengths.1" class="lexicon_definition"></a>
#### rowlengths(L::LowerTriangular{T, S<:AbstractArray{T, 2}}) [¶](#method__rowlengths.1)
`rowlengths(L)` -> a vector of the Euclidean lengths of the rows of `L`

used in `chol2cor`


*source:*
[MixedModels/src/paramlowertriangular.jl:63](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/paramlowertriangular.jl#L63)

---

<a id="method__setindex.1" class="lexicon_definition"></a>
#### setindex!{T}(A::LowerTriangular{T, Array{T, 2}},  v::AbstractArray{T, 1},  s::Symbol) [¶](#method__setindex.1)
set the lower triangle of A to v using column-major ordering


*source:*
[MixedModels/src/paramlowertriangular.jl:23](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/paramlowertriangular.jl#L23)

---

<a id="method__sqrtpwrss.1" class="lexicon_definition"></a>
#### sqrtpwrss(m::MixedModels.LinearMixedModel{T}) [¶](#method__sqrtpwrss.1)
returns the square root of the penalized residual sum-of-squares

This is the bottom right element of the bottom right block of m.R


*source:*
[MixedModels/src/pls.jl:270](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L270)

---

<a id="method__std.1" class="lexicon_definition"></a>
#### std(m::MixedModels.LinearMixedModel{T}) [¶](#method__std.1)
`std(m) -> Vector{Vector{Float64}}` estimated standard deviations of variance components


*source:*
[MixedModels/src/pls.jl:434](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L434)

---

<a id="method__tscale.1" class="lexicon_definition"></a>
#### tscale!(A::LowerTriangular{T, S<:AbstractArray{T, 2}},  B::MixedModels.HBlkDiag{T}) [¶](#method__tscale.1)
scale B using the implicit expansion of A to a homogeneous block diagonal


*source:*
[MixedModels/src/paramlowertriangular.jl:71](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/paramlowertriangular.jl#L71)

---

<a id="method__unscaledre.1" class="lexicon_definition"></a>
#### unscaledre!(y::AbstractArray{T, 1},  M::MixedModels.ScalarReMat{T},  L::LowerTriangular{T, S<:AbstractArray{T, 2}}) [¶](#method__unscaledre.1)
Add unscaled random effects to y


*source:*
[MixedModels/src/bootstrap.jl:47](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/bootstrap.jl#L47)

---

<a id="method__vcov.1" class="lexicon_definition"></a>
#### vcov(m::MixedModels.LinearMixedModel{T}) [¶](#method__vcov.1)
returns the estimated variance-covariance matrix of the fixed-effects estimator


*source:*
[MixedModels/src/pls.jl:501](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L501)

---

<a id="method__ylogyd956.1" class="lexicon_definition"></a>
#### ylogydμ{T<:AbstractFloat}(y::T<:AbstractFloat,  μ::T<:AbstractFloat) [¶](#method__ylogyd956.1)
Evaluate `y*log(y/μ)` with the correct limit as `y` approaches zero from above


*source:*
[MixedModels/src/GLMM/glmtools.jl:67](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/GLMM/glmtools.jl#L67)

## Types [Internal]

---

<a id="type__optsummary.1" class="lexicon_definition"></a>
#### MixedModels.OptSummary [¶](#type__optsummary.1)
Summary of an NLopt optimization


*source:*
[MixedModels/src/pls.jl:4](https://github.com/dmbates/MixedModels.jl/tree/f42452aa03213ace2d9d898088abbfeb5b9ee850/src/pls.jl#L4)

