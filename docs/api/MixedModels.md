# MixedModels


## Methods [Exported]

---

<a id="method__bootstrap.1" class="lexicon_definition"></a>
#### bootstrap(m::MixedModels.LinearMixedModel{T},  N::Integer,  saveresults::Function) [¶](#method__bootstrap.1)
Simulate `N` response vectors from `m`, refitting the model.  The function saveresults
is called after each refit.

To save space the last column of `m.trms[end]`, which is the response vector, is overwritten
by each simulation.  The original response is restored before returning.


*source:*
[MixedModels/src/bootstrap.jl:8](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/bootstrap.jl#L8)

---

<a id="method__fixef.2" class="lexicon_definition"></a>
#### fixef(m::MixedModels.LinearMixedModel{T}) [¶](#method__fixef.2)
    fixef(m)

Args:

- `m`: a `MixedModel`

Returns:
  A `Vector` of estimates of the fixed-effects parameters of `m`


*source:*
[MixedModels/src/pls.jl:229](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L229)

---

<a id="method__lmm.1" class="lexicon_definition"></a>
#### lmm(f::DataFrames.Formula,  fr::DataFrames.AbstractDataFrame) [¶](#method__lmm.1)
    lmm(form, frm)
    lmm(form, frm, weights)

Args:

- `form`: a `DataFrames:Formula` containing fixed-effects and random-effects terms
- `frm`: a `DataFrame` in which to evaluate `form`
- `weights`: an optional vector of prior weights in the model.  Defaults to unit weights.

Returns:
  A `LinearMixedModel`.

Notes:
  The return value is ready to be `fit!` but has not yet been fit.


*source:*
[MixedModels/src/pls.jl:110](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L110)

---

<a id="method__lmm.2" class="lexicon_definition"></a>
#### lmm(m::MixedModels.LinearMixedModel{T}) [¶](#method__lmm.2)
    lmm(m::MixedModel)

Extract the `LinearMixedModel` from a `MixedModel`.  If `m` is itself a `LinearMixedModel` this simply returns `m`.
If `m` is a `GeneralizedLinearMixedModel` this returns its `LMM` member.

Args:

- `m`: a `MixedModel`

Returns:
  A `LinearMixedModel`, either `m` itself or the `LMM` member of `m`


*source:*
[MixedModels/src/mixedmodel.jl:16](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L16)

---

<a id="method__lowerbd.1" class="lexicon_definition"></a>
#### lowerbd(m::MixedModels.MixedModel) [¶](#method__lowerbd.1)
    lowerbd(m::MixedModel)

Args:

- `m`: a `MixedModel`

Returns:
  A `Vector` of lower bounds on the covariance parameter vector `m[:θ]`


*source:*
[MixedModels/src/mixedmodel.jl:117](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L117)

---

<a id="method__lowerbd.2" class="lexicon_definition"></a>
#### lowerbd{T}(A::LowerTriangular{T, Array{T, 2}}) [¶](#method__lowerbd.2)
lower bounds on the parameters (elements in the lower triangle)


*source:*
[MixedModels/src/paramlowertriangular.jl:45](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/paramlowertriangular.jl#L45)

---

<a id="method__objective.1" class="lexicon_definition"></a>
#### objective(m::MixedModels.LinearMixedModel{T}) [¶](#method__objective.1)
    objective(m)

Args:

- `m`: a `LinearMixedModel` object

Returns:
  Negative twice the log-likelihood of model `m`


*source:*
[MixedModels/src/pls.jl:196](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L196)

---

<a id="method__pwrss.1" class="lexicon_definition"></a>
#### pwrss(m::MixedModels.LinearMixedModel{T}) [¶](#method__pwrss.1)
    pwrss(m::LinearMixedModel)

Args:

- `m`: a `LinearMixedModel`

Returns:
  The penalized residual sum-of-squares, a scalar.


*source:*
[MixedModels/src/pls.jl:286](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L286)

---

<a id="method__ranef.2" class="lexicon_definition"></a>
#### ranef(m::MixedModels.MixedModel) [¶](#method__ranef.2)
    ranef(m)
    ranef(m, uscale)

Conditional modes of the random effects in model `m`

Args:

- `m`: a fitted `MixedModel` object
- `uscale`: a `Bool` indicating conditional modes are on the `u` scale or the `b` scale.  Defaults to `false`

Returns:
  A `Vector` of matrices of the conditional modes of the random effects on the indicated scale.
  For a scalar random-effects term the matrix is `1 × k` where `k` is the number of levels of the grouping factor.
  For a vector-valued random-effects term the matrix is `l × k` where `l` is the dimension of each random effect.


*source:*
[MixedModels/src/mixedmodel.jl:182](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L182)

---

<a id="method__ranef.3" class="lexicon_definition"></a>
#### ranef(m::MixedModels.MixedModel,  uscale) [¶](#method__ranef.3)
    ranef(m)
    ranef(m, uscale)

Conditional modes of the random effects in model `m`

Args:

- `m`: a fitted `MixedModel` object
- `uscale`: a `Bool` indicating conditional modes are on the `u` scale or the `b` scale.  Defaults to `false`

Returns:
  A `Vector` of matrices of the conditional modes of the random effects on the indicated scale.
  For a scalar random-effects term the matrix is `1 × k` where `k` is the number of levels of the grouping factor.
  For a vector-valued random-effects term the matrix is `l × k` where `l` is the dimension of each random effect.


*source:*
[MixedModels/src/mixedmodel.jl:182](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L182)

---

<a id="method__refit.1" class="lexicon_definition"></a>
#### refit!(m::MixedModels.LinearMixedModel{T},  y) [¶](#method__refit.1)
refit the model `m` with response `y`


*source:*
[MixedModels/src/bootstrap.jl:143](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/bootstrap.jl#L143)

---

<a id="method__remat.1" class="lexicon_definition"></a>
#### remat(e::Expr,  df::DataFrames.DataFrame) [¶](#method__remat.1)
`remat(e,df)` -> `ReMat`

A factory for `ReMat` objects constructed from a random-effects term and a
`DataFrame`


*source:*
[MixedModels/src/remat.jl:37](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/remat.jl#L37)

---

<a id="method__reml.1" class="lexicon_definition"></a>
#### reml!(m::MixedModels.LinearMixedModel{T}) [¶](#method__reml.1)
`reml!(m,v=true)` -> m : Set m.REML to v.  If m.REML is modified, unset m.fit


*source:*
[MixedModels/src/pls.jl:351](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L351)

---

<a id="method__reml.2" class="lexicon_definition"></a>
#### reml!(m::MixedModels.LinearMixedModel{T},  v::Bool) [¶](#method__reml.2)
`reml!(m,v=true)` -> m : Set m.REML to v.  If m.REML is modified, unset m.fit


*source:*
[MixedModels/src/pls.jl:351](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L351)

---

<a id="method__sdest.1" class="lexicon_definition"></a>
#### sdest(m::MixedModels.LinearMixedModel{T}) [¶](#method__sdest.1)
    sdest(m)

Args:

- `m`: a `MixedModel` object

Returns:
  The scalar, `s`, the estimate of σ, the standard deviation of the per-observation noise


*source:*
[MixedModels/src/pls.jl:255](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L255)

---

<a id="method__simulate.1" class="lexicon_definition"></a>
#### simulate!(m::MixedModels.LinearMixedModel{T}) [¶](#method__simulate.1)
Simulate a response vector from model `m`, and refit `m`.

- m, LinearMixedModel.
- β, fixed effects parameter vector
- σ, standard deviation of the per-observation random noise term
- σv, vector of standard deviations for the scalar random effects.


*source:*
[MixedModels/src/bootstrap.jl:127](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/bootstrap.jl#L127)

---

<a id="method__varest.1" class="lexicon_definition"></a>
#### varest(m::MixedModels.LinearMixedModel{T}) [¶](#method__varest.1)
    varest(m::LinearMixedModel)

Args:

- `m`: a `LinearMixedModel`

Returns:
The scalar, s², the estimate of σ², the variance of the conditional distribution of Y given B


*source:*
[MixedModels/src/pls.jl:274](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L274)

## Types [Exported]

---

<a id="type__linearmixedmodel.1" class="lexicon_definition"></a>
#### MixedModels.LinearMixedModel{T} [¶](#type__linearmixedmodel.1)
Linear mixed-effects model representation

- `mf`: the model frame, mostly used to get the `terms` component for labelling fixed effects
- `trms`: a length `nt` vector of model matrices. Its last element is `hcat(X,y)`
- `Λ`: a length `nt - 1` vector of lower triangular matrices
- `weights`: a length `n` vector of weights
- `A`: an `nt × nt` symmetric matrix of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
- `R`: a `nt × nt` matrix of matrices - the upper Cholesky factor of `Λ'AΛ+I`
- `opt`: an `OptSummary` object


*source:*
[MixedModels/src/pls.jl:26](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L26)

---

<a id="type__remat.1" class="lexicon_definition"></a>
#### MixedModels.ReMat [¶](#type__remat.1)
`ReMat` - model matrix for a random-effects term


*source:*
[MixedModels/src/remat.jl:4](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/remat.jl#L4)

---

<a id="type__scalarremat.1" class="lexicon_definition"></a>
#### MixedModels.ScalarReMat{T} [¶](#type__scalarremat.1)
`ScalarReMat` - a model matrix for scalar random effects

The matrix is represented by the grouping factor, `f`, and a vector `z`.


*source:*
[MixedModels/src/remat.jl:11](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/remat.jl#L11)

---

<a id="type__varcorr.1" class="lexicon_definition"></a>
#### MixedModels.VarCorr [¶](#type__varcorr.1)
`VarCorr` a type to encapsulate the information on the fitted random-effects
variance-covariance matrices.

The main purpose is to isolate the logic in the show method.


*source:*
[MixedModels/src/pls.jl:393](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L393)

---

<a id="type__vectorremat.1" class="lexicon_definition"></a>
#### MixedModels.VectorReMat{T} [¶](#type__vectorremat.1)
`VectorReMat` - a representation of a model matrix for vector-valued random effects

The matrix is represented by the grouping factor, `f`, and the transposed raw
model matrix, `z`.


*source:*
[MixedModels/src/remat.jl:24](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/remat.jl#L24)


## Methods [Internal]

---

<a id="method__ld.1" class="lexicon_definition"></a>
#### LD{T}(d::Diagonal{T}) [¶](#method__ld.1)
`LD(A) -> log(det(triu(A)))` for `A` diagonal, HBlkDiag, or UpperTriangular


*source:*
[MixedModels/src/logdet.jl:4](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/logdet.jl#L4)

---

<a id="method__lt.1" class="lexicon_definition"></a>
#### LT(A::MixedModels.ScalarReMat{T}) [¶](#method__lt.1)
`LT(A) -> LowerTriangular`

Create as a lower triangular matrix compatible with the blocks of `A`
and initialized to the identity.


*source:*
[MixedModels/src/paramlowertriangular.jl:96](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/paramlowertriangular.jl#L96)

---

<a id="method__cfactor.1" class="lexicon_definition"></a>
#### cfactor!(A::AbstractArray{T, 2}) [¶](#method__cfactor.1)
Slightly modified version of `chol!` from `julia/base/linalg/cholesky.jl`

Uses `inject!` (as opposed to `copy!`), `downdate!` (as opposed to `syrk!`
    or `gemm!`) and recursive calls to `cfactor!`,


*source:*
[MixedModels/src/cfactor.jl:7](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/cfactor.jl#L7)

---

<a id="method__cfactor.2" class="lexicon_definition"></a>
#### cfactor!(R::Array{Float64, 2}) [¶](#method__cfactor.2)
`cfactor!` method for dense matrices calls `LAPACK.potrf!` directly to avoid
errors being thrown when `R` is computationally singular


*source:*
[MixedModels/src/cfactor.jl:34](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/cfactor.jl#L34)

---

<a id="method__chol2cor.1" class="lexicon_definition"></a>
#### chol2cor(L::LowerTriangular{T, S<:AbstractArray{T, 2}}) [¶](#method__chol2cor.1)
Convert a lower Cholesky factor to a correlation matrix


*source:*
[MixedModels/src/pls.jl:291](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L291)

---

<a id="method__cond.1" class="lexicon_definition"></a>
#### cond(m::MixedModels.MixedModel) [¶](#method__cond.1)
    cond(m::MixedModel)

Args:

- `m`: a `MixedModel`

Returns:
  A `Vector` of the condition numbers of the blocks of `m.Λ`


*source:*
[MixedModels/src/mixedmodel.jl:31](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L31)

---

<a id="method__densify.1" class="lexicon_definition"></a>
#### densify(S) [¶](#method__densify.1)
`densify(S[,threshold])`

Convert sparse `S` to `Diagonal` if S is diagonal
Convert sparse `S` to dense if the proportion of nonzeros exceeds `threshold`.
A no-op for other matrix types.


*source:*
[MixedModels/src/densify.jl:8](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/densify.jl#L8)

---

<a id="method__densify.2" class="lexicon_definition"></a>
#### densify(S,  threshold) [¶](#method__densify.2)
`densify(S[,threshold])`

Convert sparse `S` to `Diagonal` if S is diagonal
Convert sparse `S` to dense if the proportion of nonzeros exceeds `threshold`.
A no-op for other matrix types.


*source:*
[MixedModels/src/densify.jl:8](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/densify.jl#L8)

---

<a id="method__describeblocks.1" class="lexicon_definition"></a>
#### describeblocks(m::MixedModels.LinearMixedModel{T}) [¶](#method__describeblocks.1)
describe the blocks of the A and R matrices


*source:*
[MixedModels/src/pls.jl:460](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L460)

---

<a id="method__df.1" class="lexicon_definition"></a>
#### df(m::MixedModels.LinearMixedModel{T}) [¶](#method__df.1)
Number of parameters in the model.

Note that `size(m.trms[end],2)` is `length(coef(m)) + 1`, thereby accounting
for the scale parameter, σ, that is profiled out.


*source:*
[MixedModels/src/pls.jl:237](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L237)

---

<a id="method__downdate.1" class="lexicon_definition"></a>
#### downdate!{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}}(C::DenseArray{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}, 2},  A::DenseArray{T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64}, 2}) [¶](#method__downdate.1)
Subtract, in place, A'A or A'B from C


*source:*
[MixedModels/src/cfactor.jl:48](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/cfactor.jl#L48)

---

<a id="method__fit.1" class="lexicon_definition"></a>
#### fit!(m::MixedModels.LinearMixedModel{T}) [¶](#method__fit.1)
`fit!(m)` -> `m`

Optimize the objective using an NLopt optimizer.


*source:*
[MixedModels/src/pls.jl:128](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L128)

---

<a id="method__fit.2" class="lexicon_definition"></a>
#### fit!(m::MixedModels.LinearMixedModel{T},  verbose::Bool) [¶](#method__fit.2)
`fit!(m)` -> `m`

Optimize the objective using an NLopt optimizer.


*source:*
[MixedModels/src/pls.jl:128](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L128)

---

<a id="method__fit.3" class="lexicon_definition"></a>
#### fit!(m::MixedModels.LinearMixedModel{T},  verbose::Bool,  optimizer::Symbol) [¶](#method__fit.3)
`fit!(m)` -> `m`

Optimize the objective using an NLopt optimizer.


*source:*
[MixedModels/src/pls.jl:128](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L128)

---

<a id="method__fixef.1" class="lexicon_definition"></a>
#### fixef!(v,  m::MixedModels.LinearMixedModel{T}) [¶](#method__fixef.1)
    fixef!(v, m)

Overwrite `v` with the fixed-effects coefficients of model `m`

Args:

- `v`: a `Vector` of length `p`, the number of fixed-effects parameters
- `m`: a `MixedModel`

Returns:
  `v` with its contents overwritten by the fixed-effects parameters


*source:*
[MixedModels/src/pls.jl:214](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L214)

---

<a id="method__fnames.1" class="lexicon_definition"></a>
#### fnames(m::MixedModels.MixedModel) [¶](#method__fnames.1)
    fnames(m::MixedModel)

Args:

- `m`: a `MixedModel`

Returns:
  A `Vector{AbstractString}` of names of the grouping factors for the random-effects terms.


*source:*
[MixedModels/src/mixedmodel.jl:100](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L100)

---

<a id="method__getindex.1" class="lexicon_definition"></a>
#### getindex{T}(A::LowerTriangular{T, Array{T, 2}},  s::Symbol) [¶](#method__getindex.1)
return the lower triangle as a vector (column-major ordering)


*source:*
[MixedModels/src/paramlowertriangular.jl:7](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/paramlowertriangular.jl#L7)

---

<a id="method__grplevels.1" class="lexicon_definition"></a>
#### grplevels(m::MixedModels.MixedModel) [¶](#method__grplevels.1)
`grplevels(m)` -> Vector{Int} : number of levels in each term's grouping factor


*source:*
[MixedModels/src/mixedmodel.jl:105](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L105)

---

<a id="method__inflate.1" class="lexicon_definition"></a>
#### inflate!(A::MixedModels.HBlkDiag{T}) [¶](#method__inflate.1)
`inflate!(A)` is equivalent to `A += I`, without making a copy of A

Even if `A += I` did not make a copy, this function is needed for the special
behavior on the `HBlkDiag` type.


*source:*
[MixedModels/src/inflate.jl:7](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/inflate.jl#L7)

---

<a id="method__inject.1" class="lexicon_definition"></a>
#### inject!(d,  s) [¶](#method__inject.1)
like `copy!` but allowing for heterogeneous matrix types


*source:*
[MixedModels/src/inject.jl:4](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/inject.jl#L4)

---

<a id="method__isfit.1" class="lexicon_definition"></a>
#### isfit(m::MixedModels.LinearMixedModel{T}) [¶](#method__isfit.1)
    isfit(m)

check if a model has been fit.

Args:

- `m`; a `LinearMixedModel`

Returns:
  A logical value indicating if the model has been fit.


*source:*
[MixedModels/src/pls.jl:323](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L323)

---

<a id="method__logdet.1" class="lexicon_definition"></a>
#### logdet(m::MixedModels.LinearMixedModel{T}) [¶](#method__logdet.1)
returns `log(det(Λ'Z'ZΛ + I))`


*source:*
[MixedModels/src/logdet.jl:35](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/logdet.jl#L35)

---

<a id="method__lrt.1" class="lexicon_definition"></a>
#### lrt(mods::MixedModels.LinearMixedModel{T}...) [¶](#method__lrt.1)
Likelihood ratio test of one or more models


*source:*
[MixedModels/src/pls.jl:328](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L328)

---

<a id="method__model_response.1" class="lexicon_definition"></a>
#### model_response(m::MixedModels.LinearMixedModel{T}) [¶](#method__model_response.1)
extract the response (as a reference)

In Julia 0.5 this can be a one-liner `m.trms[end][:,end]`


*source:*
[MixedModels/src/bootstrap.jl:153](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/bootstrap.jl#L153)

---

<a id="method__ranef.1" class="lexicon_definition"></a>
#### ranef!(v::Array{T, 1},  m::MixedModels.MixedModel,  uscale) [¶](#method__ranef.1)
    ranef!(v, m, uscale)

Overwrite v with the conditional modes of the random effects for `m`

Args:

- `v`: a `Vector` of matrices
- `m`: a `MixedModel`
- `uscale`: a `Bool` indicating if the random effects on the spherical (i.e. `u`) scale are desired

Returns:
  `v`, overwritten with the conditional modes


*source:*
[MixedModels/src/mixedmodel.jl:133](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L133)

---

<a id="method__reevaluateaend.1" class="lexicon_definition"></a>
#### reevaluateAend!(m::MixedModels.LinearMixedModel{T}) [¶](#method__reevaluateaend.1)
    reevaluateAend!(m)

Reevaluate the last column of `m.A` from `m.trms`

Args:

- `m`: a `LinearMixedModel`

Returns:
  `m` with the last column of `m.A` reevaluated

Note: This function should be called after updating parts of `m.trms[end]`, typically the response.


*source:*
[MixedModels/src/bootstrap.jl:34](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/bootstrap.jl#L34)

---

<a id="method__reset952.1" class="lexicon_definition"></a>
#### resetθ!(m::MixedModels.LinearMixedModel{T}) [¶](#method__reset952.1)
    resetθ!(m)

Reset the value of `m.θ` to the initial values and mark the model as not having been fit

Args:

- `m`: a `LinearMixedModel`

Returns:
  `m`


*source:*
[MixedModels/src/bootstrap.jl:55](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/bootstrap.jl#L55)

---

<a id="method__reterms.1" class="lexicon_definition"></a>
#### reterms(m::MixedModels.MixedModel) [¶](#method__reterms.1)
    reterms(m)

Args:

- `m`: a `MixedModel`

Returns:
   A `Vector` of random-effects terms.


*source:*
[MixedModels/src/mixedmodel.jl:205](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L205)

---

<a id="method__rowlengths.1" class="lexicon_definition"></a>
#### rowlengths(L::LowerTriangular{T, S<:AbstractArray{T, 2}}) [¶](#method__rowlengths.1)
`rowlengths(L)` -> a vector of the Euclidean lengths of the rows of `L`

used in `chol2cor`


*source:*
[MixedModels/src/paramlowertriangular.jl:63](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/paramlowertriangular.jl#L63)

---

<a id="method__setindex.1" class="lexicon_definition"></a>
#### setindex!{T}(A::LowerTriangular{T, Array{T, 2}},  v::AbstractArray{T, 1},  s::Symbol) [¶](#method__setindex.1)
set the lower triangle of A to v using column-major ordering


*source:*
[MixedModels/src/paramlowertriangular.jl:23](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/paramlowertriangular.jl#L23)

---

<a id="method__sqrtpwrss.1" class="lexicon_definition"></a>
#### sqrtpwrss(m::MixedModels.LinearMixedModel{T}) [¶](#method__sqrtpwrss.1)
returns the square root of the penalized residual sum-of-squares

This is the bottom right element of the bottom right block of m.R


*source:*
[MixedModels/src/pls.jl:262](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L262)

---

<a id="method__std.1" class="lexicon_definition"></a>
#### std(m::MixedModels.MixedModel) [¶](#method__std.1)
    std(m)

Estimated standard deviations of the variance components

Args:

- `m`: a `MixedModel`

Returns:
  `Vector{Vector{Float64}}`


*source:*
[MixedModels/src/mixedmodel.jl:78](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/mixedmodel.jl#L78)

---

<a id="method__tscale.1" class="lexicon_definition"></a>
#### tscale!(A::LowerTriangular{T, S<:AbstractArray{T, 2}},  B::MixedModels.HBlkDiag{T}) [¶](#method__tscale.1)
scale B using the implicit expansion of A to a homogeneous block diagonal


*source:*
[MixedModels/src/paramlowertriangular.jl:71](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/paramlowertriangular.jl#L71)

---

<a id="method__unscaledre.1" class="lexicon_definition"></a>
#### unscaledre!(y::AbstractArray{T, 1},  M::MixedModels.ScalarReMat{T},  L::LowerTriangular{T, S<:AbstractArray{T, 2}},  u::DenseArray{T, 2}) [¶](#method__unscaledre.1)
    unscaledre!(y, M, L, u)

Add unscaled random effects to `y`.

Args:

- `y`: response vector to which the random effects are to be added
- `M`: an `ReMat`
- `L`: the `LowerTriangular` matrix defining `Λ` for this term
- `u`: a `Matrix` of random effects on the `u` scale. Defaults to a standard multivariate normal of the appropriate size.

Returns:
  the updated `y`


*source:*
[MixedModels/src/bootstrap.jl:77](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/bootstrap.jl#L77)

---

<a id="method__vcov.1" class="lexicon_definition"></a>
#### vcov(m::MixedModels.LinearMixedModel{T}) [¶](#method__vcov.1)
returns the estimated variance-covariance matrix of the fixed-effects estimator


*source:*
[MixedModels/src/pls.jl:452](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L452)

## Types [Internal]

---

<a id="type__optsummary.1" class="lexicon_definition"></a>
#### MixedModels.OptSummary [¶](#type__optsummary.1)
Summary of an NLopt optimization


*source:*
[MixedModels/src/pls.jl:4](https://github.com/dmbates/MixedModels.jl/tree/e8eb13dfad54e1650657153b055d40fb56f6ff46/src/pls.jl#L4)

