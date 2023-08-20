# Model constructors

The `LinearMixedModel` type represents a linear mixed-effects model.
Typically it is constructed from a `Formula` and an appropriate `Table` type, usually a `DataFrame`.

## Examples of linear mixed-effects model fits

For illustration, several data sets from the *lme4* package for *R* are made available in `.arrow` format in this package.
Often, for convenience, we will convert these to `DataFrame`s.
These data sets include the `dyestuff` and `dyestuff2` data sets.

```@setup Main
using DisplayAs
```

```@example Main
using DataFrames, MixedModels, StatsModels
dyestuff = MixedModels.dataset(:dyestuff)
```

```@example Main
describe(DataFrame(dyestuff))
```

### The `@formula` language in Julia

MixedModels.jl builds on the the *Julia* formula language provided by [StatsModels.jl](https://juliastats.org/StatsModels.jl/stable/formula/), which is similar to the formula language in *R* and is also based on the notation from Wilkinson and Rogers ([1973](https://dx.doi.org/10.2307/2346786)). There are two ways to construct a formula in Julia.  The first way is to enclose the formula expression in the `@formula` macro:
```@docs
@formula
```

The second way is to combine `Term`s with operators like `+`, `&`, `~`, and others at "run time".  This is especially useful if you wish to create a formula from a list a variable names.  For instance, the following are equivalent:

```@example Main
@formula(y ~ 1 + a + b + a & b) == (term(:y) ~ term(1) + term(:a) + term(:b) + term(:a) & term(:b))
```

MixedModels.jl provides additional formula syntax for representing *random-effects terms*.  Most importantly, `|` separates random effects and their grouping factors (as in the formula extension used by the *R* package [`lme4`](https://cran.r-project.org/web/packages/lme4/index.html).  Much like with the base formula language, `|` can be used within the `@formula` macro and to construct a formula programmatically:

```@example Main
@formula(y ~ 1 + a + b + (1 + a + b | g))
```

```@example Main
terms = sum(term(t) for t in [1, :a, :b])
group = term(:g)
response = term(:y)
response ~ terms + (terms | group)
```

### Models with simple, scalar random effects

A basic model with simple, scalar random effects for the levels of `batch` (the batch of an intermediate product, in this case) is declared and fit as

```@example Main
fm = @formula(yield ~ 1 + (1|batch))
fm1 = fit(MixedModel, fm, dyestuff)
DisplayAs.Text(ans) # hide
```

(If you are new to Julia you may find that this first fit takes an unexpectedly long time, due to Just-In-Time (JIT) compilation of the code. The subsequent calls to such functions are much faster.)

```@example Main
using BenchmarkTools
dyestuff2 = MixedModels.dataset(:dyestuff2)
@benchmark fit(MixedModel, $fm, $dyestuff2)
```

By default, the model is fit by maximum likelihood. To use the `REML` criterion instead, add the optional named argument `REML=true` to the call to `fit`
```@example Main
fm1reml = fit(MixedModel, fm, dyestuff, REML=true)
DisplayAs.Text(ans) # hide
```

### Floating-point type in the model

The type of `fm1`
```@example Main
typeof(fm1)
```
includes the floating point type used internally for the various matrices, vectors, and scalars that represent the model.
At present, this will always be `Float64` because the parameter estimates are optimized using the [`NLopt` package](https://github.com/JuliaOpt/NLopt.jl) which calls compiled C code that only allows for optimization with respect to a `Float64` parameter vector.

So in theory other floating point types, such as `BigFloat` or `Float32`, can be used to define a model but in practice only `Float64` works at present.

> In theory, theory and practice are the same.  In practice, they aren't.  -- Anon

### Simple, scalar random effects

A simple, scalar random effects term in a mixed-effects model formula is of the form `(1|G)`.
All random effects terms end with `|G` where `G` is the *grouping factor* for the random effect.
The name or, more generally the expression, `G`, should evaluate to a categorical array that has a distinct set of *levels*.
The random effects are associated with the levels of the grouping factor.

A *scalar* random effect is, as the name implies, one scalar value for each level of the grouping factor.
A *simple, scalar* random effects term is of the form, `(1|G)`.
It corresponds to a shift in the intercept for each level of the grouping factor.

### Models with vector-valued random effects

The *sleepstudy* data are observations of reaction time, `reaction`, on several subjects, `subj`, after 0 to 9 days of sleep deprivation, `days`.
A model with random intercepts and random slopes for each subject, allowing for within-subject correlation of the slope and intercept, is fit as
```@example Main
sleepstudy = MixedModels.dataset(:sleepstudy)
fm2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy)
DisplayAs.Text(ans) # hide
```

### Models with multiple, scalar random-effects terms

A model for the *Penicillin* data incorporates random effects for the plate, and for the sample.
As every sample is used on every plate these two factors are *crossed*.
```@example Main
penicillin = MixedModels.dataset(:penicillin)
fm3 = fit(MixedModel, @formula(diameter ~ 1 + (1|plate) + (1|sample)), penicillin)
DisplayAs.Text(ans) # hide
```

In contrast, the `cask` grouping factor is *nested* within the `batch` grouping factor in the *Pastes* data.
```@example Main
pastes = DataFrame(MixedModels.dataset(:pastes))
describe(pastes)
```
This can be expressed using the solidus (the "`/`" character) to separate grouping factors, read "`cask` nested within `batch`":
```@example Main
fm4a = fit(MixedModel, @formula(strength ~ 1 + (1|batch/cask)), pastes)
DisplayAs.Text(ans) # hide
```

If the levels of the inner grouping factor are unique across the levels of the outer grouping factor, then this nesting does not need to expressed explicitly in the model syntax. For example, defining `sample` to be the combination of `batch` and `cask`, yields a naming scheme where the nesting is apparent from the data even if not expressed in the formula. (That is, each level of `sample` occurs in conjunction with only one level of `batch`.) As such, this model is equivalent to the previous one.
```@example Main
pastes.sample = (string.(pastes.cask, "&",  pastes.batch))
fm4b = fit(MixedModel, @formula(strength ~ 1 + (1|sample) + (1|batch)), pastes)
DisplayAs.Text(ans) # hide
```

In observational studies it is common to encounter *partially crossed* grouping factors.
For example, the *InstEval* data are course evaluations by students, `s`, of instructors, `d`.
Additional covariates include the academic department, `dept`, in which the course was given and `service`, whether or not it was a service course.
```@example Main
insteval = MixedModels.dataset(:insteval)
fm5 = fit(MixedModel, @formula(y ~ 1 + service * dept + (1|s) + (1|d)), insteval)
DisplayAs.Text(ans) # hide
```

### Simplifying the random effect correlation structure

MixedModels.jl estimates not only the *variance* of the effects for each random effect level, but also the *correlation* between the random effects for different predictors.
So, for the model of the *sleepstudy* data above, one of the parameters that is estimated is the correlation between each subject's random intercept (i.e., their baseline reaction time) and slope (i.e., their particular change in reaction time per day of sleep deprivation).
In some cases, you may wish to simplify the random effects structure by removing these correlation parameters.
This often arises when there are many random effects you want to estimate (as is common in psychological experiments with many conditions and covariates), since the number of random effects parameters increases as the square of the number of predictors, making these models difficult to estimate from limited data.

The special syntax `zerocorr` can be applied to individual random effects terms inside the `@formula`:
```@example Main
fm2zerocorr_fm = fit(MixedModel, @formula(reaction ~ 1 + days + zerocorr(1 + days|subj)), sleepstudy)
DisplayAs.Text(ans) # hide
```

Alternatively, correlations between parameters can be removed by including them as separate random effects terms:
```@example Main
fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj) + (days|subj)), sleepstudy)
DisplayAs.Text(ans) # hide
```

Finally, for predictors that are categorical, MixedModels.jl will estimate correlations between each level.
Notice the large number of correlation parameters if we treat `days` as a categorical variable by giving it contrasts:
```@example Main
fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy,
    contrasts = Dict(:days => DummyCoding()))
DisplayAs.Text(ans) # hide
```

Separating the `1` and `days` random effects into separate terms removes the correlations between the intercept and the levels of `days`, but not between the levels themselves:
```@example Main
fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj) + (days|subj)), sleepstudy,
    contrasts = Dict(:days => DummyCoding()))
DisplayAs.Text(ans) # hide
```
(Notice that the variance component for `days: 1` is estimated as zero, so the correlations for this component are undefined and expressed as `NaN`, not a number.)

An alternative is to force all the levels of `days` as indicators using `fulldummy` encoding.
```@docs
fulldummy
```
```@example Main
fit(MixedModel, @formula(reaction ~ 1 + days + (1 + fulldummy(days)|subj)), sleepstudy,
    contrasts = Dict(:days => DummyCoding()))
DisplayAs.Text(ans) # hide
```
This fit produces a better fit as measured by the objective (negative twice the log-likelihood is 1610.8) but at the expense of adding many more parameters to the model.
As a result, model comparison criteria such, as `AIC` and `BIC`, are inflated.

But using `zerocorr` on the individual terms does remove the correlations between the levels:
```@example Main
fit(MixedModel, @formula(reaction ~ 1 + days + zerocorr(1 + days|subj)), sleepstudy,
    contrasts = Dict(:days => DummyCoding()))
DisplayAs.Text(ans) # hide
```
```@example Main
fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj) + zerocorr(days|subj)), sleepstudy,
    contrasts = Dict(:days => DummyCoding()))
DisplayAs.Text(ans) # hide
```
```@example Main
fit(MixedModel, @formula(reaction ~ 1 + days + zerocorr(1 + fulldummy(days)|subj)), sleepstudy,
    contrasts = Dict(:days => DummyCoding()))
DisplayAs.Text(ans) # hide
```

## Fitting generalized linear mixed models

To create a GLMM representation, the distribution family for the response, and possibly the link function, must be specified.

```@example Main
verbagg = MixedModels.dataset(:verbagg)
verbaggform = @formula(r2 ~ 1 + anger + gender + btype + situ + mode + (1|subj) + (1|item));
gm1 = fit(MixedModel, verbaggform, verbagg, Bernoulli())
DisplayAs.Text(ans) # hide
```

The canonical link, which is `LogitLink` for the `Bernoulli` distribution, is used if no explicit link is specified.

Note that, in keeping with convention in the [`GLM` package](https://github.com/JuliaStats/GLM.jl), the distribution family for a binary (i.e. 0/1) response is the `Bernoulli` distribution.
The `Binomial` distribution is only used when the response is the fraction of trials returning a positive, in which case the number of trials must be specified as the case weights.

### Optional arguments to fit

An alternative approach is to create the `GeneralizedLinearMixedModel` object then call `fit!` on it.
The optional arguments `fast` and/or `nAGQ` can be passed to the optimization process via both `fit` and `fit!` (i.e these optimization settings are not used nor recognized when constructing the model).

As the name implies, `fast=true`, provides a faster but somewhat less accurate fit.
These fits may suffice for model comparisons.
```@example Main
gm1a = fit(MixedModel, verbaggform, verbagg, Bernoulli(), fast = true)
deviance(gm1a) - deviance(gm1)
```
```@example Main
@benchmark fit(MixedModel, $verbaggform, $verbagg, Bernoulli())
```
```@example Main
@benchmark fit(MixedModel, $verbaggform, $verbagg, Bernoulli(), fast = true)
```

The optional argument `nAGQ=k` causes evaluation of the deviance function to use a `k` point
adaptive Gauss-Hermite quadrature rule.
This method only applies to models with a single, simple, scalar random-effects term, such as
```@example Main
contraception = MixedModels.dataset(:contra)
contraform = @formula(use ~ 1 + age + abs2(age) + livch + urban + (1|dist));
bernoulli = Bernoulli()
deviances = Dict{Symbol,Float64}()
b = @benchmarkable deviances[:default] = deviance(fit(MixedModel, $contraform, $contraception, $bernoulli));
run(b)
b = @benchmarkable deviances[:fast] = deviance(fit(MixedModel, $contraform, $contraception, $bernoulli, fast = true));
run(b)
b = @benchmarkable deviances[:nAGQ] = deviance(fit(MixedModel, $contraform, $contraception, $bernoulli, nAGQ=9));
run(b)
b = @benchmarkable deviances[:nAGQ_fast] = deviance(fit(MixedModel, $contraform, $contraception, $bernoulli, nAGQ=9, fast=true));
run(b)
sort(deviances)
```

# Extractor functions

`LinearMixedModel` and `GeneralizedLinearMixedModel` are subtypes of `StatsAPI.RegressionModel` which, in turn, is a subtype of `StatsBase.StatisticalModel`.
Many of the generic extractors defined in the `StatsBase` package have methods for these models.

## Model-fit statistics

The statistics describing the quality of the model fit include
```@docs
loglikelihood
aic
bic
dof
nobs
```
```@example Main
loglikelihood(fm1)
```
```@example Main
aic(fm1)
```
```@example Main
bic(fm1)
```
```@example Main
dof(fm1)   # 1 fixed effect, 2 variances
```
```@example Main
nobs(fm1)  # 30 observations
```
```@example Main
loglikelihood(gm1)
```

In general the [`deviance`](https://en.wikipedia.org/wiki/Deviance_(statistics)) of a statistical model fit is negative twice the log-likelihood adjusting for the saturated model.
```@docs
deviance(::StatisticalModel)
```

Because it is not clear what the saturated model corresponding to a particular `LinearMixedModel` should be, negative twice the log-likelihood is called the `objective`.
```@docs
objective
```
This value is also accessible as the `deviance` but the user should bear in mind that this doesn't have all the properties of a deviance which is corrected for the saturated model.
For example, it is not necessarily non-negative.
```@example Main
objective(fm1)
```
```@example Main
deviance(fm1)
```

The value optimized when fitting a `GeneralizedLinearMixedModel` is the Laplace approximation to the deviance or an adaptive Gauss-Hermite evaluation.
```@docs
MixedModels.deviance!
```
```@example Main
MixedModels.deviance!(gm1)
```

## Fixed-effects parameter estimates

The `coef` and `fixef` extractors both return the maximum likelihood estimates of the fixed-effects coefficients.
They differ in their behavior in the rank-deficient case.
The associated `coefnames` and `fixefnames` return the corresponding coefficient names.
```@docs
coef
coefnames
fixef
fixefnames
```
```@example Main
coef(fm1)
coefnames(fm1)
```
```@example Main
fixef(fm1)
fixefnames(fm1)
```

An alternative extractor for the fixed-effects coefficient is the `β` property.
Properties whose names are Greek letters usually have an alternative spelling, which is the name of the Greek letter.
```@example Main
fm1.β
```
```@example Main
fm1.beta
```
```@example Main
gm1.β
```
A full list of property names is returned by `propertynames`
```@example Main
propertynames(fm1)
```
```@example Main
propertynames(gm1)
```

The variance-covariance matrix of the fixed-effects coefficients is returned by
```@docs
vcov
```
```@example Main
vcov(fm2)
```
```@example Main
vcov(gm1)
```

The standard errors are the square roots of the diagonal elements of the estimated variance-covariance matrix of the fixed-effects coefficient estimators.
```@docs
stderror
```
```@example Main
stderror(fm2)
```
```@example Main
stderror(gm1)
```

Finally, the `coeftable` generic produces a table of coefficient estimates, their standard errors, and their ratio.
The *p-values* quoted here should be regarded as approximations.
```@docs
coeftable
```
```@example Main
coeftable(fm2)
DisplayAs.Text(ans) # hide
```

## Covariance parameter estimates

The covariance parameters estimates, in the form shown in the model summary, are a `VarCorr` object
```@example Main
VarCorr(fm2)
DisplayAs.Text(ans) # hide
```
```@example Main
VarCorr(gm1)
DisplayAs.Text(ans) # hide
```

Individual components are returned by other extractors
```@docs
varest
sdest
```
```@example Main
varest(fm2)
```
```@example Main
sdest(fm2)
```
```@example Main
fm2.σ
```

## Conditional modes of the random effects

The `ranef` extractor
```@docs
ranef
```
```@example Main
ranef(fm1)
```
```@example Main
fm1.b
```
returns the *conditional modes* of the random effects given the observed data.
That is, these are the values that maximize the conditional density of the random effects given the observed data.
For a `LinearMixedModel` these are also the conditional means.

These are sometimes called the *best linear unbiased predictors* or [`BLUPs`](https://en.wikipedia.org/wiki/Best_linear_unbiased_prediction) but that name is not particularly meaningful.

At a superficial level these can be considered as the "estimates" of the random effects, with a bit of hand waving, but pursuing this analogy too far usually results in confusion.

To obtain tables associating the values of the conditional modes with the levels of the grouping factor, use
```@docs
raneftables
```
as in
```@example Main
DataFrame(only(raneftables(fm1)))
```

The corresponding conditional variances are returned by
```@docs
condVar
```
```@example Main
condVar(fm1)
```

## Case-wise diagnostics and residual degrees of freedom

The `leverage` values
```@docs
leverage
```
```@example Main
leverage(fm1)
```
are used in diagnostics for linear regression models to determine cases that exert a strong influence on their own predicted response.

The documentation refers to a "projection".
For a linear model without random effects the fitted values are obtained by orthogonal projection of the response onto the column span of the model matrix and the sum of the leverage values is the dimension of this column span.
That is, the sum of the leverage values is the rank of the model matrix and `n - sum(leverage(m))` is the degrees of freedom for residuals.
The sum of the leverage values is also the trace of the so-called "hat" matrix, `H`.
(The name "hat matrix" reflects the fact that $\hat{\mathbf{y}} = \mathbf{H} \mathbf{y}$.  That is, `H` puts a hat on `y`.)

For a linear mixed model the sum of the leverage values will be between `p`, the rank of the fixed-effects model matrix, and `p + q` where `q` is the total number of random effects.
This number does not represent a dimension (or "degrees of freedom") of a linear subspace of all possible fitted values because the projection is not an orthogonal projection.
Nevertheless, it is a reasonable measure of the effective degrees of freedom of the model and `n - sum(leverage(m))` can be considered the effective residual degrees of freedom.

For model `fm1` the dimensions are
```@example Main
n, p, q, k = size(fm1)
```
which implies that the sum of the leverage values should be in the range [1, 7].
The actual value is
```@example Main
sum(leverage(fm1))
```

For model `fm2` the dimensions are
```@example Main
n, p, q, k = size(fm2)
```
providing a range of [2, 38] for the effective degrees of freedom for the model.
The observed value is
```@example Main
sum(leverage(fm2))
```

When a model converges to a singular covariance, such as
```@example Main
fm3 = fit(MixedModel, @formula(yield ~ 1+(1|batch)), MixedModels.dataset(:dyestuff2))
DisplayAs.Text(ans) # hide
```
the effective degrees of freedom is the lower bound.
```@example Main
sum(leverage(fm3))
```

Models for which the estimates of the variances of the random effects are large relative to the residual variance have effective degrees of freedom close to the upper bound.
```@example Main
fm4 = fit(MixedModel, @formula(diameter ~ 1+(1|plate)+(1|sample)),
    MixedModels.dataset(:penicillin))
DisplayAs.Text(ans) # hide
```
```@example Main
sum(leverage(fm4))
```

Also, a model fit by the REML criterion generally has larger estimates of the variance components and hence a larger effective degrees of freedom.
```@example Main
fm4r = fit(MixedModel, @formula(diameter ~ 1+(1|plate)+(1|sample)),
    MixedModels.dataset(:penicillin), REML=true)
DisplayAs.Text(ans) # hide
```
```@example Main
sum(leverage(fm4r))
```
