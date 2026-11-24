# Robust (sandwich) covariance estimators

Support for sandwich estimators of the covariance matrix of the fixed-effects estimates of a `LinearMixedModel` is provided by a package extension for [Vcov.jl](https://github.com/FixedEffects/Vcov.jl).
The extension is loaded automatically when Vcov.jl is loaded.

The estimators use the generalized least squares "bread" ``X'V^{-1}X``, where ``V`` is the (relative) marginal covariance of the response implied by the fitted random effects, and the scores ``x_i [V^{-1}(y - X\hat\beta)]_i``.
These can be computed without forming the ``n \times n`` matrix ``V``, because ``V^{-1}(y - X\hat\beta)`` is the vector of conditional residuals `residuals(m)`.

```@docs
vcov(::LinearMixedModel, ::Vcov.SimpleCovariance)
vcov(::LinearMixedModel, ::Vcov.RobustCovariance)
vcov(::LinearMixedModel, ::Vcov.ClusterCovariance)
```

The corresponding standard errors are available via [`stderror(m, est)`](@ref stderror), and `coeftable` and `show` accept a `vcov` keyword argument to use an estimator for the standard errors, ``z``-statistics and ``p``-values.

Generalized linear mixed models are not yet supported.

## Examples

```@example Sandwich
using MixedModels, MixedModelsDatasets, Vcov
using DisplayAs # hide
fm = lmm(@formula(reaction ~ 1 + days + (1 + days | subj)),
         MixedModelsDatasets.dataset(:sleepstudy))
DisplayAs.Text(ans) # hide
```

The simple estimator is the usual model-based covariance matrix:

```@example Sandwich
vcov(fm, Vcov.simple()) ≈ vcov(fm)
```

The cluster-robust estimator, clustering on the grouping factor:

```@example Sandwich
vcov(fm, Vcov.cluster(:subj))
```

which is close to the model-based estimate, because the random-effects structure accounts for the within-subject correlation:

```@example Sandwich
vcov(fm)
```

The corresponding standard errors:

```@example Sandwich
stderror(fm, Vcov.cluster(:subj))
```

and the coefficient table using them:

```@example Sandwich
coeftable(fm; vcov=Vcov.cluster(:subj))
```

Cluster names must be grouping factors of the model.
To cluster on another variable, pass the clusters directly, with one entry per observation used to fit the model.
For example (purely to illustrate the syntax, since `days` does not define independent units here):

```@example Sandwich
days = MixedModelsDatasets.dataset(:sleepstudy).days
vcov(fm, Vcov.cluster((:days,), (Vcov.GroupedArray(days),)))
```

The heteroskedasticity-robust (HC1) estimator is also available, but because it treats observations as independent, it is rarely appropriate for mixed models:

```@example Sandwich
vcov(fm, Vcov.robust())
```
