# Mixed-effects models in Julia

|**Documentation**|**Citation**|**Build Status**|**Code Coverage**| **Style Guide** |
|:---------------:|:----------:|:--------------:|:---------------:|:----------------|
|[![Stable Docs][docs-stable-img]][docs-stable-url] [![Dev Docs][docs-dev-img]][docs-dev-url] | [![DOI][doi-img]][doi-url] | [![Julia Current][current-img]][current-url] [![Julia Minimum Supported Version][minimum-img]][minimum-url] [![Julia Nightly][nightly-img]][nightly-url] [![PkgEval][pkgeval-img]][pkgeval-url] | [![CodeCov][codecov-img]][codecov-url] | [![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4495d1.svg)](https://github.com/invenia/BlueStyle) |

[doi-img]: https://zenodo.org/badge/9106942.svg
[doi-url]: https://zenodo.org/badge/latestdoi/9106942

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliastats.github.io/MixedModels.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliastats.github.io/MixedModels.jl/stable

[codecov-img]: https://codecov.io/github/JuliaStats/MixedModels.jl/badge.svg?branch=main
[codecov-url]: https://codecov.io/github/JuliaStats/MixedModels.jl?branch=main

[current-img]: https://github.com/JuliaStats/MixedModels.jl/actions/workflows/current.yml/badge.svg
[current-url]: https://github.com/JuliaStats/MixedModels.jl/actions?workflow=current

[nightly-img]: https://github.com/JuliaStats/MixedModels.jl/actions/workflows/nightly.yml/badge.svg
[nightly-url]: https://github.com/JuliaStats/MixedModels.jl/actions?workflow=nightly

[minimum-img]: https://github.com/JuliaStats/MixedModels.jl/actions/workflows/minimum.yml/badge.svg
[minimum-url]: https://github.com/JuliaStats/MixedModels.jl/actions?workflow=minimum

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/M/MixedModels.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html

This package defines linear mixed models (`LinearMixedModel`) and generalized linear mixed models (`GeneralizedLinearMixedModel`). Users can use the abstraction for statistical model API to build, fit (`fit`/`fit!`), and query the fitted models.

A _mixed-effects model_ is a statistical model for a _response_ variable as a function of one or more _covariates_.
For a categorical covariate the coefficients associated with the levels of the covariate are sometimes called _effects_, as in "the effect of using Treatment 1 versus the placebo".
If the potential levels of the covariate are fixed and reproducible, e.g. the levels for `Sex` could be `"F"` and `"M"`, they are modeled with _fixed-effects_ parameters.
If the levels constitute a sample from a population, e.g. the `Subject` or the `Item` at a particular observation, they are modeled as _random effects_.

A _mixed-effects_ model contains both fixed-effects and random-effects terms.

With fixed-effects it is the coefficients themselves or combinations of coefficients that are of interest.
For random effects it is the variability of the effects over the population that is of interest.

In this package random effects are modeled as independent samples from a multivariate Gaussian distribution of the form 𝓑 ~ 𝓝(0, 𝚺).
For the response vector, 𝐲, only the mean of conditional distribution, 𝓨|𝓑 = 𝐛 depends on 𝐛 and it does so through a _linear predictor expression_, 𝛈 = 𝐗𝛃 + 𝐙𝐛, where 𝛃 is the fixed-effects coefficient vector and 𝐗 and 𝐙 are model matrices of the appropriate sizes,

In a `LinearMixedModel` the conditional mean, 𝛍 = 𝔼[𝓨|𝓑 = 𝐛], is the linear predictor, 𝛈, and the conditional distribution is multivariate Gaussian, (𝓨|𝓑 = 𝐛) ~ 𝓝(𝛍, σ²𝐈).

In a `GeneralizedLinearMixedModel`, the conditional mean, 𝔼[𝓨|𝓑 = 𝐛], is related to the linear predictor via a _link function_.
Typical distribution forms are _Bernoulli_ for binary data or _Poisson_ for count data.

## Currently Tested Platforms

|OS      | OS Version    |Arch |Julia           |
|:------:|:-------------:|:---:|:--------------:|
|Linux   | Ubuntu 20.04  | x64 |v1.8            |
|Linux   | Ubuntu 20.04  | x64 |current release |
|Linux   | Ubuntu 20.04  | x64 |nightly         |
|macOS   | Monterey 12   | x64 |v1.8            |
|Windows | Server 2019   | x64 |v1.8            |

Note that previous releases still support older Julia versions.

## Version 4.0.0

Version 4.0.0 contains some user-visible changes and many changes in the underlying code.

Please see [NEWS](NEWS.md) for a complete overview, but a few key points are:

- The internal storage of the model matrices in `LinearMixedModel` has changed and been optimized. This change should be transparent to users who are not manipulating the fields of the model `struct` directly.
- The [handling of rank deficiency](https://juliastats.org/MixedModels.jl/v4.0/rankdeficiency/) continues to evolve.
- Additional [`predict` and `simulate`](https://juliastats.org/MixedModels.jl/v4.0/prediction/) methods have been added for generalizing to new data.
- `saveoptsum` and `restoreoptsum!` provide for saving and restoring the `optsum` and thus offer a way to serialize a model fit.
- There is improved support for the runtime construction of model formula, especially `RandomEffectsTerm`s and nested terms (methods for `Base.|(::AbstractTerm, ::AbstractTerm)` and `Base./(::AbstractTerm, ::AbstractTerm)`).
- A progress display is shown by default for models taking more than a few hundred milliseconds to fit. This can be disabled with the keyword argument `progress=false`.

## Quick Start
```julia-repl
julia> using MixedModels

julia> m1 = fit(MixedModel, @formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff))
Linear mixed model fit by maximum likelihood
 yield ~ 1 + (1 | batch)
   logLik   -2 logLik     AIC       AICc        BIC
  -163.6635   327.3271   333.3271   334.2501   337.5307

Variance components:
            Column    Variance Std.Dev.
batch    (Intercept)  1388.3332 37.2603
Residual              2451.2501 49.5101
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
────────────────────────────────────────────────
              Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────
(Intercept)  1527.5     17.6946  86.33    <1e-99
────────────────────────────────────────────────

julia> using Random

julia> bs = parametricbootstrap(MersenneTwister(42), 1000, m1);
Progress: 100%%|████████████████████████████████████████████████| Time: 0:00:00

julia> propertynames(bs)
13-element Vector{Symbol}:
 :allpars
 :objective
 :σ
 :β
 :se
 :coefpvalues
 :θ
 :σs
 :λ
 :inds
 :lowerbd
 :fits
 :fcnames

julia> bs.coefpvalues # returns a row table
1000-element Vector{NamedTuple{(:iter, :coefname, :β, :se, :z, :p), Tuple{Int64, Symbol, Float64, Float64, Float64, Float64}}}:
 (iter = 1, coefname = Symbol("(Intercept)"), β = 1517.0670832927115, se = 20.76271142094811, z = 73.0669059804057, p = 0.0)
 (iter = 2, coefname = Symbol("(Intercept)"), β = 1503.5781855888436, se = 8.1387737362628, z = 184.7425956676446, p = 0.0)
 (iter = 3, coefname = Symbol("(Intercept)"), β = 1529.2236379016574, se = 16.523824785737837, z = 92.54659001356465, p = 0.0)
 ⋮
 (iter = 998, coefname = Symbol("(Intercept)"), β = 1498.3795009457242, se = 25.649682012258104, z = 58.417079019913054, p = 0.0)
 (iter = 999, coefname = Symbol("(Intercept)"), β = 1526.1076747922416, se = 16.22412120273579, z = 94.06411945042063, p = 0.0)
 (iter = 1000, coefname = Symbol("(Intercept)"), β = 1557.7546433870125, se = 12.557577103806015, z = 124.04898098653763, p = 0.0)

julia> using DataFrames

julia> DataFrame(bs.coefpvalues) # puts it into a DataFrame
1000×6 DataFrame
│ Row  │ iter  │ coefname    │ β       │ se      │ z       │ p       │
│      │ Int64 │ Symbol      │ Float64 │ Float64 │ Float64 │ Float64 │
├──────┼───────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ 1    │ 1     │ (Intercept) │ 1517.07 │ 20.7627 │ 73.0669 │ 0.0     │
│ 2    │ 2     │ (Intercept) │ 1503.58 │ 8.13877 │ 184.743 │ 0.0     │
│ 3    │ 3     │ (Intercept) │ 1529.22 │ 16.5238 │ 92.5466 │ 0.0     │
⋮
│ 998  │ 998   │ (Intercept) │ 1498.38 │ 25.6497 │ 58.4171 │ 0.0     │
│ 999  │ 999   │ (Intercept) │ 1526.11 │ 16.2241 │ 94.0641 │ 0.0     │
│ 1000 │ 1000  │ (Intercept) │ 1557.75 │ 12.5576 │ 124.049 │ 0.0     │

julia> DataFrame(bs.β)
1000×3 DataFrame
│ Row  │ iter  │ coefname    │ β       │
│      │ Int64 │ Symbol      │ Float64 │
├──────┼───────┼─────────────┼─────────┤
│ 1    │ 1     │ (Intercept) │ 1517.07 │
│ 2    │ 2     │ (Intercept) │ 1503.58 │
│ 3    │ 3     │ (Intercept) │ 1529.22 │
⋮
│ 998  │ 998   │ (Intercept) │ 1498.38 │
│ 999  │ 999   │ (Intercept) │ 1526.11 │
│ 1000 │ 1000  │ (Intercept) │ 1557.75 │
```

## Funding Acknowledgement

The development of this package was supported by the Center for Interdisciplinary Research, Bielefeld (ZiF)/Cooperation Group "Statistical models for psychological and linguistic data".
