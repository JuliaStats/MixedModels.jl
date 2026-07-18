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

|OS      | OS Version    |Arch    |Julia           |
|:------:|:-------------:|:------:|:--------------:|
|Linux   | Ubuntu 22.04  | x64    |v1.10           |
|Linux   | Ubuntu 24.04  | x64    |current release |
|Linux   | Ubuntu 22.04  | x64    |nightly         |
|macOS   | Sonoma 14     | aarm64 |v1.10           |
|macOS   | Sequoia 15    | aarm64 |current release |
|Windows | Server 2022   | x64    |v1.10           |

Note that previous releases still support older Julia versions.

## Version 5.0

Version 5.0.0 contains some user-visible changes and many changes in the underlying code.

Please see [NEWS](NEWS.md) for a complete overview, but a few key points are:
- Options related to multithreading in the bootstrap have been completely removed.
- Model fitting now uses unconstrained optimization, with a post-fit canonicalization step so that the diagonal elements of the lower Cholesky factor are non-negative. Relatedly, support for constrained optimization has been completely removed and the `lowerbd` field of `OptSummary` dropped.
- The default optimizer has changed to use NLopt's implementation of NEWUOA. Further changes to the default optimizer are considered non-breaking.
- The `profile` function now respects backend and optimizer settings.
- The deprecated `hide_progress` keyword argument has been removed in favor of the shorter and affirmative `progress`.
- A fitlog is always kept and stored as a Tables.jl-compatible column table.

## Quick Start
```julia-repl
julia> using MixedModels

julia> using MixedModelsDatasets: dataset

julia> m1 = lmm(@formula(yield ~ 1 + (1|batch)), dataset(:dyestuff))
Linear mixed model fit by maximum likelihood
 yield ~ 1 + (1 | batch)
   logLik   -2 logLik     AIC       AICc        BIC
  -163.6635   327.3271   333.3271   334.2501   337.5307

Variance components:
            Column    Variance Std.Dev.
batch    (Intercept)  1388.3332 37.2603
Residual              2451.2500 49.5101
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
────────────────────────────────────────────────
              Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────
(Intercept)  1527.5     17.6946  86.33    <1e-99
────────────────────────────────────────────────

julia> using Random

julia> bs = parametricbootstrap(MersenneTwister(42), 1000, m1)
Progress: 100%%|████████████████████████████████████████████████| Time: 0:00:00
MixedModelBootstrap with 1000 samples
     parameter  min      q25       median    mean      q75       max
   ┌────────────────────────────────────────────────────────────────────
 1 │ β1         1474.0   1515.62   1527.68   1527.4    1539.56   1584.57
 2 │ σ          26.6353  43.7165   48.4817   48.8499   53.8964   73.8684
 3 │ σ1         0.0      16.835    28.1067   27.7039   39.491    83.688
 4 │ θ1         0.0      0.340364  0.561701  0.588678  0.840284  2.24396

julia> bs.coefpvalues # returns a row table

julia> using DataFrames

julia> DataFrame(bs.coefpvalues) # puts it into a DataFrame
1000×6 DataFrame
  Row │ iter   coefname     β        se       z         p
      │ Int64  Symbol       Float64  Float64  Float64   Float64
──────┼─────────────────────────────────────────────────────────
    1 │     1  (Intercept)  1552.65   9.8071  158.319       0.0
    2 │     2  (Intercept)  1557.33  21.0679   73.9197      0.0
  ⋮   │   ⋮         ⋮          ⋮        ⋮        ⋮         ⋮
  999 │   999  (Intercept)  1503.1   30.3349   49.5501      0.0
 1000 │  1000  (Intercept)  1565.47  24.5067   63.8794      0.0
                                                996 rows omitted
```

## Funding Acknowledgement

The development of this package was supported by the Center for Interdisciplinary Research, Bielefeld (ZiF)/Cooperation Group "Statistical models for psychological and linguistic data".
