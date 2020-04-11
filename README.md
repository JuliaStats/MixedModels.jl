# Mixed-effects models in Julia

|**Documentation**|**Citation**|**Build Status**|**Code Coverage**|
|:-:|:-:|:-:|:-:|
|[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][doi-img]][doi-url] | [![T1-url][T1-img]][pkgeval-url] [![T2-url][T2-img]][pkgeval-url] [![PkgEval][pkgeval-img]][pkgeval-url] | [![][codecov-img]][codecov-url]|

[doi-img]: https://zenodo.org/badge/9106942.svg
[doi-url]: https://zenodo.org/badge/latestdoi/9106942

[docs-latest-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-latest-url]: https://juliastats.github.io/MixedModels.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliastats.github.io/MixedModels.jl/stable

[codecov-img]: https://codecov.io/github/JuliaStats/MixedModels.jl/badge.svg?branch=master
[codecov-url]: https://codecov.io/github/JuliaStats/MixedModels.jl?branch=master

[T1-img]: https://github.com/JuliaStats/MixedModels.jl/workflows/Tier1/badge.svg
[T1-url]: https://github.com/JuliaStats/MixedModels.jl/actions?workflow=Tier1

[T2-img]: https://github.com/JuliaStats/MixedModels.jl/workflows/Tier2/badge.svg
[T2-url]: https://github.com/JuliaStats/MixedModels.jl/actions?workflow=Tier2

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

## Currently Supported Platforms

|OS|OS Version|Arch|Julia|Tier|
|:-:|:-:|:-:|:-:|:-:|
|Linux|Ubuntu 18.04|x64|v1.4|1|
|macOS|Catalina 10.15|x64|v1.4|1|
|Windows|Server 2019|x64|v1.4|1|
|Linux|Ubuntu 18.04|x86|v1.4|2|
|Windows|Server 2019|x86|v1.4|2|

## Version 2.0.0

Version 2.0.0 contains some user-visible changes and many changes in the underlying code.

The user-visible changes include:

- Update formula specification to `StatsModels v"0.6.2"`, allowing for function calls within the fixed-effects terms and for interaction terms on the left-hand side of a random-effects term.

- Use of properties in a model in addition to extractor functions.  For example, to obtain the covariance parameter, $\theta$, from a model, the recommended approach now is to access the `θ` property, as in `m.θ`, instead of the extractor `getθ(m)`.

- `bootstrap` is now named `parametricbootstrap` to avoid conflict with a similar name in the `Bootstrap` package.  The bootstrap sample is returned as a `Table`.

- A `fit` method for the abstract type `MixedModel` has been added.  It is called as

```
julia> using Tables, MixedModels

julia> Dyestuff = columntable((batch = string.(repeat('A':'F', inner=5)),
       yield = [1545, 1440, 1440, 1520, 1580, 1540, 1555, 1490, 1560, 1495, 1595, 1550, 1605,
        1510, 1560, 1445, 1440, 1595, 1465, 1545, 1595, 1630, 1515, 1635, 1625, 1520, 1455,
        1450, 1480, 1445]));

julia> m1 = fit(MixedModel, @formula(yield ~ 1 + (1|batch)), Dyestuff)
    Linear mixed model fit by maximum likelihood
     yield ~ 1 + (1 | batch)
       logLik   -2 logLik     AIC        BIC    
     -163.66353  327.32706  333.32706  337.53065

        Variance components:
                  Column    Variance  Std.Dev.
     batch    (Intercept)  1388.3334 37.260347
     Residual              2451.2500 49.510100
     Number of obs: 30; levels of grouping factors: 6

     Fixed-effects parameters:
    ──────────────────────────────────────────────────
                 Estimate  Std.Error  z value  P(>|z|)
    ──────────────────────────────────────────────────
    (Intercept)    1527.5    17.6946   86.326   <1e-99
    ──────────────────────────────────────────────────
```

The development of this package was supported by the Center for Interdisciplinary Research, Bielefeld (ZiF)/Cooperation Group "Statistical models for psychological and linguistic data".
