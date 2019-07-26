# Mixed-effects models in Julia

|**Documentation**|**Citation**|**Build Status**|**Code Coverage**|
|:-:|:-:|:-:|:-:|
|[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][doi-img]][doi-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] | [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url]|

[doi-img]: https://zenodo.org/badge/9106942.svg
[doi-url]: https://zenodo.org/badge/latestdoi/9106942

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://dmbates.github.io/MixedModels.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://dmbates.github.io/MixedModels.jl/stable

[travis-img]: https://travis-ci.org/dmbates/MixedModels.jl.svg?branch=master
[travis-url]: https://travis-ci.org/dmbates/MixedModels.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/dmbates/MixedModels.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/dmbates/mixedmodels-jl

[coveralls-img]: https://coveralls.io/repos/github/dmbates/MixedModels.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/dmbates/MixedModels.jl?branch=master

[codecov-img]: https://codecov.io/github/dmbates/MixedModels.jl/badge.svg?branch=master
[codecov-url]: https://codecov.io/github/dmbates/MixedModels.jl?branch=master

This package defines the `LinearMixedModel` and `GeneralizedLinearMixedModel` types and methods to `fit!` them and examine the results.

A _mixed-effects model_ is a statistical model for a _response_ variable as a function of one or more _covariates_.
For a categorical covariate the coefficients associated with the levels of the covariate are sometimes called _effects_, as in "the effect of using Treatment 1 versus the placebo".
If the potential levels of the covariate are fixed and reproducible, e.g. the levels for `Sex` could be `"F"` and `"M"`, they are modeled with _fixed-effects_ parameters.
If the levels constitute a sample from a population, e.g. the `Subject` or the `Item` at a particular observation, they are modeled as _random effects_.

A _mixed-effects_ model contains both fixed-effects and random-effects terms.

With fixed-effects it is the coefficients themselves or combinations of coefficients that are of interest.
For random effects it is the variability of the effects over the population that is of interest.

In this package random effects are modeled as independent samples from a multivariate Gaussian distribution of the form $\mathcal{B}\sim\mathcal{N}(\mathbf{0},\mathbf{\Sigma})$.
For the response vector, $\mathbf{y}$, only the mean of conditional distribution, $\mathcal{Y}|\mathcal{B} = \mathbf{b}$ depends on $\mathbf{b}$ and it does so through a _linear predictor expression_, $\mathbf{\eta}=\mathbf{X\beta+Zb}$ where $\mathbf{\beta}$ is the fixed-effects coefficient vector and $\mathbf{X}$ and $\mathbf{Z}$ are model matrices of the appropriate sizes,

In a `LinearMixedModel` the conditional mean, $\mathbf{\mu}_{\mathcal{Y}|\mathcal{B}=\mathbf{b}}$ is the linear predictor, $\mathbf{\eta}$, and the conditional distribution is multivariate Gaussian, $(\mathcal{Y}|\mathcal{B}=\mathbf{b})\sim\mathcal{N}(\mathbf{\mu}, \sigma^2\mathbf{I})$.

In a `GeneralizedLinearMixedModel`, the conditional mean, $\mathbf{\mu}_{\mathcal{Y}|\mathcal{B}=\mathbf{b}}$, is related to the linear predictor via a _link function_.
Typical distribution forms are _Bernoulli_ for binary data or _Poisson_ for count data.

## Version 2.0.0

Version 2.0.0, to be released on Aug. 1, 2019 contains some user-visible changes and many changes in the underlying code.

The user-visible changes include:

- Update formula specification to `StatsModels v"0.6.2", allowing for function calls within the fixed-effects terms and for interaction terms on the left-hand side of a random-effects term.

- Use of properties in a model in addition to extractor functions.  For example, to obtain the covariance parameter, $\theta$, from a model, the recommended approach now is to access the `θ` property, as in `m.θ`, instead of the extractor `getθ(m)`.

- `bootstrap` is now named `parametricbootstrap` to avoid conflict with a similar name in the `Bootstrap` package.  The bootstrap sample is returned as a `Table`.
