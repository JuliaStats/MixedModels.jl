# Limitations of MixedModels.jl

We expect that MixedModels.jl will generally be best in class for the types of models that it can fit.
We use cutting edge algorithms based on penalized least squares and sparse matrix methods that take advantage of the particular sparsity and structure that arises in the case of the linear mixed effects model with an unconstrained covariance structure.
Glossing over a fair number of technical details, MixedModels.jl uses a different, novel formulation of the underlying numerical problem which tends to be much more efficient computationally and allows us to fit models with multiple crossed, partially crossed or nested grouping variables without any special treatment.

## Very few options for covariance structure

Nonetheless, there is no free lunch and the tradeoff that we make is that it is *much* more difficult to formulate constraints on the covariance structure (whether on the random effects or on the response/residuals) in our formulation. MixedModels.jl currently supports precisely two covariance structures explicitly:

1. unconstrained
2. zero correlation (diagonal covariance structure)

It is also possible to express some models with compound symmetry by clever manipulation of the formula syntax (i.e. `(1+c|g)` for categorical `c` with compound symmetry is the same as `(1|g) + (1|g&c)`).

MixedModels.jl does support constraining the residual variance to known scalar value, which is useful in meta-analysis.

[Metida.jl](https://github.com/PharmCat/Metida.jl) may provide an alternative if this functionality is required (not an endorsement).

## No support for sandwich/robust variance-covariance estimators

[*This may change in the foreseeable future!*](https://github.com/JuliaStats/MixedModels.jl/pull/768)

If this would be a valuable feature, then please [file an issue](https://github.com/JuliaStats/MixedModels.jl/issues/new). Issues are prioritized by the developers' own needs and potential impact for users, so showing a large need for a feature will tend to increase its priority.

[FixedEffectsModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) may be a viable alternative (not an endorsement). It provides "fast estimation of linear models with IV and high dimensional categorical variables" and provides similar functionality to Stata's `reghdfe` and R's `lfe` and `fixest`.

## Generalized linear mixed models with a dispersion parameter

Model families with a dispersion parameter (normal with a non-identity link, gamma, inverse Gaussian) are supported, including `simulate` and `parametricbootstrap`, but there are two things to be aware of.

First, how ϕ is estimated depends on `fast`. With `fast=true` it is plugged in as the Pearson moment estimator `pwrss(m) / nobs(m)`, which is what lme4's `sigma()` reports. With `fast=false` (the default) it is a parameter of the outer optimization and converges to the conditional MLE. The two agree for the normal family, where the moment estimator *is* the MLE, and differ for gamma and inverse Gaussian, where the MLE solves a digamma equation instead.

Second, the results deliberately do not match lme4. `glmer` optimizes a penalized objective that treats ϕ as 1 while reporting `VarCorr` on the scale relative to σ; MixedModels.jl is consistent about the relative scaling throughout, so both θ̂ and the deviance differ. Do not expect agreement to more than the leading digit or two for these families.

Convergence for inverse Gaussian in particular can be poor, and is sensitive to the link — this is a property of the likelihood surface rather than of the implementation.

## No support for polytomous responses

Multinomial and ordered responses are not supported. We are unaware of a Julia package offering support for this.

## No support for regularization of the fixed effects

[HighDimMixedModels.jl](https://github.com/solislemuslab/HighDimMixedModels.jl) may provide an alternative if this functionality is required (not an endorsement).

## No support for generalized additive mixed models

Generalized additive models can be expressed a mixed model, so supporting this would require "only" adding a translation layer.

## No support for nonlinear mixed effects models

[Pumas.jl (commercial)](https://pumas.ai/our-products/products-suite/pumas) provides this (not an endorsement).

## No support for Satterthwaite nor Kenward-Roger degree of freedom approximations.

There are some philosophical and practical [issues](https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#ddf) with these approaches, but an implementation can be found in [MixedModelsSmallSample.jl](https://arnostrouwen.github.io/MixedModelsSmallSample.jl/dev/) (not an endorsement).
