# Parametric bootstrap for linear mixed-effects models

Julia is well-suited to implementing bootstrapping and other simulation-based methods for statistical models.
The `parametricbootstrap` function in the [MixedModels package](https://github.com/JuliaStats/MixedModels.jl) provides an efficient parametric bootstrap for linear mixed-effects models.

```@docs
parametricbootstrap
```

## The parametric bootstrap

[Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) is a family of procedures
for generating sample values of a statistic, allowing for visualization of the distribution of the
statistic or for inference from this sample of values.

A _parametric bootstrap_ is used with a parametric model, `m`, that has been fit to data.
The procedure is to simulate `n` response vectors from `m` using the estimated parameter values
and refit `m` to these responses in turn, accumulating the statistics of interest at each iteration.

The parameters of a `LinearMixedModel` object are the fixed-effects
parameters, `β`, the standard deviation, `σ`, of the per-observation noise, and the covariance
parameter, `θ`, that defines the variance-covariance matrices of the random effects.

For example, a simple linear mixed-effects model for the `Dyestuff` data in the [`lme4`](http://github.com/lme4/lme4)
package for [`R`](https://www.r-project.org) is fit by

```@example Main
using DataFrames, Gadfly, MixedModels, Random
```

```@example Main
dyestuff = MixedModels.dataset(:dyestuff)
m1 = fit(MixedModel, @formula(yield ~ 1 + (1 | batch)), dyestuff)
```

To bootstrap the model parameters, first initialize a random number generator

```@example Main
const rng = MersenneTwister(1234321);
```

then create a bootstrap sample

```@example Main
samp = parametricbootstrap(rng, 10_000, m1);
propertynames(samp)
```

As shown above, the sample has several named properties, which allow for convenient extraction of information.  For example, a density plot of the estimates of `σ`, the residual standard deviation, can be created as
```@example Main
plot(x=samp.σ, Geom.density, Guide.xlabel("Parametric bootstrap estimates of σ"))
```
For the estimates of the intercept parameter, the `getproperty` extractor must be used
```@example Main
plot(x = first.(samp.β), Geom.density, Guide.xlabel("Parametric bootstrap estimates of β₁"))
```

The `σs` property contains the estimates of the standard deviation of the random effects in a hierarchical format.
```@example Main
typeof(samp.σs)
```

This is to allow for random effects associated with more than one grouping factor.
If we only have one grouping factor for random effects, which is the case here, we can use the `first` extractor, as in
```@example Main
first(samp.σs)
```
or, to carry this one step further,
```@example Main
plot(x=first.(first(samp.σs)), Geom.density,
    Guide.xlabel("Parametric bootstrap estimates of σ₁"))
```

Notice that this density plot has a spike, or mode, at zero.
Although this mode appears to be diffuse, this is an artifact of the way that density plots are created.
In fact, it is a pulse, as can be seen from a histogram.

```@example Main
plot(x=first.(first(samp.σs)), Geom.histogram,
    Guide.xlabel("Parametric bootstrap estimates of σ₁"))
```

A value of zero for the standard deviation of the random effects is an example of a *singular* covariance.
It is easy to detect the singularity in the case of a scalar random-effects term.
However, it is not as straightforward to detect singularity in vector-valued random-effects terms.

For example, if we bootstrap a model fit to the `sleepstudy` data
```@example Main
sleepstudy = MixedModels.dataset(:sleepstudy)
m2 = fit(
    MixedModel,
    @formula(reaction ~ 1+days+(1+days|subj)), 
    sleepstudy,
)
```
```@example Main
samp2 = parametricbootstrap(rng, 10_000, m2, use_threads=true);
```
the singularity can be exhibited as a standard deviation of zero or as a correlation of $\pm1$.
The `σρs` property of the sample is a vector of named tuples
```@example Main
σρ = first(samp2.σρs);
typeof(σρ)
```
where the first element of the tuple is itself a tuple of standard deviations and the second (also the last) element of the tuple is the correlation.

A histogram of the estimated correlations from the bootstrap sample has a spike at `+1`.
```@example Main
ρs = first.(last.(σρ))
plot(x = ρs, Geom.histogram,
    Guide.xlabel("Parametric bootstrap samples of correlation of random effects"))
```
or, as a count,
```@example Main
sum(ρs .≈ 1)
```

Close examination of the histogram shows a few values of `-1`.
```@example Main
sum(ρs .≈ -1)
```

Furthermore there are even a few cases where the estimate of the standard deviation of the random effect for the intercept is zero.
```@example Main
sum(first.(first.(first.(σρ))) .≈ 0)
```

There is a general condition to check for singularity of an estimated covariance matrix or matrices in a bootstrap sample.
The parameter optimized in the estimation is `θ`, the relative covariance parameter.
Some of the elements of this parameter vector must be non-negative and, when one of these components is approximately zero, one of the covariance matrices will be singular.

The boundary values are available as
```@example Main
samp2.m.optsum.lowerbd
```
so the check on singularity becomes
```@example Main
sum(θ -> any(θ .≈ samp2.m.optsum.lowerbd), samp2.θ)
```

The `issingular` method for a `LinearMixedModel` object that tests if a parameter vector `θ` corresponds to a boundary or singular fit.
The default value of `θ` is that from the model but another value can be given as the second argument.

This operation is encapsulated in a method for the `issingular` function.
```@example Main
sum(issingular(samp2))
```
