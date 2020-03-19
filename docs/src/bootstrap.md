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
using DataFrames
using DataFramesMeta  # dplyr-like operations
using Gadfly          # plotting package
using MixedModels
using Random
```

```@example Main
dyestuff = MixedModels.dataset(:dyestuff)
m1 = fit(MixedModel, @formula(yield ~ 1 + (1 | batch)), dyestuff)
```

To bootstrap the model parameters, first initialize a random number generator then create a bootstrap sample

```@example Main
const rng = MersenneTwister(1234321);
samp = parametricbootstrap(rng, 10_000, m1, use_threads=true);
df = DataFrame(samp.allpars);
first(df, 10)
```

Especially for those with a background in [`R`](https://www.R-project.org/) or [`pandas`](https://pandas.pydata.org),
the simplest way of accessing the parameter estimates in the parametric bootstrap object is to create a `DataFrame` from the `allpars` property as shown above.

The [`DataFramesMeta`](https://github.com/JuliaData/DataFramesMeta.jl) package provides macros for extracting rows or columns of a dataframe.
A density plot of the estimates of `σ`, the residual standard deviation, can be created as
```@example Main
σres = @where(df, :type .== "σ", :group .== "residual").value
plot(x = σres, Geom.density, Guide.xlabel("Parametric bootstrap estimates of σ"))
```
For the estimates of the intercept parameter, the `getproperty` extractor must be used
```@example Main
plot(@where(df, :type .== "β"), x = :value, Geom.density, Guide.xlabel("Parametric bootstrap estimates of β₁"))
```

A density plot of the estimates of the standard deviation of the random effects is obtained as
```@example Main
σbatch = @where(df, :type .== "σ", :group .== "batch").value
plot(x = σbatch, Geom.density,
    Guide.xlabel("Parametric bootstrap estimates of σ₁"))
```

Notice that this density plot has a spike, or mode, at zero.
Although this mode appears to be diffuse, this is an artifact of the way that density plots are created.
In fact, it is a pulse, as can be seen from a histogram.

```@example Main
plot(x = σbatch, Geom.histogram,
    Guide.xlabel("Parametric bootstrap estimates of σ₁"))
```

The bootstrap sample can be used to generate intervals that cover a certain percentage of the bootstrapped values.
We refer to these as "coverage intervals", similar to a confidence interval.
The shortest such intervals, obtained with the `shortestcovint` extractor, correspond to a highest posterior density interval in Bayesian inference.

```@docs
shortestcovint
```

We generate these for all random and fixed effects:
```@example Main
by(df, [:type, :group, :names], interval = :value => shortestcovint)
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
df2 = DataFrame(samp2.allpars);
first(df2, 10)
```
the singularity can be exhibited as a standard deviation of zero or as a correlation of $\pm1$.

```@example Main
by(df2, [:type,:group,:names], interval = :value=>shortestcovint)
```

A histogram of the estimated correlations from the bootstrap sample has a spike at `+1`.
```@example Main
ρs = @where(df2, :type .== "ρ", :group .== "subj").value
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
σs = @where(df2, :type .== "σ", :group .== "subj", :names .== "(Intercept)").value 
sum(σs .≈ 0)
```

There is a general condition to check for singularity of an estimated covariance matrix or matrices in a bootstrap sample.
The parameter optimized in the estimation is `θ`, the relative covariance parameter.
Some of the elements of this parameter vector must be non-negative and, when one of these components is approximately zero, one of the covariance matrices will be singular.

The `issingular` method for a `LinearMixedModel` object that tests if a parameter vector `θ` corresponds to a boundary or singular fit.

This operation is encapsulated in a method for the `issingular` function.
```@example Main
sum(issingular(samp2))
```
