# Parametric bootstrap for mixed-effects models

Julia is well-suited to implementing bootstrapping and other simulation-based methods for statistical models.
The `parametricbootstrap` function in the [MixedModels package](https://github.com/JuliaStats/MixedModels.jl) provides an efficient parametric bootstrap for mixed-effects models.

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
using Gadfly          # plotting package
using MixedModels
using Random
```

```@example Main
dyestuff = MixedModels.dataset(:dyestuff)
m1 = fit(MixedModel, @formula(yield ~ 1 + (1 | batch)), dyestuff)
```

To bootstrap the model parameters, first initialize a random number generator then create a bootstrap sample and extract the `tbl` property, which is a `Table` - a lightweight dataframe-like object.

```@example Main
const rng = MersenneTwister(1234321);
samp = parametricbootstrap(rng, 10_000, m1);
tbl = samp.tbl
```

A density plot of the estimates of `σ`, the residual standard deviation, can be created as
```@example Main
plot(x = tbl.σ, Geom.density, Guide.xlabel("Parametric bootstrap estimates of σ"))
```

or, for the intercept parameter
```@example Main
plot(x = tbl.β1, Geom.density, Guide.xlabel("Parametric bootstrap estimates of β₁"))
```

A density plot of the estimates of the standard deviation of the random effects is obtained as
```@example Main
plot(x = tbl.σ1, Geom.density,
    Guide.xlabel("Parametric bootstrap estimates of σ₁"))
```

Notice that this density plot has a spike, or mode, at zero.
Although this mode appears to be diffuse, this is an artifact of the way that density plots are created.
In fact, it is a pulse, as can be seen from a histogram.

```@example Main
plot(x = tbl.σ1, Geom.histogram,
    Guide.xlabel("Parametric bootstrap estimates of σ₁"))
```

The bootstrap sample can be used to generate intervals that cover a certain percentage of the bootstrapped values.
We refer to these as "coverage intervals", similar to a confidence interval.
The shortest such intervals, obtained with the `shortestcovint` extractor, correspond to a highest posterior density interval in Bayesian inference.

```@docs
shortestcovint
```

We generate these directly from the original bootstrap object:
```@example Main
Table(shortestcovint(samp))
```

A value of zero for the standard deviation of the random effects is an example of a *singular* covariance.
It is easy to detect the singularity in the case of a scalar random-effects term.
However, it is not as straightforward to detect singularity in vector-valued random-effects terms.

For example, if we bootstrap a model fit to the `sleepstudy` data
```@example Main
sleepstudy = MixedModels.dataset(:sleepstudy)
contrasts = Dict(:subj => Grouping())
m2 = let f = @formula reaction ~ 1+days+(1+days|subj)
    fit(MixedModel, f, sleepstudy; contrasts)
end
```
```@example Main
samp2 = parametricbootstrap(rng, 10_000, m2);
tbl2 = samp2.tbl
```
the singularity can be exhibited as a standard deviation of zero or as a correlation of $\pm1$.

```@example Main
shortestcovint(samp2)
```

A histogram of the estimated correlations from the bootstrap sample has a spike at `+1`.
```@example Main
plot(x = tbl2.ρ1, Geom.histogram,
    Guide.xlabel("Parametric bootstrap samples of correlation of random effects"))
```
or, as a count,
```@example Main
count(tbl2.ρ1 .≈ 1)
```

Close examination of the histogram shows a few values of `-1`.
```@example Main
count(tbl2.ρ1 .≈ -1)
```

Furthermore there are even a few cases where the estimate of the standard deviation of the random effect for the intercept is zero.
```@example Main
count(tbl2.σ1 .≈ 0)
```

There is a general condition to check for singularity of an estimated covariance matrix or matrices in a bootstrap sample.
The parameter optimized in the estimation is `θ`, the relative covariance parameter.
Some of the elements of this parameter vector must be non-negative and, when one of these components is approximately zero, one of the covariance matrices will be singular.

The `issingular` method for a `MixedModel` object that tests if a parameter vector `θ` corresponds to a boundary or singular fit.

This operation is encapsulated in a method for the `issingular` function.
```@example Main
count(issingular(samp2))
```

## Reduced Precision Bootstrap

`parametricbootstrap` accepts an optional keyword argument `optsum_overrides`, which can be used to override the convergence criteria for bootstrap replicates. One possibility is setting `ftol_rel=1e-8`, i.e., considering the model converged when the relative change in the objective between optimizer iterations is smaller than 0.00000001.
This threshold corresponds approximately to the precision from treating the value of the objective as a single precision (`Float32`) number, while not changing the precision of the intermediate computations.
The resultant loss in precision will generally be smaller than the variation that the bootstrap captures, but can greatly speed up the fitting process for each replicates, especially for large models.
More directly, lowering the fit quality for each replicate will reduce the quality of each replicate, but this may be more than compensated for by the ability to fit a much larger number of replicates in the same time.

```@example Main
t = @timed parametricbootstrap(MersenneTwister(42), 1000, m2; progress=false)
t.time
```

```@example Main
optsum_overrides = (; ftol_rel=1e-8)
t = @timed parametricbootstrap(MersenneTwister(42), 1000, m2; optsum_overrides, progress=false)
t.time
```

## Distributed Computing and the Bootstrap

Earlier versions of MixedModels.jl supported a multi-threaded bootstrap via the `use_threads` keyword argument.
However, with improved BLAS multithreading, the Julia-level threads often wound up competing with the BLAS threads, leading to no improvement or even a worsening of performance when `use_threads=true`.
Nonetheless, the bootstrap is a classic example of an [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) problem and so we provide a few convenience methods for combining results computed separately.
In particular, there are `vcat` and an optimized `reduce(::typeof(vcat))` methods for `MixedModelBootstrap` objects.
For computers with many processors (as opposed to a single processor with several cores) or for computing clusters, these provide a convenient way to split the computation across nodes.

```@example Main
using Distributed
using ProgressMeter
# you already have 1 proc by default, so add the number of additional cores with `addprocs`
# you need at least as many RNGs as cores you want to use in parallel
# but you shouldn't use all of your cores because nested within this
# is the multithreading of the linear algebra
@info "Currently using $(nprocs()) processors total and $(nworkers()) for work"

# copy everything to workers
@showprogress for w in workers()
    remotecall_fetch(() -> coefnames(m2), w)
end

# split the replicates across the workers
# this rounds down, so if the number of workers doesn't divide the
# number of replicates, you'll be a few replicates short!
n_replicates = 1000
n_rep_per_worker = n_replicates ÷ nworkers()
# NB: You need a different seed/RNG for each worker otherwise you will
# have copies of the same replicates and not independent replicates!
pb_map = @showprogress pmap(MersenneTwister.(1:nworkers())) do rng
    parametricbootstrap(rng, n_rep_per_worker, m2; optsum_overrides)
end;

# get rid of all the workers
# rmprocs(workers())

confint(reduce(vcat, pb_map))
```
