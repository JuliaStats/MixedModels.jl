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

```@setup Main
using DisplayAs
```

```@example Main
using DataFrames
using Gadfly          # plotting package
using MixedModels
using Random
```

```@example Main
dyestuff = MixedModels.dataset(:dyestuff)
m1 = fit(MixedModel, @formula(yield ~ 1 + (1 | batch)), dyestuff)
DisplayAs.Text(ans) # hide
```

To bootstrap the model parameters, first initialize a random number generator then create a bootstrap sample

```@example Main
const rng = MersenneTwister(1234321);
samp = parametricbootstrap(rng, 10_000, m1);
df = DataFrame(samp.allpars);
first(df, 10)
```

Especially for those with a background in [`R`](https://www.R-project.org/) or [`pandas`](https://pandas.pydata.org),
the simplest way of accessing the parameter estimates in the parametric bootstrap object is to create a `DataFrame` from the `allpars` property as shown above.

We can use `filter` to filter out relevant rows of a dataframe.
A density plot of the estimates of `σ`, the residual standard deviation, can be created as
```@example Main
σres = filter(df) do row # create a thunk that operates on rows
    row.type == "σ" && row.group == "residual" # our filtering rule
end

plot(x = σres.value, Geom.density, Guide.xlabel("Parametric bootstrap estimates of σ"))
```
For the estimates of the intercept parameter, the `getproperty` extractor must be used
```@example Main
plot(filter(:type => ==("β"),  df), x = :value, Geom.density, Guide.xlabel("Parametric bootstrap estimates of β₁"))
```

A density plot of the estimates of the standard deviation of the random effects is obtained as
```@example Main
σbatch = filter(df) do row # create a thunk that operates on rows
    row.type == "σ" && row.group == "batch" # our filtering rule
end
plot(x = σbatch.value, Geom.density,
    Guide.xlabel("Parametric bootstrap estimates of σ₁"))
```

Notice that this density plot has a spike, or mode, at zero.
Although this mode appears to be diffuse, this is an artifact of the way that density plots are created.
In fact, it is a pulse, as can be seen from a histogram.

```@example Main
plot(x = σbatch.value, Geom.histogram,
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
combine(groupby(df, [:type, :group, :names]), :value => shortestcovint => :interval)
```

We can also generate this directly from the original bootstrap object:
```@example Main
DataFrame(shortestcovint(samp))
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
DisplayAs.Text(ans) # hide
```
```@example Main
samp2 = parametricbootstrap(rng, 10_000, m2);
df2 = DataFrame(samp2.allpars);
first(df2, 10)
```
the singularity can be exhibited as a standard deviation of zero or as a correlation of $\pm1$.

```@example Main
DataFrame(shortestcovint(samp2))
```

A histogram of the estimated correlations from the bootstrap sample has a spike at `+1`.
```@example Main
ρs = filter(df2) do row
    row.type == "ρ" && row.group == "subj"
end
plot(x = ρs.value, Geom.histogram,
    Guide.xlabel("Parametric bootstrap samples of correlation of random effects"))
```
or, as a count,
```@example Main
count(ρs.value .≈ 1)
```

Close examination of the histogram shows a few values of `-1`.
```@example Main
count(ρs.value .≈ -1)
```

Furthermore there are even a few cases where the estimate of the standard deviation of the random effect for the intercept is zero.
```@example Main
σs = filter(df2) do row
    row.type == "σ" && row.group == "subj" && row.names == "(Intercept)"
end
count(σs.value .≈ 0)
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
t = @timed parametricbootstrap(MersenneTwister(42), 1000, m2; hide_progress=true)
t.time
```

```@example Main
optsum_overrides = (; ftol_rel=1e-8)
t = @timed parametricbootstrap(MersenneTwister(42), 1000, m2; optsum_overrides, hide_progress=true)
t.time
```

## Distributed Computing and the Bootstrap

Earlier versions of MixedModels.jl supported a multi-threaded bootstrap via the `use_threads` keyword argument.
However, with improved BLAS multithreading, the Julia-level threads often wound up competing with the BLAS threads, leading to no improvement or even a worsening of performance when `use_threads=true`.
Nonetheless, the bootstrap is a classic example of an [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) problem and so we provide a few convenience methods for combining results computed separately.
In particular, there are `vcat` and an optimized `reduce(::typeof(vcat))` methods for `MixedModelBootstrap` objects.
For computers with many processors (and not a few cores in a single processor) or computing clusters, these provide a convenient way to split the computation across nodes.

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

# 10 replicates computed on each work
n_replicates = 1000
n_rep_per_worker = n_replicates ÷ nworkers()
pb_map = @showprogress pmap(MersenneTwister.(1:nworkers())) do rng
    parametricbootstrap(rng, n_rep_per_worker, m2; optsum_overrides)
end;

# get rid of all the workers
# rmprocs(workers())

confint(reduce(vcat, pb_map))
```
