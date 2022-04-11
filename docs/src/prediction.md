# Prediction and simulation in Mixed-Effects Models

We recommend the [MixedModelsSim.jl](https://github.com/RePsychLing/MixedModelsSim.jl/) package and associated documentation for useful tools in constructing designs to simulate. For now, we'll use the sleep study data as a starting point.

```@example Main
using DataFrames
using MixedModels
using StatsBase
using DisplayAs # hide
# use a DataFrame to make it easier to change things later
slp = DataFrame(MixedModels.dataset(:sleepstudy))
slpm = fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj)), slp)
DisplayAs.Text(slpm) # hide
```

## Prediction

The simplest form of prediction are the fitted values from the model: they are indeed the model's predictions for the observed data.

```@example Main
predict(slpm) ≈ fitted(slpm)
```

When generalizing to new data, we need to consider what happens if there are new, previously unobserved levels of the grouping variable(s).
MixedModels.jl provides three options:

1. `:error`: error on encountering unobserved levels
2. `:population`: use population values (i.e. only the fixed effects) for observations with unobserved levels
3. `:missing`: return `missing` for observations with unobserved levels.

Providing either no prediction (`:error`, `:missing`) or providing the population-level values seem to be the most reasonable ways for *predicting* new values.
For *simulating* new values based on previous estimates of the variance components, use `simulate`.

In the case where there are no new levels of the grouping variable, all three of these methods provide the same results:

```@example Main
predict(slpm, slp; new_re_levels=:population) ≈ fitted(slpm)
```

```@example Main
predict(slpm, slp; new_re_levels=:missing) ≈ fitted(slpm)
```

```@example Main
predict(slpm, slp; new_re_levels=:error) ≈ fitted(slpm)
```

In the case where there are new levels of the grouping variable, these methods differ.

```@example Main
# create a new level
slp2 = transform(slp, :subj => ByRow(x -> (x == "S308" ? "NEW" : x)) => :subj)
DisplayAs.Text(ans) # hide
```

```@example Main
try
  predict(slpm, slp2; new_re_levels=:error)
catch e
  show(e)
end
```

```@example Main
predict(slpm, slp2; new_re_levels=:missing)
```

```@example Main
predict(slpm, slp2; new_re_levels=:population)
```

!!! note
    Currently, we do not support predicting based on a subset of the random effects.


!!! note
    `predict` is deterministic (within the constraints of floating point) and never adds noise to the result.
    If you want to construct prediction intervals, then `simulate` will generate new data with noise (including new values of the random effects).

For generalized linear mixed models, there is an additional keyword argument to `predict`: `type` specifies whether the predictions are returned on the scale of the linear predictor (`:linpred`) or on the level of the response `(:response)` (i.e. the level at which the values were originally observed).

```@example Main
cbpp = DataFrame(MixedModels.dataset(:cbpp))
cbpp.rate = cbpp.incid ./ cbpp.hsz
gm = fit(MixedModel, @formula(rate ~ 1 + period + (1|herd)), cbpp, Binomial(), wts=float(cbpp.hsz))
predict(gm, cbpp; type=:response) ≈ fitted(gm)
```

```@example Main
logit(x) = log(x / (1 - x))
predict(gm, cbpp; type=:linpred) ≈ logit.(fitted(gm))
```

## Simulation

In contrast to `predict`, `simulate` and `simulate!` introduce randomness.
This randomness occurs both at the level of the observation-level (residual) variance and at the level of the random effects, where new conditional modes are sampled based on the specified covariance parameter (θ; see [Details of the parameter estimation](@ref)), which defaults to the estimated value of the model.
For reproducibility, we specify a pseudorandom generator here; if none is provided, the global PRNG is taken as the default.

The simplest example of `simulate` takes a fitted model and generates a new response vector based on the existing model matrices combined with noise.

```@example Main
using Random
ynew = simulate(MersenneTwister(42), slpm)
```

The simulated response can also be placed in a pre-allocated vector:

```@example Main
ynew2 = zeros(nrow(slp))
simulate!(MersenneTwister(42), ynew2, slpm)
ynew2 ≈ ynew
```

Or even directly replace the previous response vector in a model, at which point the model must be refit to the new values:

```@example Main
slpm2 = deepcopy(slpm)
refit!(simulate!(MersenneTwister(42), slpm2))
DisplayAs.Text(ans) # hide
```

This inplace simulation actually forms the basis of [`parametricbootstrap`](@ref).

Finally, we can also simulate the response from entirely new data.
```@example Main
df = DataFrame(days = repeat(1:10, outer=20), subj=repeat(1:20, inner=10))
df[!, :subj] = string.("S", lpad.(df.subj, 2, "0"))
df[!, :reaction] .= 0
df
DisplayAs.Text(df) # hide
```

```@example Main
ysim = simulate(MersenneTwister(42), slpm, df)
```

Note that this is a convenience method for creating a new model and then using the parameters from the old model to call `simulate` on that model.
In other words, this method incurs the cost of constructing a new model and then discarding it.
If you could re-use that model (e.g., fitting that model as part of a simulation study), it often makes sense to do these steps to perform these steps explicitly and avoid the unnecessary construction and discarding of an intermediate model:

```@example Main
msim = LinearMixedModel(@formula(reaction ~ 1 + days + (1|subj)), df)
simulate!(MersenneTwister(42), msim; θ=slpm.θ, β=slpm.β, σ=slpm.σ)
response(msim) ≈ ysim
```

```@example Main
fit!(msim)
DisplayAs.Text(ans) # hide
```

For simulating from generalized linear mixed models, there is no `type` option because the observation-level always occurs at the level of the response and not of the linear predictor.

!!! warning
    Simulating the model response in place may not yield the same result as simulating into a pre-allocated or new vector, depending on choice of pseudorandom number generator.
    Random number generation in Julia allows optimization based on type, and the internal storage type of the model response (currently a view into a matrix storing the concatenated fixed-effects model matrix and the response) may not match the type of a pre-allocated or new vector.
    See also [discussion here](https://discourse.julialang.org/t/weird-prng-behavior/63186).

!!! note
    All the methods that take new data as a table construct an additional `MixedModel` behind the scenes, even when the new data is exactly the same as the data that the model was fitted to.
    For the simulation methods in particular, these thus form a convenience wrapper for constructing a new model and calling `simulate` without new data on that model with the parameters from the original model.
