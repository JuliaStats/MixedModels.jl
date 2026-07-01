```@meta
CurrentModule = MixedModels
CollapsedDocStrings = true
```

# Additional Functionality in Other Packages

```@setup Ecosystem
using DisplayAs
```

Several packages extend the functionality of MixedModels.jl, both in ways specific to mixed models and in ways applicable to more general regression models. In the following, we will use the models from the previous sections to showcase this functionality.

```@example Ecosystem
using MixedModels
progress = isinteractive()
```

```@example Ecosystem
insteval = MixedModels.dataset("insteval")
ie1 = fit(MixedModel,
          @formula(y ~ 1 + studage + lectage + service + (1|s) + (1|d) + (1|dept)),
          insteval; progress)

```

```@example Ecosystem
ie2 = fit(MixedModel,
          @formula(y ~ 1 + studage + lectage + service +
                      (1 | s) +
                      (1 + service | d) +
                      (1 + service | dept)),
          insteval; progress)
```

```@example Ecosystem
sleepstudy = MixedModels.dataset("sleepstudy")
ss1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj)), sleepstudy; progress)
```

```@example Ecosystem
ss2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy; progress)
```

```@example Ecosystem
using DataFrames
contra = DataFrame(MixedModels.dataset("contra"))
contra[!, :anych] .= contra[!, :livch] .!= "0"
contrasts = Dict(:livch => EffectsCoding(; base="0"),
                 :urban => HelmertCoding(),
                 :anych => HelmertCoding())
gm1 = fit(MixedModel,
          @formula(use ~ 1 + urban + anych * age + abs2(age) + (1 | dist & urban)),
          contra,
          Bernoulli();
          contrasts,
          progress)
```

## MixedModelsExtras.jl

[https://palday.github.io/MixedModelsExtras.jl/v2](https://palday.github.io/MixedModelsExtras.jl/v2)

MixedModelsExtras.jl is a collection of odds-and-ends that may be useful when working with mixed effects models, but which we do not want to include in MixedModels.jl at this time.
Some functions may one day migrate to MixedModels.jl, when we are happy with their performance and interface (e.g. `vif`), but some are intentionally omitted from MixedModels.jl (e.g. `r2`, `adjr2`).

```@example Ecosystem
using MixedModelsExtras
```

```@example Ecosystem
r2(ss2; conditional=true)
```

```@example Ecosystem
r2(ss2; conditional=false)
```

```@example Ecosystem
icc(ie2)
```

```@example Ecosystem
icc(ie2, :dept)
```

```@example Ecosystem
vif(ie1)
```

```@example Ecosystem
DataFrame(; coef=fixefnames(ie1)[2:end], VIF=vif(ie1))
```

```@example Ecosystem
gvif(ie1)
```

```@example Ecosystem
DataFrame(; term=termnames(ie1)[2][2:end], GVIF=gvif(ie1))
```

## RegressionFormulae.jl

[https://github.com/kleinschmidt/RegressionFormulae.jl](https://github.com/kleinschmidt/RegressionFormulae.jl)

RegressionFormulae.jl provides a few extensions to the somewhat more restricted variant of the Wilkinson-Roger notation found in Julia. In particular, it adds `/` for nested designs within the fixed effects and `^` for computing interactions only up to a certain order.

```@example Ecosystem
using RegressionFormulae

fit(MixedModel,
          @formula(y ~ 1 + service / (studage + lectage) +
                      (1 | s) +
                      (1 | d) +
                      (1 | dept)),
          insteval; progress)
```

```@example Ecosystem
fit(MixedModel,
          @formula(y ~ 1 + (studage + lectage + service)^2 +
                      (1 | s) +
                      (1 | d) +
                      (1 | dept)),
          insteval; progress)
```

## BoxCox.jl

[https://palday.github.io/BoxCox.jl/v0.3/](https://palday.github.io/BoxCox.jl/v0.3/)

BoxCox.jl implements a the Box-Cox transformation in an efficient way. Via package extensions, it supports specializations for MixedModels.jl and several plotting functions, but does not incur a dependency penalty for this functionality when MixedModels.jl or Makie.jl are not loaded.


```@example Ecosystem
using BoxCox

bc = fit(BoxCoxTransformation, ss2)
```

```@example Ecosystem
using CairoMakie
boxcoxplot(bc; conf_level=0.95)
```

The estimated λ is very close to -1, i.e. the reciprocal of reaction time, which has a natural interpretation as speed. In other words, the Box-Cox transformation suggests that we should consider modelling the sleepstudy data as speed (reaction per unit time) instead of reaction time:

```@example Ecosystem
fit(MixedModel, @formula(1000 / reaction ~ 1 + days + (1 + days|subj)), sleepstudy)
```

(We multiply by 1000 to get the responses per _second_ instead of the responses per _millisecond_.)

!!! tip
    BoxCox.jl also works with classical linear models.

## Effects.jl

[https://beacon-biosignals.github.io/Effects.jl/v1.2/](https://beacon-biosignals.github.io/Effects.jl/v1.2/)

Effects.jl provides a convenient method to compute *effects*, i.e. predictions and associated prediction intervals computed at points on a reference grid. For models with a nonlinear link function, Effects.jl will also compute appropriate errors on the response scale based on the difference method.

For MixedModels.jl, the predictions are computed based on the fixed effects only.

The functionality of Effects.jl was inspired by the `effects` and `emmeans` packages in R and the methods within are based on @fox:effect:2003.

```@example Ecosystem
using Effects
```


```@example Ecosystem
design = Dict(:age => -15:1:20,
              :anych => [true, false])

eff_logit = effects(design, gm1; eff_col="use", level=0.95)

first(eff_logit, 10)
```

```@example Ecosystem
eff_prob = effects(design, gm1; eff_col="use", level=0.95, invlink=AutoInvLink())

first(eff_prob, 10)
```

Effects are particularly nice for visualizing the model fit and its predictions.

```@example Ecosystem
using AlgebraOfGraphics # like ggplot2, but an algebra instead of a grammar
using CairoMakie

plt1 = data(eff_logit) * mapping(:age; color=:anych) *
      (mapping(:use) * visual(Lines) +
       mapping(:lower, :upper) * visual(Band; alpha=0.3))
draw(plt1)
```

```@example Ecosystem
plt2 = data(eff_prob) * mapping(:age; color=:anych) *
      (mapping(:use) * visual(Lines) +
       mapping(:lower, :upper) * visual(Band; alpha=0.3))
draw(plt2)
```

```@example Ecosystem
using Statistics: mean
contra_by_age = transform(contra,
                          :age => ByRow(x -> round(Int, x)),
                          :use => ByRow(==("Y"));
                          renamecols=false)
contra_by_age = combine(groupby(contra_by_age, [:age, :anych]),
                        :use => mean => :use)
plt3 = plt2 +
       data(contra_by_age) *
       mapping(:age, :use;
               color=:anych => "children") * visual(Scatter)

draw(plt3;
     axis=(; title="Estimated contraceptive use by age and children",
            limits=(nothing, (0, 1)) # ylim=0,1, xlim=auto
            ))
```

Effects and estimated marginal (least squares) means are closely related and partially concepts. Effects.jl provides convenience function `emmeans` and `empairs` for computing EM means and pairwise differences of EM means.

```@example Ecosystem
emmeans(gm1)
```

```@example Ecosystem
empairs(gm1; dof=Inf)
```

!!! tip
    Effects.jl will work with any package that supports the StatsAPI.jl-based `RegressionModel` interface.

## Margins.jl

## StandardizedPredictors.jl

[https://beacon-biosignals.github.io/StandardizedPredictors.jl/v1/](https://beacon-biosignals.github.io/StandardizedPredictors.jl/v1/)

StandardizedPredictors.jl provides a convenient way to express centering, scaling, and z-standardization as a "contrast" via the pseudo-contrasts `Center`, `Scale`, `ZScore`.
Because these use the usual contrast machinery, they work well with any packages that use that machinery correctly (e.g. Effects.jl). The default behavior is to empirically compute the center and scale, but these can also be explicitly provided, either as a number or as a function (e.g. `median` to use the median for centering.)

```@example Ecosystem
using StandardizedPredictors

contrasts = Dict(:days => Center())
fit(MixedModel,
    @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy;
    contrasts)
```

!!! tip
    StandardizedPredictors.jl will work with any package that supports the StatsModels.jl-based `@formula` and contrast machinery.

## MixedModelsSmallSample.jl

[https://arnostrouwen.github.io/MixedModelsSmallSample.jl/stable/](https://arnostrouwen.github.io/MixedModelsSmallSample.jl/stable/)

MixedModelsSmallSample.jl provides the Satterthwaite and Kenward-Roger approximations for denominator degrees of freedom.

!!! tip
    There is a reason why `MixedModelsSmallSample` has "small sample" in its name:

    1. the underlying computations, especially in the Kenward-Roger case, do not scale well to large samples and may be **very** slow and memory intensive.
    2. the correction has the biggest impact on small samples -- the ``t`` distribution rapidly converges to the normal distribution as the degrees of freedom gets large.

```@example Ecosystem
using MixedModelsSmallSample

sw = small_sample_adjust(ss1, Satterthwaite())
coeftable(ss1)
DisplayAs.Text(ans) # hide
```

```@example Ecosystem
coeftable(sw)
DisplayAs.Text(ans) # hide
```

```@example Ecosystem
# Kenward-Roger degrees of freedom requires REML fit

ss1_reml = refit!(ss1; REML=true)
kr = small_sample_adjust(ss1_reml, KenwardRoger())

coeftable(ss1_reml)
DisplayAs.Text(ans) # hide
```

```@example Ecosystem
coeftable(kr)
DisplayAs.Text(ans) # hide
```

## RCall.jl and JellyMe4.jl

[https://juliainterop.github.io/RCall.jl/stable/](https://juliainterop.github.io/RCall.jl/stable/)

[https://github.com/palday/JellyMe4.jl/](https://github.com/palday/JellyMe4.jl/)

RCall.jl provides a convenient interface for interoperability with R from Julia. JellyMe4.jl extends the functionality of RCall so that MixedModels.jl-fitted models and lme4-fitted models can be translated to each other. In practical terms, this means that you can enjoy the speed of Julia for model fitting, but use all the extra packages you love from R's larger ecosystem.

## MixedModelsSerialization.jl

[https://juliamixedmodels.github.io/MixedModelsSerialization.jl/stable/api/](https://juliamixedmodels.github.io/MixedModelsSerialization.jl/stable/api/)

MixedModelsSerialization.jl provides a reduced-memory "summary" representation of a fitted model, `LinearMixedModelSummary`, that discards the model matrices and other data-sized fields while retaining the fixed- and random-effects estimates, θ, the log-likelihood, and other quantities needed to support many `StatsAPI` and `MixedModels` methods (e.g. `coef`, `vcov`, `VarCorr`, `coeftable`). This makes it practical to save and later reload models fit to very large datasets without also serializing the original data. The package is a proving ground for these ideas so that its API can evolve and have breaking releases independently of MixedModels.jl.

## MixedModelsSim.jl

[https://repsychling.github.io/MixedModelsSim.jl/stable/](https://repsychling.github.io/MixedModelsSim.jl/stable/)

MixedModelsSim.jl provides utilities for generating experimental designs, especially designs with crossed grouping factors (e.g. "Subject" and "Item") and both within- and between-unit experimental factors. Combined with `simulate`/`parametricbootstrap` from MixedModels.jl, it is commonly used for power analysis: create a design, specify hypothesized effect sizes and variance components, simulate many datasets, and examine the distribution of the resulting test statistics.
