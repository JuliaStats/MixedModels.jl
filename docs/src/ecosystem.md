# Additional Functionality in Other Packages

```@setup Ecosystem
using DisplayAs
```

Several packages extend the functionality of MixedModels.jl, both in ways specific to mixed models and in ways applicable to more general regression models. In the following, we will use the models from the previous sections to showcase this functionality.

```@example Ecosystem
using MixedModels
progress = false
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

https://palday.github.io/MixedModelsExtras.jl/v2

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

https://github.com/kleinschmidt/RegressionFormulae.jl

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

https://palday.github.io/BoxCox.jl/v0.3/

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

https://beacon-biosignals.github.io/Effects.jl/v1.2/

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
```

```@example Ecosystem
eff_prob = effects(design, gm1; eff_col="use", level=0.95, invlink=AutoInvLink())
```

Effects are particularly nice for visualizing the model fit and its predictions.

```@example Ecosystem
using AlgebraOfGraphics # like ggplot2, but an algebra instead of a grammar
using CairoMakie

plt1 = data(eff_logit) *
      mapping(:age, :use; color=:anych) *
      (visual(Lines) + mapping(; lower=:lower, upper=:upper) * visual(LinesFill))
draw(plt1)
```

```@example Ecosystem
plt2 = data(eff_prob) *
      mapping(:age, :use; color=:anych => "children") *
      (visual(Lines) + mapping(; lower=:lower, upper=:upper) * visual(LinesFill))
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

## StandardizedPredictors.jl

https://beacon-biosignals.github.io/StandardizedPredictors.jl/v1/

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

# RCall.jl and JellyMe4.jl

https://juliainterop.github.io/RCall.jl/stable/

https://github.com/palday/JellyMe4.jl/

RCall.jl provides a convenient interface for interoperability with R from Julia. JellyMe4.jl extends the functionality of RCall so that MixedModels.jl-fitted models and lme4-fitted models can be translated to each other. In practical terms, this means that you can enjoy the speed of Julia for model fitting, but use all the extra packages you love from R's larger ecosystem.

<!--
MixedModelsSerializtion.jl
MixedModelsSim.jl
-->
