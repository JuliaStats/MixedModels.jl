# MixedModels.jl Documentation

```@meta
CurrentModule = MixedModels
```

*MixedModels.jl* is a Julia package providing capabilities for fitting and examining linear and generalized linear mixed-effect models.
It is similar in scope to the [*lme4*](https://github.com/lme4/lme4) package for `R`.

# TLDR

```@setup Main
using DisplayAs
```
You can fit a model using a `lmer`-style model formula using `@formula` and a dataset.
Here is a short example of how to fit a linear mixed-effects modeling using the `dyestuff` dataset:

```@example Main
using DataFrames, MixedModels           # load packages
dyestuff = MixedModels.dataset(:dyestuff);              # load dataset

lmod = lmm(@formula(yield ~ 1 + (1|batch)), dyestuff)   # fit the model!
DisplayAs.Text(ans) # hide
```
For a generalized linear mixed-effect model, you have to specify a distribution for the response variable (and optionally a link function).
A quick example of generalized linear model using the `verbagg` dataset:

```@example Main
using DataFrames, MixedModels               # load packages
verbagg = MixedModels.dataset(:verbagg);    # load dataset

frm = @formula(r2 ~ 1 + anger + gender + btype + situ + mode + (1|subj) + (1|item));
bernmod = glmm(frm, verbagg, Bernoulli())   # fit the model!
DisplayAs.Text(ans) # hide
```

```@contents
Pages = [
        "constructors.md",
        "optimization.md",
        "GaussHermite.md",
        "bootstrap.md",
        "rankdeficiency.md",
        "mime.md",
]
Depth = 2
```
