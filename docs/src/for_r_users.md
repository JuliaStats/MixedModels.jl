```@setup RUsers
using MixedModels
```

```@meta
CurrentModule = MixedModels
CollapsedDocStrings = true
```

# Finding Your Way Around for R Users

MixedModels.jl is the Julia successor to [`lme4`](https://github.com/lme4/lme4/) and is the continuation of the same research programme.
If you already know `lme4`, the transition is generally straightforward, but there are a handful of important differences to keep in mind.

## Getting started with Julia

Julia is available from [julialang.org](https://julialang.org).
The recommended installation method is [`juliaup`](https://github.com/JuliaLang/juliaup), the Julia version manager, which works similarly to `rig` for R.

### Package management and environments

Julia has a first-class package management system.
Each project can have its own environment — a `Project.toml` recording which packages and versions are required, paired with a `Manifest.toml` recording the exact resolved dependencies.
This is similar in spirit to `renv` in R, but is the standard way of working in Julia rather than an add-on.

```julia
using Pkg
Pkg.activate(".")   # use this directory as the project environment
Pkg.instantiate()   # install packages listed in Project.toml (first time only)
```

### Installing MixedModels.jl

Once Julia is running, install MixedModels.jl from the built-in package manager:

```julia
using Pkg
Pkg.add("MixedModels")
```

Alternatively, type `]` at the Julia REPL prompt to enter package-manager mode, then:

```
pkg> add MixedModels
```

### Loading packages

Julia does not expose a large set of functions on start-up.
Operations that R users treat as "basic" -- reading CSV files, manipulating data frames, or fitting ordinary linear models -- all require loading separate packages.
This also means that even relatively simple analyses will typically start with several `using` statements.

| Task                      | R                      | Julia                          |
|:---------------------- ---|:-----------------------|:-------------------------------|
| Read CSV                  | `read.csv()` (base)    | `CSV.jl`                       |
| Data frames               | `data.frame` (base)    | `DataFrames.jl`                |
| Factors                   | `factor()` (base)      | `CategoricalArrays.jl`         |
| Linear model              | `lm()` (stats)         | `GLM.jl`                       |
| Mixed model               | `lme4`                 | `MixedModels.jl`               |
| Effects / EMMs            | `effects`, `emmeans`   | `Effects.jl`                   |
| Standardize predictors    | `scale()` (base)       | `StandardizedPredictors.jl`    |
| R² / ICC                  | `MuMIn`, `performance` | `MixedModelsExtras.jl`         |
| Box-Cox                   | `MASS::boxcox`         | `BoxCox.jl`                    |
| Grammar-of-graphics plots | `ggplot2`              | `AlgebraOfGraphics.jl` + Makie |
| Mixed-model plots         | `sjPlot`               | `MixedModelsMakie.jl`          |


### JIT compilation

The first call to `fit` (or `lmm` or any complicated Julia function) may take a surprisingly long time.
This is due to Just-In-Time (JIT) compilation: Julia compiles each method specialization the first time it is called.
Subsequent calls are much faster.

## Data loading and manipulation

### Reading data

```julia
using CSV, DataFrames

df = CSV.read("mydata.csv", DataFrame)
```

R data files (`.rds`, `.rda`) can be read with [RData.jl](https://github.com/JuliaData/RData.jl).
Many classic datasets distributed with R packages are accessible via [RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl).
The datasets used in MixedModels.jl examples are available through [MixedModelsDatasets.jl](https://github.com/JuliaStats/MixedModelsDatasets.jl):

```julia
using MixedModelsDatasets: dataset
sleepstudy = dataset("sleepstudy")
```

### Data frame manipulation

[DataFrames.jl](https://dataframes.juliadata.org/stable/) is the standard package for tabular data.
Its transform/combine/select mini-language is powerful but has a learning curve for users coming from `dplyr`.
Several add-ons ease the transition:

- [Tidier.jl](https://github.com/TidierOrg/) — close to the tidyverse idiom
- [DataFrameMacros.jl](https://github.com/jkrumbiegel/DataFrameMacros.jl) — macro-based helpers
- [Chain.jl](https://github.com/jkrumbiegel/Chain.jl) — pipe-based composition (like `|>` chains or `magrittr`'s `%>%`)


## Formula syntax (Wilkinson-Roger Notation)

MixedModels.jl uses a formula syntax based on Wilkinson-Rogers notation, the same foundation as R's formula interface.
There are a few important differences.

### The `@formula` macro

In Julia, formulas must be written inside the `@formula` macro.
Formulas in Julia are implemented as a domain-specific language processed at parse time by a macro, rather than as a built-in language feature.

```r
# R
lmer(reaction ~ 1 + days + (1 + days | subj), data = sleepstudy)
```

```julia
# Julia
lmm(@formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy)
```

The majority of syntax is identical, but there are a few key differences:

- **Interaction operator:** R uses `:` for interactions; Julia uses `&`. The `*` operator (main effects plus interaction) works identically in both languages.
- **Interactions up to a certain order:** The `^` notation requires [RegressionFormulae.jl](https://github.com/kleinschmidt/RegressionFormulae.jl), which is loaded automatically by MixedModels.jl.
- **Suppressing the intercept:** the use of `-1` to suppress the intercept term is not supported in Julia. Use `0` instead.
- **Zero-correlation random effects::** lme4 uses the `||` operator to fit random effects without estimating the correlation parameters; MixedModels.jl uses the `zerocorr` function inside the formula:

```r
# R (lme4)
(1 + days || subj)
```

```julia
# Julia
zerocorr(1 + days | subj)   # inside @formula
```

## Fitting models

The convenience functions `lmm` (linear) and `glmm` (generalized linear) mirror `lmer` and `glmer` from lme4.
The more general `fit(MixedModel, ...)` form works for both and is the canonical interface.

```r
# R (lme4)
m <- lmer(reaction ~ 1 + days + (1 + days | subj), data = sleepstudy)
```

```julia
# Julia
m = lmm(@formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy)
# equivalent to
m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy)
```

```r
# R (lme4)
gm <- glmer(use ~ 1 + age + (1 | district), data = contra, family = binomial)
```

```julia
# Julia
gm = glmm(@formula(use ~ 1 + age + (1 | district)), contra, Bernoulli())
# equivalent to
gm = fit(MixedModel, @formula(use ~ 1 + age + (1 | district)), contra, Bernoulli())
```

### Factors and contrast coding

Julia's equivalent of R's `factor` is provided by [CategoricalArrays.jl](https://categoricalarrays.juliadata.org/stable/).
MixedModels.jl accepts both `CategoricalArray` columns and plain `String`/`Symbol` columns, applying default treatment coding when no contrasts are specified.

```julia
using CategoricalArrays
df.group = categorical(df.group)
```

Contrast schemes are set per-variable via a `Dict` passed to the `contrasts` keyword argument of the fitting function.
This is analogous to setting `contrasts(df$a) <- contr.helmert(...)` in R, but is scoped to the model call rather than mutating the data.

```julia
fit(MixedModel,
    @formula(y ~ 1 + a + (1 | g)), df;
    contrasts = Dict(:a => HelmertCoding()))
```

The available coding schemes include `DummyCoding`, `EffectsCoding`, `HelmertCoding`, `SeqDiffCoding` (successive differences), `HypothesisCoding` (custom hypotheses), and more.
See the [StatsModels.jl contrasts documentation](https://juliastats.org/StatsModels.jl/stable/contrasts/) for details.


### REML vs ML

lme4 uses REML by default; MixedModels.jl uses ML by default.
To fit with REML, pass the keyword argument:

```julia
lmm(@formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy; REML=true)
```

## Extracting model information

Many extractor functions share the same name in lme4 and MixedModels.jl.
Notable exceptions are documented in the table below.

| lme4              | MixedModels.jl     | Notes                                   |
|:------------------|:-------------------|:----------------------------------------|
| `fixef(m)`        | `fixef(m)`         | fixed-effects coefficients (full rank)  |
| `ranef(m)`        | `ranef(m)`         | conditional modes of the random effects |
| `VarCorr(m)`      | `VarCorr(m)`       | variance–covariance components          |
| `sigma(m)`        | `dispersion(m)`    | residual standard deviation             |
| `fitted(m)`       | `fitted(m)`        | fitted values                           |
| `residuals(m)`    | `residuals(m)`     | residuals                               |
| `predict(m)`      | `predict(m)`       | predictions                             |
| `logLik(m)`       | `loglikelihood(m)` | log-likelihood                          |
| `AIC(m)`          | `aic(m)`           | Akaike information criterion            |
| `BIC(m)`          | `bic(m)`           | Bayesian information criterion          |
| `nobs(m)`         | `nobs(m)`          | number of observations                  |
| `coefnames(m)`    | `fixefnames(m)`    | names of the fixed-effects coefficients |
| `summary(m)`      | `m` / `show(m)`    | print a model summary                   |

### $p$-values

Unlike lme4, MixedModels.jl computes $p$-values for fixed effects by default by treating the $t$-value as $z$-values and thus avoiding challenges in defining [degrees of freedom](https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#why-doesnt-lme4-display-denominator-degrees-of-freedomp-values-what-other-options-do-i-have).
As the degrees of freedom grows, the $t$ distribution converges to the $z$ distribution, so treating $t$ as $z$ is the same as treating the denominator degrees of freedom as "arbitrarily large".

In addition to the Wald approximation, MixedModels.jl also supports the [parametric bootstrap](bootstrap.md) and [profiling](@ref `profile`)

Likelihood ratio tests are available via `likelihoodratiotest` and `lrtest`.
`lrtest` is the standard name used throughout the
JuliaStats ecosystem and uses the same output, while `likelihoodratiotest` is specific to MixedModels.jl uses a more specialized output.
The computed values are the same.

For Satterthwaite or Kenward-Roger degrees of freedom (the approach taken by `lmerTest` in R), see [MixedModelsSmallSample.jl](https://arnostrouwen.github.io/MixedModelsSmallSample.jl/dev/). Note this is a separate community package, not maintained by the MixedModels.jl team.
