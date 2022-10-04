# API

In addition to its own functionality, MixedModels.jl also implements extensive support for the [`StatsAPI.StatisticalModel`](https://github.com/JuliaStats/StatsAPI.jl/blob/main/src/statisticalmodel.jl) and [`StatsAPI.RegressionModel`](https://github.com/JuliaStats/StatsAPI.jl/blob/main/src/regressionmodel.jl) API.

## Types

```@autodocs
Modules = [MixedModels]
Order   = [:type]
```

## Exported Functions
```@autodocs
Modules = [MixedModels]
Private = false
Order   = [:function]
```

## Methods from `StatsAPI.jl`, `StatsBase.jl`, `StatsModels.jl` and `GLM.jl`

```julia
aic
aicc
bic
coef
coefnames
coeftable
deviance
dispersion
dispersion_parameter
dof
dof_residual
fit
fit!
fitted
formula
isfitted
islinear
leverage
loglikelihood
meanresponse
modelmatrix
model_response
nobs
predict
residuals
response
responsename
StatsModels.lrtest # not exported
std
stderror
vcov
weights
```

### MixedModels.jl "alternatives" and extensions to StatsAPI and GLM functions

The following are MixedModels.jl-specific functions and not simply methods for functions defined in StatsAPI and GLM.jl.

```julia
coefpvalues
condVar
condVarTables
fitted!
fixef
fixefnames
likelihoodratiotest # not exported
pwrss
ranef
raneftables
refit!
shortestcovint
sdest
simulate
simulate!
stderrror!
varest
```

## Non-Exported Functions

Note that unless discussed elsewhere in the online documentation, non-exported functions should be considered implementation details.

```@autodocs
Modules = [MixedModels]
Public  = false
Order   = [:function]
Filter = f -> !startswith(string(f), "_")
```
