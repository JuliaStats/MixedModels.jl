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

## Non-Exported Functions

Note that unless discussed elsewhere in the online documentation, non-exported functions should be considered implementation details.

```@autodocs
Modules = [MixedModels]
Public  = false
Order   = [:function]
Filter = f -> !startswith(string(f), "_")
```
