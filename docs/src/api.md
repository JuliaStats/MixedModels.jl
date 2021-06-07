# API

In addition to its own functionality, MixedModels.jl also implements extensive support for the `StatsBase.StatisticalModel` and `StatsBase.RegressionModel` API.

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
