# Gradient and Hessian via ForwardDiff.jl

Experimental support for computing the gradient and the Hessian of the objective function (i.e. negative twice the profiled log likelihood) via ForwardDiff.jl are provided as a package extension.

The core functionality is provided by defining appropriate methods for `ForwardDiff.gradient` and `ForwardDiff.hessian`:

```@docs
ForwardDiff.gradient(::LinearMixedModel{T}, ::Vector{T}) where {T}
ForwardDiff.hessian(::LinearMixedModel{T}, ::Vector{T}) where {T}
```

## Exact zero at optimum for trivial models

```@example ForwardDiff
using MixedModels, ForwardDiff
using DisplayAs # hide
fm1 = lmm(@formula(yield ~ 1 + (1|batch)), MixedModels.dataset(:dyestuff2))
DisplayAs.Text(ans) # hide
```

```@example ForwardDiff
ForwardDiff.gradient(fm1)
```

```@example ForwardDiff
ForwardDiff.hessian(fm1)
```

## Approximate zero at optimum for non trivial models

```@example ForwardDiff
fm2 = lmm(@formula(reaction ~ 1 + days + (1+days|subj)), MixedModels.dataset(:sleepstudy))
DisplayAs.Text(ans) # hide
```

```@example ForwardDiff
ForwardDiff.gradient(fm2)
```

```@example ForwardDiff
ForwardDiff.hessian(fm2)
```
