# Parametric bootstrap for linear mixed-effects models

Julia is well-suited to implementing bootstrapping and other simulation-based methods for statistical models.
The `bootstrap!` function in the [MixedModels package](https://github.com/dmbates/MixedModels.jl) provides
an efficient parametric bootstrap for linear mixed-effects models, assuming that the results of interest
from each simulated response vector can be incorporated into a vector of floating-point values.

## The parametric bootstrap

[Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) is a family of procedures
for generating sample values of a statistic, allowing for visualization of the distribution of the
statistic or for inference from this sample of values.

A _parametric bootstrap_ is used with a parametric model, `m`, that has been fitted to data.
The procedure is to simulate `n` response vectors from `m` using the estimated parameter values
and refit `m` to these responses in turn, accumulating the statistics of interest at each iteration.

The parameters of a linear mixed-effects model as fit by the `lmm` function are the fixed-effects
parameters, `β`, the standard deviation, `σ`, of the per-observation noise, and the covariance
parameter, `θ`, that defines the variance-covariance matrices of the random effects.

For example, a simple linear mixed-effects model for the `Dyestuff` data in the [`lme4`](http://github.com/lme4/lme4)
package for [`R`](https://www.r-project.org) is fit by
````julia
julia> using DataFrames, MixedModels, RData, Gadfly
Error: ArgumentError: Package Gadfly not found in current path:
- Run `import Pkg; Pkg.add("Gadfly")` to install the Gadfly package.


````




````julia
julia> ds = names!(dat[:Dyestuff], [:Batch, :Yield])
30×2 DataFrame
│ Row │ Batch        │ Yield   │
│     │ Categorical… │ Float64 │
├─────┼──────────────┼─────────┤
│ 1   │ A            │ 1545.0  │
│ 2   │ A            │ 1440.0  │
│ 3   │ A            │ 1440.0  │
│ 4   │ A            │ 1520.0  │
│ 5   │ A            │ 1580.0  │
│ 6   │ B            │ 1540.0  │
│ 7   │ B            │ 1555.0  │
⋮
│ 23  │ E            │ 1515.0  │
│ 24  │ E            │ 1635.0  │
│ 25  │ E            │ 1625.0  │
│ 26  │ F            │ 1520.0  │
│ 27  │ F            │ 1455.0  │
│ 28  │ F            │ 1450.0  │
│ 29  │ F            │ 1480.0  │
│ 30  │ F            │ 1445.0  │

julia> m1 = fit(LinearMixedModel, @formula(Yield ~ 1 + (1 | Batch)), ds)
Error: MethodError: Cannot `convert` an object of type Tuple{Array{Float64,2},ReMat{Float64,1}} to an object of type Array{Float64,2}
Closest candidates are:
  convert(::Type{Array{S,N}}, !Matched::PooledArrays.PooledArray{T,R,N,RA} where RA) where {S, T, R, N} at /home/bates/.julia/packages/PooledArrays/ufJSl/src/PooledArrays.jl:288
  convert(::Type{Array{T,N}}, !Matched::StaticArrays.SizedArray{S,T,N,M} where M) where {T, S, N} at /home/bates/.julia/packages/StaticArrays/3KEjZ/src/SizedArray.jl:62
  convert(::Type{T<:Array}, !Matched::AbstractArray) where T<:Array at array.jl:474
  ...

````





Now bootstrap the model parameters
````julia
julia> results = bootstrap(100_000, m1);
Error: UndefVarError: bootstrap not defined

julia> show(names(results))
Error: UndefVarError: results not defined

````




The results for each bootstrap replication are stored in the columns of the matrix passed in as the first
argument.  A density plot of the bootstrapped values of `σ` is created as
````julia

plot(results, x = :σ, Geom.density, Guide.xlabel("Parametric bootstrap estimates of σ"))
````


````
Error: UndefVarError: Guide not defined
````



````
Error: UndefVarError: Guide not defined
````



````
Error: UndefVarError: Guide not defined
````





The distribution of the bootstrap samples of `σ` is a bit skewed but not terribly so.  However, the
distribution of the bootstrap samples of the estimate of `σ₁` is highly skewed and has a spike at
zero.
