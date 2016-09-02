# Parametric bootstrap for linear mixed-effects models

Julia is well-suited to implementing bootstrapping and other simulation-based methods for statistical models.  The `bootstrap!` function implements a parametric bootstrap for linear mixed-effects models in an efficient way, assuming that the results of interest from each simulated response vector can be incorporated into a vector of floating-point values.

## Arguments to the bootstrap! function

As indicated by the name, `bootstrap!` is a mutating function.  It modifies its first argument, which is an `k × n` matrix where `n` is the number of replications and `k` is the length of the vector of results to be saved from each replication.  The second argument is `m`, the model to bootstrap, and the third is a mutating function of a vector of length `k` and the model, `m`.  Technically, the model `m` is also modified in the function but it should be restored to its original state before `bootstrap!` returns.

```@meta
DocTestSetup = quote
    using DataFrames, MixedModels
    include(Pkg.dir("MixedModels", "test", "data.jl"))
    m = fit!(lmm(Yield ~ 1 + (1 | Batch), ds))
end
```

### An example

Consider using the parametric bootstrap to evaluate the distribution of the estimates of the variance components in the simple model
```julia
julia> using DataFrames, MixedModels

julia> include(Pkg.dir("MixedModels", "test", "data.jl"));

julia> m1 = fit!(lmm(Yield ~ 1 + (1 | Batch), ds))
Linear mixed model fit by maximum likelihood
 Formula: Yield ~ 1 + (1 | Batch)
   logLik    -2 logLik     AIC        BIC    
  -163.66353  327.32706  333.32706  337.53065

Variance components:
              Column    Variance  Std.Dev.
 Batch    (Intercept)  1388.3334 37.260346
 Residual              2451.2500 49.510100
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)    1527.5   17.6946  86.326  <1e-99


julia> function f!(v, m)
           v[1] = σ̂² = varest(m)
           v[2] = σ̂² * abs2(m.Λ[1][1])
           v
       end
f! (generic function with 1 method)

julia> f!(zeros(2), m1)
2-element Array{Float64,1}:
 2451.25
 1388.33

julia> srand(1234321);

julia> sigmas = bootstrap!(zeros(2, 100_000), m, f!)
2×100000 Array{Float64,2}:
 4547.01   2302.38   2513.48   2832.77  …  3735.86  1617.55  2624.33   1473.15
  204.834   653.688   473.595  1685.59        0.0   1324.83   287.775  1826.86

julia> sum(x -> x < 0.001, Compat.view(sigmas, 2, :))  # near zero estimates of σ₁ 
10090
```
