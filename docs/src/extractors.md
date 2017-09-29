# Extractor functions

`LinearMixedModel` and `GeneralizedLinearMixedModel` are subtypes of `StatsBase.RegressionModel`
Many of the generic extractors defined in the `StatsBase` package have methods for these models.
```@docs
StatsBase.coef
StatsBase.coeftable
StatsBase.dof
StatsBase.deviance
StatsBase.fitted
StatsBase.loglikelihood
StatsBase.stderr
StatsBase.vcov
```

Other extractors are defined in the `MixedModels` package itself.
```@docs
fixef
fnames
getΛ
getθ
lowerbd
objective
pwrss
ranef
sdest
varest
```

Applied to one of the models previously fit these yield
````julia
julia> using DataFrames, RData, MixedModels

julia> const dat = convert(Dict{Symbol,DataFrame}, load(Pkg.dir("MixedModels", "test", "dat.rda")));

julia> fm1 = fit!(lmm(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -163.66353  327.32706  333.32706  337.53065

Variance components:
              Column    Variance  Std.Dev. 
 G        (Intercept)  1388.3333 37.260345
 Residual              2451.2500 49.510100
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)    1527.5   17.6946  86.326  <1e-99


julia> fixef(fm1)
1-element Array{Float64,1}:
 1527.5

julia> coef(fm1)
1-element Array{Float64,1}:
 1527.5

julia> coeftable(fm1)
             Estimate Std.Error z value P(>|z|)
(Intercept)    1527.5   17.6946  86.326  <1e-99


julia> getΛ(fm1)
1-element Array{Float64,1}:
 0.752581

julia> getθ(fm1)
1-element Array{Float64,1}:
 0.752581

julia> loglikelihood(fm1)
-163.6635299405672

julia> pwrss(fm1)
73537.50049200655

julia> showall(ranef(fm1))
Array{Float64,2}[[-16.6282 0.369516 26.9747 -21.8014 53.5798 -42.4943]]
julia> showall(ranef(fm1, uscale=true))
Array{Float64,2}[[-22.0949 0.490999 35.8429 -28.9689 71.1948 -56.4648]]
julia> sdest(fm1)
49.51010014532609

julia> std(fm1)
2-element Array{Array{Float64,1},1}:
 [37.2603]
 [49.5101]

julia> stderr(fm1)
1-element Array{Float64,1}:
 17.6946

julia> varest(fm1)
2451.2500164002186

julia> vcov(fm1)
1×1 Array{Float64,2}:
 313.097

````


