# Model constructors

The `lmm` function creates a linear mixed-effects model representation from a `Formula` and an appropriate `data` type.
At present a `DataFrame` is required but that is expected to change.
```@docs
lmm
```

## Examples of linear mixed-effects model fits

For illustration, several data sets from the *lme4* package for *R* are made available in `.rda` format in this package.
These include the `Dyestuff` and `Dyestuff2` data sets.
````julia
julia> using DataFrames, RData, MixedModels

julia> const dat = convert(Dict{Symbol,DataFrame}, load(Pkg.dir("MixedModels", "test", "dat.rda")));

julia> dump(dat[:Dyestuff])
DataFrames.DataFrame  30 observations of 2 variables
  G: DataArrays.PooledDataArray{String,UInt8,1}(30) Union{Nulls.Null, String}["A", "A", "A", "A"]
  Y: DataArrays.DataArray{Float64,1}(30) Union{Float64, Nulls.Null}[1545.0, 1440.0, 1440.0, 1520.0]


````




The columns in these data sets have been renamed for convenience.
The response is always named `Y`.
Potential grouping factors for random-effects terms are named `G`, `H`, etc.

### Models with simple, scalar random effects

The formula language in *Julia* is similar to that in *R* except that the formula must be enclosed in a call to the `@formula` macro.
A basic model with simple, scalar random effects for the levels of `G` (the batch of an intermediate product, in this case) is declared and fit as
````julia
julia> fm1 = fit!(lmm(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -163.66353  327.32706  333.32706  337.53065

Variance components:
              Column    Variance  Std.Dev. 
 G        (Intercept)  1388.3332 37.260344
 Residual              2451.2500 49.510100
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)    1527.5   17.6946  86.326  <1e-99


````





(If you are new to Julia you may find that this first fit takes an unexpectedly long time, due to Just-In-Time (JIT) compilation of the code.
The second and subsequent calls to such functions are much faster.)

````julia
julia> @time fit!(lmm(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff2]))
  0.011636 seconds (5.84 k allocations: 756.859 KiB)
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -81.436518 162.873037 168.873037 173.076629

Variance components:
              Column    Variance  Std.Dev. 
 G        (Intercept)   0.000000 0.0000000
 Residual              13.346099 3.6532314
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)    5.6656  0.666986 8.49433  <1e-16


````





### Simple, scalar random effects

A simple, scalar random effects term in a mixed-effects model formula is of the form `(1|G)`.
All random effects terms end with `|G` where `G` is the *grouping factor* for the random effect.
The name or, more generally, the expression `G` should evaluate to a categorical array that has a distinct set of *levels*.
The random effects are associated with the levels of the grouping factor.

A *scalar* random effect is, as the name implies, one scalar value for each level of the grouping factor.
A *simple, scalar* random effects term is of the form, `(1|G)`.
It corresponds to a shift in the intercept for each level of the grouping factor.

### Models with vector-valued random effects

The *sleepstudy* data are observations of reaction time, `Y`, on several subjects, `G`, after 0 to 9 days of sleep deprivation, `U`.
A model with random intercepts and random slopes for each subject, allowing for within-subject correlation of the slope and intercept, is fit as
````julia
julia> fm2 = fit!(lmm(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + U + ((1 + U) | G)
   logLik   -2 logLik     AIC        BIC    
 -875.96967 1751.93934 1763.93934 1783.09709

Variance components:
              Column    Variance  Std.Dev.   Corr.
 G        (Intercept)  565.51067 23.780468
          U             32.68212  5.716828  0.08
 Residual              654.94145 25.591824
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   251.405   6.63226 37.9064  <1e-99
U             10.4673   1.50224 6.96781  <1e-11


````





A model with uncorrelated random effects for the intercept and slope by subject is fit as
````julia
julia> fm3 = fit!(lmm(@formula(Y ~ 1 + U + (1|G) + (0+U|G)), dat[:sleepstudy]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + U + (1 | G) + ((0 + U) | G)
   logLik   -2 logLik     AIC        BIC    
 -876.00163 1752.00326 1762.00326 1777.96804

Variance components:
              Column    Variance  Std.Dev.   Corr.
 G        (Intercept)  584.258974 24.17145
          U             33.632805  5.79938  0.00
 Residual              653.115782 25.55613
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   251.405   6.70771   37.48  <1e-99
U             10.4673   1.51931 6.88951  <1e-11


````





Although technically there are two random-effects *terms* in the formula for *fm3* both have the same grouping factor
and, internally, are amalgamated into a single vector-valued term.

### Models with multiple, scalar random-effects terms

A model for the *Penicillin* data incorporates random effects for the plate, `G`, and for the sample, `H`.
As every sample is used on every plate these two factors are *crossed*.
````julia
julia> fm4 = fit!(lmm(@formula(Y ~ 1 + (1|G) + (1|H)), dat[:Penicillin]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G) + (1 | H)
   logLik   -2 logLik     AIC        BIC    
 -166.09417  332.18835  340.18835  352.06760

Variance components:
              Column    Variance  Std.Dev. 
 G        (Intercept)  0.7149795 0.8455646
 H        (Intercept)  3.1351924 1.7706474
 Residual              0.3024264 0.5499331
 Number of obs: 144; levels of grouping factors: 24, 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   22.9722  0.744596 30.8519  <1e-99


````





In contrast the sample, `G`, grouping factor is *nested* within the batch, `H`, grouping factor in the *Pastes* data.
That is, each level of `G` occurs in conjunction with only one level of `H`.
````julia
julia> fm5 = fit!(lmm(@formula(Y ~ 1 + (1|G) + (1|H)), dat[:Pastes]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G) + (1 | H)
   logLik   -2 logLik     AIC        BIC    
 -123.99723  247.99447  255.99447  264.37184

Variance components:
              Column    Variance  Std.Dev.  
 G        (Intercept)  8.4336167 2.90406898
 H        (Intercept)  1.1991793 1.09507045
 Residual              0.6780021 0.82340884
 Number of obs: 60; levels of grouping factors: 30, 10

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   60.0533  0.642136 93.5212  <1e-99


````





In observational studies it is common to encounter *partially crossed* grouping factors.
For example, the *InstEval* data are course evaluations by students, `G`, of instructors, `H`.
Additional covariates include the academic department, `I`, in which the course was given and `A`, whether or not it was a service course.
````julia
julia> fm6 = fit!(lmm(@formula(Y ~ 1 + A * I + (1|G) + (1|H)), dat[:InstEval]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + A * I + (1 | G) + (1 | H)
     logLik        -2 logLik          AIC             BIC       
 -1.18792777×10⁵  2.37585553×10⁵  2.37647553×10⁵  2.37932876×10⁵

Variance components:
              Column     Variance   Std.Dev.  
 G        (Intercept)  0.105417913 0.32468125
 H        (Intercept)  0.258416315 0.50834665
 Residual              1.384727796 1.17674458
 Number of obs: 73421; levels of grouping factors: 2972, 1128

  Fixed-effects parameters:
                Estimate Std.Error  z value P(>|z|)
(Intercept)      3.22961  0.064053  50.4209  <1e-99
A: 1            0.252025 0.0686507  3.67112  0.0002
I: 5            0.129536  0.101294  1.27882  0.2010
I: 10          -0.176751 0.0881352 -2.00545  0.0449
I: 12          0.0517102 0.0817523 0.632523  0.5270
I: 6           0.0347319  0.085621 0.405647  0.6850
I: 7             0.14594 0.0997984  1.46235  0.1436
I: 4            0.151689 0.0816897  1.85689  0.0633
I: 8            0.104206  0.118751 0.877517  0.3802
I: 9           0.0440401 0.0962985  0.45733  0.6474
I: 14          0.0517546 0.0986029 0.524879  0.5997
I: 1           0.0466719  0.101942 0.457828  0.6471
I: 3           0.0563461 0.0977925  0.57618  0.5645
I: 11          0.0596536  0.100233 0.595151  0.5517
I: 2          0.00556284  0.110867 0.050176  0.9600
A: 1 & I: 5    -0.180757  0.123179 -1.46744  0.1423
A: 1 & I: 10   0.0186492  0.110017 0.169512  0.8654
A: 1 & I: 12   -0.282269 0.0792937  -3.5598  0.0004
A: 1 & I: 6    -0.494464 0.0790278 -6.25684   <1e-9
A: 1 & I: 7    -0.392054  0.110313 -3.55403  0.0004
A: 1 & I: 4    -0.278547 0.0823727 -3.38154  0.0007
A: 1 & I: 8    -0.189526  0.111449 -1.70056  0.0890
A: 1 & I: 9    -0.499868 0.0885423 -5.64553   <1e-7
A: 1 & I: 14   -0.497162 0.0917162 -5.42065   <1e-7
A: 1 & I: 1     -0.24042 0.0982071  -2.4481  0.0144
A: 1 & I: 3    -0.223013 0.0890548 -2.50422  0.0123
A: 1 & I: 11   -0.516997 0.0809077 -6.38997   <1e-9
A: 1 & I: 2    -0.384773  0.091843 -4.18946   <1e-4


````





## Fitting generalized linear mixed models

To create a GLMM using
```@docs
glmm
```
the distribution family for the response, and possibly the link function, must be specified.

````julia
julia> gm1 = fit!(glmm(@formula(r2 ~ 1 + a + g + b + s + m + (1|id) + (1|item)), dat[:VerbAgg],
    Bernoulli()))
Generalized Linear Mixed Model fit by minimizing the Laplace approximation to the deviance
  Formula: r2 ~ 1 + a + g + b + s + m + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance (Laplace approximation): 8135.8329

Variance components:
          Column    Variance   Std.Dev.  
 id   (Intercept)  1.79357917 1.33924575
 item (Intercept)  0.11713603 0.34225142

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
              Estimate Std.Error  z value P(>|z|)
(Intercept)   0.553446  0.385367  1.43615  0.1510
a            0.0574199 0.0167532  3.42741  0.0006
g: M          0.320748  0.191212  1.67745  0.0935
b: scold       -1.0598   0.18415 -5.75508   <1e-8
b: shout      -2.10382  0.186509   -11.28  <1e-28
s: self       -1.05437  0.151187 -6.97393  <1e-11
m: do        -0.706998     0.151  -4.6821   <1e-5


````





The canonical link, which is `GLM.LogitLink` for the `Bernoulli` distribution, is used if no explicit link is specified.

In the [`GLM` package](https://github.com/JuliaStats/GLM.jl) the appropriate distribution for a 0/1 response is the `Bernoulli` distribution.
The `Binomial` distribution is only used when the response is the fraction of trials returning a positive, in which case the number of trials must be specified as the case weights.

# Extractor functions

`LinearMixedModel` and `GeneralizedLinearMixedModel` are subtypes of `StatsBase.RegressionModel` which, in turn, is a subtype of `StatsBase.StatisticalModel`.
Many of the generic extractors defined in the `StatsBase` package have methods for these models.

## Model-fit statistics

The statistics describing the quality of the model fit include
```@docs
loglikelihood(::StatisticalModel)
aic
bic
dof(::StatisticalModel)
nobs(::StatisticalModel)
```
````julia
julia> loglikelihood(fm1)
-163.6635299405682

julia> aic(fm1)
333.3270598811364

julia> bic(fm1)
337.5306520261229

julia> loglikelihood(gm1)
-4067.916428181088

````





The `deviance` generic is documented as returning negative twice the log-likelihood adjusting for the saturated model.
```@docs
deviance(::StatisticalModel)
```

Because it is not clear what the saturated model corresponding to a particular `LinearMixedModel` should be, negative twice the log-likelihood is called the `objective`.
```@docs
objective
```
This value is also accessible as the `deviance` but the user should bear in mind that this doesn't have all the properties of a deviance which is corrected for the saturated model.
For example, it is not necessarily non-negative.
````julia
julia> objective(fm1)
327.3270598811364

julia> deviance(fm1)
327.3270598811364

````





The value optimized when fitting a `GeneralizedLinearMixedModel` is the Laplace approximation to the deviance.
```@docs
LaplaceDeviance
```
````julia
julia> LaplaceDeviance(gm1)
8135.832856362156

````





## Fixed-effects parameter estimates

The `coef` and `fixef` extractors both return the maximum likelihood estimates of the fixed-effects coefficients.
```@docs
coef
fixef
```
````julia
julia> show(coef(fm1))
[1527.5]
julia> show(fixef(fm1))
[1527.5]
julia> show(fixef(gm1))
[0.553446, 0.0574199, 0.320748, -1.0598, -2.10382, -1.05437, -0.706998]
````





The variance-covariance matrix of the fixed-effects coefficients is returned by
```@docs
vcov
```
````julia
julia> vcov(fm2)
2×2 Array{Float64,2}:
 43.9868   -1.37039
 -1.37039   2.25671

julia> vcov(gm1)
7×7 Array{Float64,2}:
  0.148508    -0.0056049    -0.00977128   -0.0169693    -0.0171417    -0.0114539    -0.0114551  
 -0.0056049    0.000280668   7.19153e-5   -1.43716e-5   -2.90569e-5   -1.47973e-5   -1.02416e-5 
 -0.00977128   7.19153e-5    0.0365618    -9.25614e-5   -0.000162389  -8.04419e-5   -5.25875e-5 
 -0.0169693   -1.43716e-5   -9.25614e-5    0.0339111     0.0171821     0.000265802   0.000172098
 -0.0171417   -2.90569e-5   -0.000162389   0.0171821     0.0347854     0.000658966   0.000520523
 -0.0114539   -1.47973e-5   -8.04419e-5    0.000265802   0.000658966   0.0228575     0.000247782
 -0.0114551   -1.02416e-5   -5.25875e-5    0.000172098   0.000520523   0.000247782   0.022801   

````





The standard errors are the square roots of the diagonal elements of the estimated variance-covariance matrix of the coefficients.
```@docs
stderr
```
````julia
julia> stderr(fm2)
2-element Array{Float64,1}:
 6.63226
 1.50224

julia> stderr(gm1)
7-element Array{Float64,1}:
 0.385367 
 0.0167532
 0.191212 
 0.18415  
 0.186509 
 0.151187 
 0.151    

````





Finally, the `coeftable` generic produces a table of coefficient estimates, their standard errors, and their ratio.
The *p-values* quoted here should be regarded as approximations.
```@docs
coeftable
```
````julia
julia> coeftable(fm2)
             Estimate Std.Error z value P(>|z|)
(Intercept)   251.405   6.63226 37.9064  <1e-99
U             10.4673   1.50224 6.96781  <1e-11


````





## Covariance parameter estimates

The covariance parameters estimates, in the form shown in the model summary, are a `VarCorr` object
```@docs
VarCorr
```
````julia
julia> VarCorr(fm2)
Variance components:
              Column    Variance  Std.Dev.   Corr.
 G        (Intercept)  565.51067 23.780468
          U             32.68212  5.716828  0.08
 Residual              654.94145 25.591824


julia> VarCorr(gm1)
Variance components:
          Column    Variance   Std.Dev.  
 id   (Intercept)  1.79357917 1.33924575
 item (Intercept)  0.11713603 0.34225142


````





Individual components are returned by other extractors
```@docs
varest
sdest
```
````julia
julia> varest(fm2)
654.9414513584776

julia> sdest(fm2)
25.591823916213507

````





## Conditional modes of the random effects

The `ranef` extractor
```@docs
ranef
```
````julia
julia> ranef(fm1)
1-element Array{Array{Float64,2},1}:
 [-16.6282 0.369516 26.9747 -21.8014 53.5798 -42.4943]

julia> ranef(fm1, named=true)[1]
1×6 Named Array{Float64,2}
      A ╲ B │        A         B         C         D         E         F
────────────┼───────────────────────────────────────────────────────────
(Intercept) │ -16.6282  0.369516   26.9747  -21.8014   53.5798  -42.4943

````




returns the *conditional modes* of the random effects given the observed data.
That is, these are the values that maximize the conditional density of the random effects given the observed data.
For a `LinearMixedModel` these are also the conditional mean values.

These are sometimes called the *best linear unbiased predictors* or [`BLUPs`](https://en.wikipedia.org/wiki/Best_linear_unbiased_prediction) but that name is not particularly meaningful.

At a superficial level these can be considered as the "estimates" of the random effects, with a bit of hand waving, but pursuing this analogy too far usually results in confusion.

The corresponding conditional variances are returned by
```@docs
condVar
```
````julia
julia> condVar(fm1)
1-element Array{Array{Float64,3},1}:
 [362.31]

[362.31]

[362.31]

[362.31]

[362.31]

[362.31]

````





# Optimization of the objective

To determine the maximum likelihood estimates (mle's) of the parameters in a `LinearMixedModel` the `objective`, negative twice the log-likelihood, is minimized.  
This objective is on the scale of the [*deviance*](https://en.wikipedia.org/wiki/Deviance_(statistics)).

 would involve a nonlinear optimization over all the parameters but the optimization can be simplified by evaluating the profiled log-likelihood.  

In practice it is more common to minimize negative twice the log-likelihood which is the `objective` for a `LinearMixedModel`.

By definition the objective is a function of all the parameters in the model.
However, it is possible to evaluate a *profiled log-likelihood*, which is a function of only the parameters θ that determine the *relative covariance factor*.
That is, given a value of θ, it is possible through a direct (i.e. non-iterative) calculation to determine the estimates of β, the fixed-effects coefficients, and σ, the standard deviation of the per-observation noise term.

# Internal representation

A `LinearMixedModel` is composed of a vector of terms and some blocked arrays associated with them.
```@docs
LinearMixedModel
```

Other extractors are defined in the `MixedModels` package itself.
```@docs
fnames
getΛ
getθ
lowerbd
```

Applied to one of the models previously fit these yield
````julia
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
-163.6635299405682

julia> pwrss(fm1)
73537.50101605429

julia> showall(ranef(fm1))
Array{Float64,2}[[-16.6282 0.369516 26.9747 -21.8014 53.5798 -42.4943]]
julia> showall(ranef(fm1, uscale=true))
Array{Float64,2}[[-22.0949 0.490999 35.8429 -28.9689 71.1948 -56.4648]]
julia> sdest(fm1)
49.51010032173714

julia> std(fm1)
2-element Array{Array{Float64,1},1}:
 [37.2603]
 [49.5101]

julia> stderr(fm1)
1-element Array{Float64,1}:
 17.6946

julia> varest(fm1)
2451.2500338684763

julia> vcov(fm1)
1×1 Array{Float64,2}:
 313.097

````


