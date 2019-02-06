# Model constructors

The `LinearMixedModel` type represents a linear mixed-effects model.
Typically it is constructed from a `Formula` and an appropriate `data` type, usually a `DataFrame`.
```@docs
LinearMixedModel
```

## Examples of linear mixed-effects model fits

For illustration, several data sets from the *lme4* package for *R* are made available in `.rda` format in this package.
These include the `Dyestuff` and `Dyestuff2` data sets.
````julia
julia> using DataFrames, MixedModels, RData, StatsBase

julia> const dat = Dict(Symbol(k)=>v for (k,v) in 
    load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")));

julia> describe(dat[:Dyestuff])
2×8 DataFrames.DataFrame. Omitted printing of 1 columns
│ Row │ variable │ mean   │ min    │ median │ max    │ nunique │ nmissing │
│     │ Symbol   │ Union… │ Any    │ Union… │ Any    │ Union…  │ Nothing  │
├─────┼──────────┼────────┼────────┼────────┼────────┼─────────┼──────────┤
│ 1   │ G        │        │ A      │        │ F      │ 6       │          │
│ 2   │ Y        │ 1527.5 │ 1440.0 │ 1530.0 │ 1635.0 │         │          │

````




The columns in these data sets have been renamed for convenience.
The response is always named `Y`.
Potential grouping factors for random-effects terms are named `G`, `H`, etc.
Numeric covariates are named starting with `U`.
Categorical covariates not suitable as grouping factors are named starting with `A`.


### Models with simple, scalar random effects

The formula language in *Julia* is similar to that in *R* except that the formula must be enclosed in a call to the `@formula` macro.
A basic model with simple, scalar random effects for the levels of `G` (the batch of an intermediate product, in this case) is declared and fit as
````julia
julia> fm1 = fit!(LinearMixedModel(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff]))
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


````





(If you are new to Julia you may find that this first fit takes an unexpectedly long time, due to Just-In-Time (JIT) compilation of the code.
The second and subsequent calls to such functions are much faster.)

````julia
julia> @time fit!(LinearMixedModel(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff2]));
  0.004960 seconds (2.09 k allocations: 87.922 KiB)

````





By default, the model fit is by maximum likelihood.  To use the `REML` criterion instead, add the optional named argument `REML = true` to the call to `fit!`
````julia
julia> fm1R = fit!(LinearMixedModel(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff]), REML=true)
Linear mixed model fit by REML
 Formula: Y ~ 1 + (1 | G)
 REML criterion at convergence: 319.65427684225216

Variance components:
              Column    Variance  Std.Dev. 
 G        (Intercept)  1764.0510 42.000607
 Residual              2451.2498 49.510098
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)    1527.5   19.3834 78.8045  <1e-99


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
julia> fm2 = fit(LinearMixedModel, @formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy])
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
julia> fm3 = fit(LinearMixedModel, @formula(Y ~ 1 + U + (1|G) + (0+U|G)), dat[:sleepstudy])
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + U + (1 | G) + ((0 + U) | G)
   logLik   -2 logLik     AIC        BIC    
 -876.00163 1752.00326 1762.00326 1777.96804

Variance components:
              Column    Variance  Std.Dev.   Corr.
 G        (Intercept)  584.258973 24.17145
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
julia> fm4 = fit(LinearMixedModel, @formula(Y ~ 1 + (1|G) + (1|H)), dat[:Penicillin])
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G) + (1 | H)
   logLik   -2 logLik     AIC        BIC    
 -166.09417  332.18835  340.18835  352.06760

Variance components:
              Column    Variance   Std.Dev. 
 G        (Intercept)  0.71497949 0.8455646
 H        (Intercept)  3.13519326 1.7706477
 Residual              0.30242640 0.5499331
 Number of obs: 144; levels of grouping factors: 24, 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   22.9722  0.744596 30.8519  <1e-99


````





In contrast the sample, `G`, grouping factor is *nested* within the batch, `H`, grouping factor in the *Pastes* data.
That is, each level of `G` occurs in conjunction with only one level of `H`.
````julia
julia> fm5 = fit(LinearMixedModel, @formula(Y ~ 1 + (1|G) + (1|H)), dat[:Pastes])
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G) + (1 | H)
   logLik   -2 logLik     AIC        BIC    
 -123.99723  247.99447  255.99447  264.37184

Variance components:
              Column    Variance  Std.Dev.  
 G        (Intercept)  8.4336167 2.90406899
 H        (Intercept)  1.1991787 1.09507018
 Residual              0.6780021 0.82340886
 Number of obs: 60; levels of grouping factors: 30, 10

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   60.0533  0.642136 93.5212  <1e-99


````





In observational studies it is common to encounter *partially crossed* grouping factors.
For example, the *InstEval* data are course evaluations by students, `G`, of instructors, `H`.
Additional covariates include the academic department, `I`, in which the course was given and `A`, whether or not it was a service course.
````julia
julia> fm6 = fit(LinearMixedModel, @formula(Y ~ 1 + A * I + (1|G) + (1|H)), dat[:InstEval])
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + A + I + A & I + (1 | G) + (1 | H)
     logLik        -2 logLik          AIC             BIC       
 -1.18792777×10⁵  2.37585553×10⁵  2.37647553×10⁵  2.37932876×10⁵

Variance components:
              Column    Variance   Std.Dev.  
 G        (Intercept)  0.10541790 0.32468122
 H        (Intercept)  0.25841634 0.50834667
 Residual              1.38472780 1.17674458
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
I: 2          0.00556285  0.110867 0.050176  0.9600
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

To create a GLMM representation
```@docs
GeneralizedLinearMixedModel
```
the distribution family for the response, and possibly the link function, must be specified.

````julia
julia> verbaggform = @formula(r2 ~ 1 + a + g + b + s + m + (1|id) + (1|item));

julia> gm1 = fit(GeneralizedLinearMixedModel, verbaggform, dat[:VerbAgg], Bernoulli())
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  Formula: r2 ~ 1 + a + g + b + s + m + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 8135.8329

Variance components:
          Column     Variance   Std.Dev. 
 id   (Intercept)  1.793480630 1.3392090
 item (Intercept)  0.117154495 0.3422784

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
              Estimate Std.Error  z value P(>|z|)
(Intercept)   0.553223  0.385364  1.43558  0.1511
a            0.0574216 0.0167528  3.42759  0.0006
g: M          0.320801  0.191207  1.67777  0.0934
b: scold      -1.05979  0.184162 -5.75464   <1e-8
b: shout       -2.1038  0.186521 -11.2792  <1e-28
s: self       -1.05402  0.151197 -6.97116  <1e-11
m: do        -0.707036   0.15101 -4.68205   <1e-5


````





The canonical link, which is `GLM.LogitLink` for the `Bernoulli` distribution, is used if no explicit link is specified.

Note that, in keeping with convention in the [`GLM` package](https://github.com/JuliaStats/GLM.jl), the distribution family for a binary (i.e. 0/1) response is the `Bernoulli` distribution.
The `Binomial` distribution is only used when the response is the fraction of trials returning a positive, in which case the number of trials must be specified as the case weights.

### Optional arguments to fit!

An alternative approach is to create the `GeneralizedLinearMixedModel` object then call `fit!` on it.
In this form optional arguments `fast` and/or `nAGQ` can be passed to the optimization process.

As the name implies, `fast=true`, provides a faster but somewhat less accurate fit.
These fits may suffice for model comparisons.
````julia
julia> gm1a = fit!(GeneralizedLinearMixedModel(verbaggform, dat[:VerbAgg], Bernoulli()), fast=true)
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  Formula: r2 ~ 1 + a + g + b + s + m + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 8136.1709

Variance components:
          Column    Variance   Std.Dev.  
 id   (Intercept)  1.79270002 1.33891748
 item (Intercept)  0.11875573 0.34460953

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
              Estimate Std.Error  z value P(>|z|)
(Intercept)   0.548543  0.385673   1.4223  0.1549
a            0.0543802 0.0167462  3.24732  0.0012
g: M          0.304244  0.191141  1.59172  0.1114
b: scold      -1.01749  0.185216 -5.49352   <1e-7
b: shout      -2.02067  0.187522 -10.7756  <1e-26
s: self       -1.01255   0.15204 -6.65975  <1e-10
m: do        -0.679102  0.151857 -4.47198   <1e-5


julia> deviance(gm1a) - deviance(gm1)
0.33801208853947173

julia> @time fit(GeneralizedLinearMixedModel, verbaggform, dat[:VerbAgg], Bernoulli());
 48.354810 seconds (50.44 M allocations: 425.030 MiB, 0.19% gc time)

julia> @time fit!(GeneralizedLinearMixedModel(verbaggform, dat[:VerbAgg], Bernoulli()), fast=true);
  1.455715 seconds (2.43 M allocations: 26.565 MiB, 0.50% gc time)

````





The optional argument `nAGQ=k` causes evaluation of the deviance function to use a `k` point
adaptive Gauss-Hermite quadrature rule.
This method only applies to models with a single, simple, scalar random-effects term, such as
````julia
julia> contraform = @formula(use ~ 1 + a + l + urb + (1|d))
Formula: use ~ 1 + a + l + urb + (1 | d)

julia> @time gm2 = fit!(GeneralizedLinearMixedModel(contraform, dat[:Contraception], Bernoulli()), nAGQ=9)
  1.707273 seconds (10.05 M allocations: 155.271 MiB, 3.82% gc time)
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 9)
  Formula: use ~ 1 + a + l + urb + (1 | d)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 2413.3485

Variance components:
       Column    Variance   Std.Dev.  
 d (Intercept)  0.21548177 0.46420014

 Number of obs: 1934; levels of grouping factors: 60

Fixed-effects parameters:
               Estimate  Std.Error  z value P(>|z|)
(Intercept)    -1.68982   0.145706 -11.5975  <1e-30
a            -0.0265921 0.00782925 -3.39651  0.0007
l: 1            1.10911   0.156852  7.07105  <1e-11
l: 2            1.37627   0.173343  7.93957  <1e-14
l: 3+            1.3453   0.177809  7.56598  <1e-13
urb: Y          0.73235   0.118484  6.18102   <1e-9


julia> @time deviance(fit!(GeneralizedLinearMixedModel(contraform, dat[:Contraception], Bernoulli()), nAGQ=9, fast=true))
  0.129813 seconds (431.83 k allocations: 5.154 MiB)
2413.663718869012

julia> @time deviance(fit!(GeneralizedLinearMixedModel(contraform, dat[:Contraception], Bernoulli())))
  0.689261 seconds (4.60 M allocations: 42.840 MiB, 1.82% gc time)
2413.6156912345245

julia> @time deviance(fit!(GeneralizedLinearMixedModel(contraform, dat[:Contraception], Bernoulli()), fast=true))
  0.084874 seconds (225.54 k allocations: 3.042 MiB)
2413.6618664984017

````





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
-163.66352994056865

julia> aic(fm1)
333.3270598811373

julia> bic(fm1)
337.5306520261238

julia> dof(fm1)   # 1 fixed effect, 2 variances
3

julia> nobs(fm1)  # 30 observations
30

julia> loglikelihood(gm1)
-4067.916429808715

````





In general the [`deviance`](https://en.wikipedia.org/wiki/Deviance_(statistics)) of a statistical model fit is negative twice the log-likelihood adjusting for the saturated model.
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
327.3270598811373

julia> deviance(fm1)
327.3270598811373

````





The value optimized when fitting a `GeneralizedLinearMixedModel` is the Laplace approximation to the deviance or an adaptive Gauss-Hermite evaluation.
```@docs
deviance!
```
````julia
julia> MixedModels.deviance!(gm1)
8135.832859617447

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
[0.0574216, -1.05402, -0.707036, -1.05979, 0.320801, -2.1038, 0.553223]
````





An alternative extractor for the fixed-effects coefficient is the `β` property.
Properties whose names are Greek letters usually have an alternative spelling, which is the name of the Greek letter.
````julia
julia> show(fm1.β)
[1527.5]
julia> show(fm1.beta)
[1527.5]
julia> show(gm1.β)
[0.0574216, -1.05402, -0.707036, -1.05979, 0.320801, -2.1038, 0.553223]
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
  0.148506    -0.00560462   -0.00977081   …  -0.0114554    -0.0114566  
 -0.00560462   0.000280655   7.19123e-5      -1.47964e-5   -1.02415e-5 
 -0.00977081   7.19123e-5    0.03656         -8.04385e-5   -5.25882e-5 
 -0.0169716   -1.43714e-5   -9.25614e-5       0.000265781   0.000172095
 -0.017144    -2.90564e-5   -0.000162389      0.000658924   0.000520519
 -0.0114554   -1.47964e-5   -8.04385e-5   …   0.0228606     0.00024777 
 -0.0114566   -1.02415e-5   -5.25882e-5       0.00024777    0.0228041  

````





The standard errors are the square roots of the diagonal elements of the estimated variance-covariance matrix of the fixed-effects coefficient estimators.
```@docs
stderror
```
````julia
julia> show(StatsBase.stderror(fm2))
[6.63226, 1.50224]
julia> show(StatsBase.stderror(gm1))
[0.385364, 0.0167528, 0.191207, 0.184162, 0.186521, 0.151197, 0.15101]
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
          Column     Variance   Std.Dev. 
 id   (Intercept)  1.793480630 1.3392090
 item (Intercept)  0.117154495 0.3422784


````





Individual components are returned by other extractors
```@docs
varest
sdest
```
````julia
julia> varest(fm2)
654.941450830681

julia> sdest(fm2)
25.591823905901684

julia> fm2.σ
25.591823905901684

````





## Conditional modes of the random effects

The `ranef` extractor
```@docs
ranef
```
````julia
julia> ranef(fm1)
1-element Array{Array{Float64,2},1}:
 [-16.6282 0.369516 … 53.5798 -42.4943]

julia> fm1.b
1-element Array{Array{Float64,2},1}:
 [-16.6282 0.369516 … 53.5798 -42.4943]

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


