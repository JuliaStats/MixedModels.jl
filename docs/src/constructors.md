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
 Y ~ 1 + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -163.66353  327.32706  333.32706  337.53065

Variance components:
            Column    Variance  Std.Dev. 
G        (Intercept)  1388.3333 37.260345
Residual              2451.2500 49.510100
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
──────────────────────────────────────────────────
             Estimate  Std.Error  z value  P(>|z|)
──────────────────────────────────────────────────
(Intercept)    1527.5    17.6946   86.326   <1e-99
──────────────────────────────────────────────────

````





An alternative expression is
````julia
julia> fm1 = fit(MixedModel, @formula(Y ~ 1 + (1|G)), dat[:Dyestuff])
Linear mixed model fit by maximum likelihood
 Y ~ 1 + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -163.66353  327.32706  333.32706  337.53065

Variance components:
            Column    Variance  Std.Dev. 
G        (Intercept)  1388.3333 37.260345
Residual              2451.2500 49.510100
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
──────────────────────────────────────────────────
             Estimate  Std.Error  z value  P(>|z|)
──────────────────────────────────────────────────
(Intercept)    1527.5    17.6946   86.326   <1e-99
──────────────────────────────────────────────────

````




(If you are new to Julia you may find that this first fit takes an unexpectedly long time, due to Just-In-Time (JIT) compilation of the code.
The second and subsequent calls to such functions are much faster.)

````julia
julia> @time fit(MixedModel, @formula(Y ~ 1 + (1|G)), dat[:Dyestuff2]);
  0.661148 seconds (1.03 M allocations: 55.163 MiB, 7.06% gc time)

````





By default, the model fit is by maximum likelihood.  To use the `REML` criterion instead, add the optional named argument `REML = true` to the call to `fit!`
````julia
julia> fm1R = fit(MixedModel, @formula(Y ~ 1 + (1|G)), dat[:Dyestuff], REML=true)
Linear mixed model fit by REML
 Y ~ 1 + (1 | G)
 REML criterion at convergence: 319.6542768422538

Variance components:
            Column    Variance  Std.Dev. 
G        (Intercept)  1764.0506 42.000602
Residual              2451.2499 49.510099
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
──────────────────────────────────────────────────
             Estimate  Std.Error  z value  P(>|z|)
──────────────────────────────────────────────────
(Intercept)    1527.5    19.3834  78.8045   <1e-99
──────────────────────────────────────────────────

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
julia> fm2 = fit(MixedModel, @formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy])
Linear mixed model fit by maximum likelihood
 Y ~ 1 + U + (1 + U | G)
   logLik   -2 logLik     AIC        BIC    
 -875.96967 1751.93934 1763.93934 1783.09709

Variance components:
            Column    Variance  Std.Dev.   Corr.
G        (Intercept)  565.51069 23.780469
         U             32.68212  5.716828  0.08
Residual              654.94145 25.591824
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
───────────────────────────────────────────────────
             Estimate  Std.Error   z value  P(>|z|)
───────────────────────────────────────────────────
(Intercept)  251.405     6.63226  37.9064    <1e-99
U             10.4673    1.50224   6.96781   <1e-11
───────────────────────────────────────────────────

````





A model with uncorrelated random effects for the intercept and slope by subject is fit as
````julia
julia> fm3 = fit!(zerocorr!(LinearMixedModel(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy])))
Linear mixed model fit by maximum likelihood
 Y ~ 1 + U + (1 + U | G)
   logLik   -2 logLik     AIC        BIC    
 -876.00163 1752.00326 1762.00326 1777.96804

Variance components:
            Column    Variance  Std.Dev.   Corr.
G        (Intercept)  584.258970 24.17145
         U             33.632805  5.79938  0.00
Residual              653.115782 25.55613
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
───────────────────────────────────────────────────
             Estimate  Std.Error   z value  P(>|z|)
───────────────────────────────────────────────────
(Intercept)  251.405     6.70771  37.48      <1e-99
U             10.4673    1.51931   6.88951   <1e-11
───────────────────────────────────────────────────

````





Note that the use of `zerocorr!` requires the model to be constructed, then altered to eliminate
the correlation of the random effects, then fit with a call to the mutating function, `fit!`.
```@docs
zerocorr!
```

### Models with multiple, scalar random-effects terms

A model for the *Penicillin* data incorporates random effects for the plate, `G`, and for the sample, `H`.
As every sample is used on every plate these two factors are *crossed*.
````julia
julia> fm4 = fit(MixedModel, @formula(Y ~ 1 + (1|G) + (1|H)), dat[:Penicillin])
Linear mixed model fit by maximum likelihood
 Y ~ 1 + (1 | G) + (1 | H)
   logLik   -2 logLik     AIC        BIC    
 -166.09417  332.18835  340.18835  352.06760

Variance components:
            Column    Variance   Std.Dev. 
G        (Intercept)  0.71497949 0.8455646
H        (Intercept)  3.13519360 1.7706478
Residual              0.30242640 0.5499331
 Number of obs: 144; levels of grouping factors: 24, 6

  Fixed-effects parameters:
──────────────────────────────────────────────────
             Estimate  Std.Error  z value  P(>|z|)
──────────────────────────────────────────────────
(Intercept)   22.9722   0.744596  30.8519   <1e-99
──────────────────────────────────────────────────

````





In contrast the sample, `G`, grouping factor is *nested* within the batch, `H`, grouping factor in the *Pastes* data.
That is, each level of `G` occurs in conjunction with only one level of `H`.
````julia
julia> fm5 = fit(MixedModel, @formula(Y ~ 1 + (1|G) + (1|H)), dat[:Pastes])
Linear mixed model fit by maximum likelihood
 Y ~ 1 + (1 | G) + (1 | H)
   logLik   -2 logLik     AIC        BIC    
 -123.99723  247.99447  255.99447  264.37184

Variance components:
            Column    Variance   Std.Dev.  
G        (Intercept)  8.43361634 2.90406893
H        (Intercept)  1.19918042 1.09507097
Residual              0.67800208 0.82340882
 Number of obs: 60; levels of grouping factors: 30, 10

  Fixed-effects parameters:
──────────────────────────────────────────────────
             Estimate  Std.Error  z value  P(>|z|)
──────────────────────────────────────────────────
(Intercept)   60.0533   0.642136  93.5212   <1e-99
──────────────────────────────────────────────────

````





In observational studies it is common to encounter *partially crossed* grouping factors.
For example, the *InstEval* data are course evaluations by students, `G`, of instructors, `H`.
Additional covariates include the academic department, `I`, in which the course was given and `A`, whether or not it was a service course.
````julia
julia> fm6 = fit!(LinearMixedModel(@formula(Y ~ 1 + A * I + (1|G) + (1|H)), dat[:InstEval]))
Linear mixed model fit by maximum likelihood
 Y ~ 1 + A + I + A & I + (1 | G) + (1 | H)
     logLik        -2 logLik          AIC             BIC       
 -1.18792777×10⁵  2.37585553×10⁵  2.37647553×10⁵  2.37932876×10⁵

Variance components:
            Column    Variance   Std.Dev.  
G        (Intercept)  0.10541798 0.32468136
H        (Intercept)  0.25841636 0.50834669
Residual              1.38472777 1.17674456
 Number of obs: 73421; levels of grouping factors: 2972, 1128

  Fixed-effects parameters:
────────────────────────────────────────────────────────
                Estimate  Std.Error     z value  P(>|z|)
────────────────────────────────────────────────────────
(Intercept)    3.22961    0.064053   50.4209      <1e-99
A: 1           0.252025   0.0686507   3.67112     0.0002
I: 5           0.129536   0.101294    1.27882     0.2010
I: 10         -0.176751   0.0881352  -2.00545     0.0449
I: 12          0.0517102  0.0817523   0.632522    0.5270
I: 6           0.0347319  0.085621    0.405647    0.6850
I: 7           0.14594    0.0997984   1.46235     0.1436
I: 4           0.151689   0.0816897   1.85689     0.0633
I: 8           0.104206   0.118751    0.877517    0.3802
I: 9           0.0440401  0.0962985   0.457329    0.6474
I: 14          0.0517546  0.0986029   0.524879    0.5997
I: 1           0.0466719  0.101942    0.457828    0.6471
I: 3           0.0563461  0.0977925   0.57618     0.5645
I: 11          0.0596536  0.100233    0.59515     0.5517
I: 2           0.0055628  0.110867    0.0501756   0.9600
A: 1 & I: 5   -0.180757   0.123179   -1.46744     0.1423
A: 1 & I: 10   0.0186492  0.110017    0.169513    0.8654
A: 1 & I: 12  -0.282269   0.0792937  -3.55979     0.0004
A: 1 & I: 6   -0.494464   0.0790278  -6.25683     <1e-9 
A: 1 & I: 7   -0.392054   0.110313   -3.55403     0.0004
A: 1 & I: 4   -0.278547   0.0823727  -3.38154     0.0007
A: 1 & I: 8   -0.189526   0.111449   -1.70056     0.0890
A: 1 & I: 9   -0.499868   0.0885423  -5.64553     <1e-7 
A: 1 & I: 14  -0.497162   0.0917162  -5.42065     <1e-7 
A: 1 & I: 1   -0.24042    0.0982071  -2.4481      0.0144
A: 1 & I: 3   -0.223013   0.0890548  -2.50422     0.0123
A: 1 & I: 11  -0.516997   0.0809077  -6.38997     <1e-9 
A: 1 & I: 2   -0.384773   0.091843   -4.18946     <1e-4 
────────────────────────────────────────────────────────

````





## Fitting generalized linear mixed models

To create a GLMM representation
```@docs
GeneralizedLinearMixedModel
```
the distribution family for the response, and possibly the link function, must be specified.

````julia
julia> verbaggform = @formula(r2 ~ 1 + a + g + b + s + m + (1|id) + (1|item));

julia> gm1 = fit(MixedModel, verbaggform, dat[:VerbAgg], Bernoulli())
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  r2 ~ 1 + a + g + b + s + m + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 8135.8329

Variance components:
        Column    Variance   Std.Dev.  
id   (Intercept)  1.64355891 1.28201362
item (Intercept)  0.10735667 0.32765327

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
─────────────────────────────────────────────────────
              Estimate  Std.Error    z value  P(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.553512  0.368905     1.50042   0.1335
a             0.05742   0.0160373    3.58041   0.0003
g: M          0.320766  0.183041     1.75243   0.0797
b: scold     -1.05987   0.176294    -6.01194   <1e-8 
b: shout     -2.10387   0.178552   -11.783     <1e-31
s: self      -1.05433   0.144737    -7.28446   <1e-12
m: do        -0.707051  0.144558    -4.89112   <1e-5 
─────────────────────────────────────────────────────

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
  r2 ~ 1 + a + g + b + s + m + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 8136.1709

Variance components:
        Column     Variance   Std.Dev.  
id   (Intercept)  1.636122430 1.27911001
item (Intercept)  0.108383391 0.32921633

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
──────────────────────────────────────────────────────
               Estimate  Std.Error    z value  P(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.548543   0.368446     1.4888    0.1365
a             0.0543802  0.0159982    3.39915   0.0007
g: M          0.304244   0.182603     1.66614   0.0957
b: scold     -1.01749    0.176943    -5.75038   <1e-8 
b: shout     -2.02067    0.179146   -11.2795    <1e-28
s: self      -1.01255    0.145248    -6.97114   <1e-11
m: do        -0.679102   0.145074    -4.68108   <1e-5 
──────────────────────────────────────────────────────

julia> deviance(gm1a) - deviance(gm1)
0.33801565450448834

julia> @time fit!(GeneralizedLinearMixedModel(verbaggform, dat[:VerbAgg], Bernoulli()));
  5.250240 seconds (14.91 M allocations: 124.927 MiB, 0.57% gc time)

julia> @time fit!(GeneralizedLinearMixedModel(verbaggform, dat[:VerbAgg], Bernoulli()), fast=true);
  0.867577 seconds (2.38 M allocations: 26.275 MiB, 0.53% gc time)

````





The optional argument `nAGQ=k` causes evaluation of the deviance function to use a `k` point
adaptive Gauss-Hermite quadrature rule.
This method only applies to models with a single, simple, scalar random-effects term, such as
````julia
julia> contraform = @formula(use ~ 1 + a + abs2(a) + l + urb + (1|d))
FormulaTerm
Response:
  use(unknown)
Predictors:
  1
  a(unknown)
  (a)->abs2(a)
  l(unknown)
  urb(unknown)
  (d)->1 | d

julia> @time gm2 = fit!(GeneralizedLinearMixedModel(contraform, dat[:Contraception], Bernoulli()), nAGQ=9)
  4.579913 seconds (9.52 M allocations: 322.623 MiB, 4.09% gc time)
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 9)
  use ~ 1 + a + :(abs2(a)) + l + urb + (1 | d)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 2372.4589

Variance components:
     Column    Variance   Std.Dev. 
d (Intercept)  0.22226267 0.4714474

 Number of obs: 1934; levels of grouping factors: 60

Fixed-effects parameters:
─────────────────────────────────────────────────────────
                Estimate    Std.Error    z value  P(>|z|)
─────────────────────────────────────────────────────────
(Intercept)  -1.03542     0.171936     -6.02211    <1e-8 
a             0.00353273  0.00909429    0.388455   0.6977
abs2(a)      -0.00456321  0.000714442  -6.3871     <1e-9 
l: 1          0.815154    0.159787      5.10151    <1e-6 
l: 2          0.916537    0.182361      5.02594    <1e-6 
l: 3+         0.915357    0.183022      5.00135    <1e-6 
urb: Y        0.696695    0.118143      5.89703    <1e-8 
─────────────────────────────────────────────────────────

julia> @time deviance(fit!(GeneralizedLinearMixedModel(contraform, dat[:Contraception], Bernoulli()), nAGQ=9, fast=true))
  0.101267 seconds (413.46 k allocations: 5.063 MiB)
2372.513592622964

julia> @time deviance(fit!(GeneralizedLinearMixedModel(contraform, dat[:Contraception], Bernoulli())))
  0.398170 seconds (1.50 M allocations: 13.667 MiB)
2372.7285823782704

julia> @time deviance(fit!(GeneralizedLinearMixedModel(contraform, dat[:Contraception], Bernoulli()), fast=true))
  0.074245 seconds (236.92 k allocations: 3.239 MiB, 13.11% gc time)
2372.784429135894

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
-163.66352994057004

julia> aic(fm1)
333.3270598811401

julia> bic(fm1)
337.5306520261266

julia> dof(fm1)   # 1 fixed effect, 2 variances
3

julia> nobs(fm1)  # 30 observations
30

julia> loglikelihood(gm1)
-4067.9164280257423

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
327.3270598811401

julia> deviance(fm1)
327.3270598811401

````





The value optimized when fitting a `GeneralizedLinearMixedModel` is the Laplace approximation to the deviance or an adaptive Gauss-Hermite evaluation.
```@docs
MixedModels.deviance!
```
````julia
julia> MixedModels.deviance!(gm1)
8135.832856051482

````





## Fixed-effects parameter estimates

The `coef` and `fixef` extractors both return the maximum likelihood estimates of the fixed-effects coefficients.
```@docs
coef
fixef
```
````julia
julia> show(coef(fm1))
[1527.4999999999993]
julia> show(fixef(fm1))
[1527.4999999999993]
julia> show(fixef(gm1))
[0.5535116086379206, 0.05742001139348312, 0.32076602931752635, -1.059867665970283, -2.103869984409917, -1.0543329105826895, -0.7070511487154889]
````





An alternative extractor for the fixed-effects coefficient is the `β` property.
Properties whose names are Greek letters usually have an alternative spelling, which is the name of the Greek letter.
````julia
julia> show(fm1.β)
[1527.4999999999993]
julia> show(fm1.beta)
[1527.4999999999993]
julia> show(gm1.β)
[0.5535116086379206, 0.05742001139348312, 0.32076602931752635, -1.059867665970283, -2.103869984409917, -1.0543329105826895, -0.7070511487154889]
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
  0.136091    -0.00513612   -0.00895404   …  -0.0104975    -0.0104986  
 -0.00513612   0.000257194   6.59009e-5      -1.35602e-5   -9.38556e-6 
 -0.00895404   6.59009e-5    0.0335039       -7.37176e-5   -4.81924e-5 
 -0.0155523   -1.31704e-5   -8.4825e-5        0.000243582   0.000157714
 -0.0157104   -2.6628e-5    -0.000148816      0.000603881   0.000477019
 -0.0104975   -1.35602e-5   -7.37176e-5   …   0.0209489     0.000227073
 -0.0104986   -9.38556e-6   -4.81924e-5       0.000227073   0.0208971  

````





The standard errors are the square roots of the diagonal elements of the estimated variance-covariance matrix of the fixed-effects coefficient estimators.
```@docs
stderror
```
````julia
julia> show(StatsBase.stderror(fm2))
[6.632257825314581, 1.502235453639816]
julia> show(StatsBase.stderror(gm1))
[0.36890512061819736, 0.01603727875112492, 0.1830407884432926, 0.1762937885165568, 0.17855176330630151, 0.1447372582156264, 0.14455818519253938]
````





Finally, the `coeftable` generic produces a table of coefficient estimates, their standard errors, and their ratio.
The *p-values* quoted here should be regarded as approximations.
```@docs
coeftable
```
````julia
julia> coeftable(fm2)
───────────────────────────────────────────────────
             Estimate  Std.Error   z value  P(>|z|)
───────────────────────────────────────────────────
(Intercept)  251.405     6.63226  37.9064    <1e-99
U             10.4673    1.50224   6.96781   <1e-11
───────────────────────────────────────────────────

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
G        (Intercept)  565.51069 23.780469
         U             32.68212  5.716828  0.08
Residual              654.94145 25.591824


julia> VarCorr(gm1)
Variance components:
        Column    Variance   Std.Dev.  
id   (Intercept)  1.64355891 1.28201362
item (Intercept)  0.10735667 0.32765327



````





Individual components are returned by other extractors
```@docs
varest
sdest
```
````julia
julia> varest(fm2)
654.9414513956141

julia> sdest(fm2)
25.59182391693906

julia> fm2.σ
25.59182391693906

````





## Conditional modes of the random effects

The `ranef` extractor
```@docs
ranef
```
````julia
julia> ranef(fm1)
1-element Array{Array{Float64,2},1}:
 [-16.62822143006434 0.36951603177972425 … 53.57982460798641 -42.49434365460919]

julia> fm1.b
1-element Array{Array{Float64,2},1}:
 [-16.62822143006434 0.36951603177972425 … 53.57982460798641 -42.49434365460919]

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
 [362.3104715146578]

[362.3104715146578]

[362.3104715146578]

[362.3104715146578]

[362.3104715146578]

[362.3104715146578]

````


