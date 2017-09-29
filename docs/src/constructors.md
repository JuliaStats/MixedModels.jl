# Model constructors

The `lmm` function creates a linear mixed-effects model representation from a formula/data representation.
```@docs
lmm
```

## Examples of linear mixed-effects model fits

For illustration, data sets from the *lme4* package for *R* are made available in `.rda` format in this package.
````julia
julia> using DataFrames, RData, MixedModels

julia> const dat = convert(Dict{Symbol,DataFrame}, load(Pkg.dir("MixedModels", "test", "dat.rda")));

julia> dat[:Dyestuff]
30×2 DataFrames.DataFrame
│ Row │ G   │ Y      │
├─────┼─────┼────────┤
│ 1   │ "A" │ 1545.0 │
│ 2   │ "A" │ 1440.0 │
│ 3   │ "A" │ 1440.0 │
│ 4   │ "A" │ 1520.0 │
│ 5   │ "A" │ 1580.0 │
│ 6   │ "B" │ 1540.0 │
│ 7   │ "B" │ 1555.0 │
│ 8   │ "B" │ 1490.0 │
⋮
│ 22  │ "E" │ 1630.0 │
│ 23  │ "E" │ 1515.0 │
│ 24  │ "E" │ 1635.0 │
│ 25  │ "E" │ 1625.0 │
│ 26  │ "F" │ 1520.0 │
│ 27  │ "F" │ 1455.0 │
│ 28  │ "F" │ 1450.0 │
│ 29  │ "F" │ 1480.0 │
│ 30  │ "F" │ 1445.0 │

````




These are the `Dyestuff` data from the *lme4* package.
The columns have been renamed for convenience in comparing models between examples.
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
julia> @time fit!(lmm(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff2]))
  0.000995 seconds (1.50 k allocations: 81.748 KiB)
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





### Models with vector-valued random effects

The *sleepstudy* data are observations of reaction time, `Y`, on several subjects, `G`, after 0 to 9 days of sleep deprivation.
A model with random intercepts and, possibly correlated, random slopes for each subject is fit as
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
 G        (Intercept)  584.258968 24.17145
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
 H        (Intercept)  3.1351920 1.7706474
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
 G        (Intercept)  8.4336166 2.90406897
 H        (Intercept)  1.1991794 1.09507048
 Residual              0.6780021 0.82340884
 Number of obs: 60; levels of grouping factors: 30, 10

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   60.0533  0.642136 93.5212  <1e-99


````





In observational studies it is common to encounter *partially crossed* grouping factors.
For example, the *InstEval* data are course evaluations by students, `G`, of instructors, `H`.
Additional covariates include the academic department, `H`, in which the course was given and `A`, whether or not it was a service course.
````julia
julia> fm6 = fit!(lmm(@formula(Y ~ 1 + A * I + (1|G) + (1|H)), dat[:InstEval]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + A * I + (1 | G) + (1 | H)
     logLik        -2 logLik          AIC             BIC       
 -1.18792777×10⁵  2.37585553×10⁵  2.37647553×10⁵  2.37932876×10⁵

Variance components:
              Column     Variance   Std.Dev.  
 G        (Intercept)  0.105417976 0.32468135
 H        (Intercept)  0.258416368 0.50834670
 Residual              1.384727771 1.17674457
 Number of obs: 73421; levels of grouping factors: 2972, 1128

  Fixed-effects parameters:
                Estimate Std.Error   z value P(>|z|)
(Intercept)      3.22961  0.064053   50.4209  <1e-99
A: 1            0.252025 0.0686507   3.67112  0.0002
I: 5            0.129536  0.101294   1.27882  0.2010
I: 10          -0.176751 0.0881352  -2.00545  0.0449
I: 12          0.0517102 0.0817524  0.632522  0.5270
I: 6           0.0347319  0.085621  0.405647  0.6850
I: 7             0.14594 0.0997984   1.46235  0.1436
I: 4            0.151689 0.0816897   1.85689  0.0633
I: 8            0.104206  0.118751  0.877517  0.3802
I: 9           0.0440401 0.0962985  0.457329  0.6474
I: 14          0.0517546 0.0986029  0.524879  0.5997
I: 1           0.0466719  0.101942  0.457828  0.6471
I: 3           0.0563461 0.0977925   0.57618  0.5645
I: 11          0.0596536  0.100233   0.59515  0.5517
I: 2          0.00556281  0.110867 0.0501756  0.9600
A: 1 & I: 5    -0.180757  0.123179  -1.46744  0.1423
A: 1 & I: 10   0.0186492  0.110017  0.169513  0.8654
A: 1 & I: 12   -0.282269 0.0792937  -3.55979  0.0004
A: 1 & I: 6    -0.494464 0.0790278  -6.25683   <1e-9
A: 1 & I: 7    -0.392054  0.110313  -3.55403  0.0004
A: 1 & I: 4    -0.278547 0.0823727  -3.38154  0.0007
A: 1 & I: 8    -0.189526  0.111449  -1.70056  0.0890
A: 1 & I: 9    -0.499868 0.0885423  -5.64553   <1e-7
A: 1 & I: 14   -0.497162 0.0917162  -5.42065   <1e-7
A: 1 & I: 1     -0.24042 0.0982071   -2.4481  0.0144
A: 1 & I: 3    -0.223013 0.0890548  -2.50422  0.0123
A: 1 & I: 11   -0.516997 0.0809077  -6.38997   <1e-9
A: 1 & I: 2    -0.384773  0.091843  -4.18946   <1e-4


````





## Fitting generalized linear mixed models

To create a GLMM using
```@docs
glmm
```
the distribution family for the response, given the random effects, must be specified.

````julia
julia> gm1 = fit!(glmm(@formula(r2 ~ 1 + a + g + b + s + m + (1|id) + (1|item)), dat[:VerbAgg],
    Bernoulli()))
Generalized Linear Mixed Model fit by minimizing the Laplace approximation to the deviance
  Formula: r2 ~ 1 + a + g + b + s + m + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance (Laplace approximation): 8135.8329

Variance components:
          Column     Variance   Std.Dev. 
 id   (Intercept)  1.793470989 1.3392054
 item (Intercept)  0.117151977 0.3422747

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
              Estimate Std.Error  z value P(>|z|)
(Intercept)   0.553345  0.385363  1.43591  0.1510
a            0.0574211 0.0167527  3.42757  0.0006
g: M          0.320792  0.191206  1.67773  0.0934
b: scold      -1.05975   0.18416 -5.75448   <1e-8
b: shout       -2.1038  0.186519 -11.2793  <1e-28
s: self       -1.05429  0.151196   -6.973  <1e-11
m: do         -0.70698  0.151009 -4.68172   <1e-5


````





The canonical link is used if no link is specified.
