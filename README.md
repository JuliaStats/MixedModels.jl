# Linear mixed-effects models in [Julia](http://julialang.org)

[![Build Status](https://travis-ci.org/dmbates/MixedModels.jl.svg?branch=master)](https://travis-ci.org/dmbates/MixedModels.jl)
[![Coverage Status](https://img.shields.io/coveralls/dmbates/MixedModels.jl.svg)](https://coveralls.io/r/dmbates/MixedModels.jl?branch=master)
[![MixedModels](http://pkg.julialang.org/badges/MixedModels_0.3.svg)](http://pkg.julialang.org/?pkg=MixedModels&ver=0.3)
[![MixedModels](http://pkg.julialang.org/badges/MixedModels_nightly.svg)](http://pkg.julialang.org/?pkg=MixedModels&ver=nightly)

## Fitting linear mixed-effects models

The `lmm` function is similar to the `lmer` function in the
[lme4](http://cran.R-project.org/package=lme4) package for
[R](http://www.R-project.org).  The first two arguments for in the `R`
version are `formula` and `data`.  The principle method for the
`Julia` version takes these arguments.

### A model fit to the `Dyestuff` data from the `lme4` package

The simplest example of a mixed-effects model that we use in the
[lme4 package for R](https://github.com/lme4/lme4) is a model fit to
the `Dyestuff` data.

```R
> str(Dyestuff)
'data.frame':	30 obs. of  2 variables:
 $ Batch: Factor w/ 6 levels "A","B","C","D",..: 1 1 1 1 1 2 2 2 2 2 ...
 $ Yield: num  1545 1440 1440 1520 1580 ...
> (fm1 <- lmer(Yield ~ 1|Batch, Dyestuff, REML=FALSE))
Linear mixed model fit by maximum likelihood ['lmerMod']
Formula: Yield ~ 1 | Batch
   Data: Dyestuff

      AIC       BIC    logLik  deviance
 333.3271  337.5307 -163.6635  327.3271

Random effects:
 Groups   Name        Variance Std.Dev.
 Batch    (Intercept) 1388     37.26
 Residual             2451     49.51
 Number of obs: 30, groups: Batch, 6

Fixed effects:
            Estimate Std. Error t value
(Intercept)  1527.50      17.69   86.33
```

These `Dyestuff` data are available in the `RDatasets` package for `julia`
```julia
julia> using MixedModels, RDatasets

julia> ds = dataset("lme4","Dyestuff");

julia> head(ds)
6x2 DataFrame
|-------|-------|-------|
| Row # | Batch | Yield |
| 1     | A     | 1545  |
| 2     | A     | 1440  |
| 3     | A     | 1440  |
| 4     | A     | 1520  |
| 5     | A     | 1580  |
| 6     | B     | 1540  |
```

`lmm` defaults to maximum likelihood estimation whereas `lmer` in `R`
defaults to REML estimation.

```julia
julia> m = fit(lmm(Yield ~ 1 + (1|Batch), ds))
Linear mixed model fit by maximum likelihood
Formula: Yield ~ 1 + (1 | Batch)

 logLik: -163.663530, deviance: 327.327060

 Variance components:
                Variance    Std.Dev.
 Batch        1388.331690   37.260323
 Residual     2451.250503   49.510105
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value
(Intercept)    1527.5   17.6945  86.326
```

In general the model should be fit through an explicit call to the `fit`
function, which may take a second argument indicating a verbose fit.

```julia
julia> gc(); @time fit(lmm(Yield ~ 1 + (1|Batch),ds),true);
f_1: 327.76702, [1.0]
f_2: 328.63496, [0.428326]
f_3: 327.33773, [0.787132]
f_4: 328.27031, [0.472809]
f_5: 327.33282, [0.727955]
f_6: 327.32706, [0.752783]
f_7: 327.32706, [0.752599]
f_8: 327.32706, [0.752355]
f_9: 327.32706, [0.752575]
f_10: 327.32706, [0.75258]
FTOL_REACHED
elapsed time: 0.165568641 seconds (2007424 bytes allocated)
```

The numeric representation of the model has type
```julia
julia> typeof(m)
LinearMixedModel{PLSOne} (constructor with 2 methods)
```
A `LinearMixedModel` is parameterized by the type of the solver
for the penalized least-squares (PLS) problem.  The simplicity of the
PLS problem when there is a single grouping factor is exploited in the
`PLSOne` class which provides for evaluation of the objective function
and of its gradient.

Those familiar with the `lme4` package for `R` will see the usual
suspects.
```julia
julia> fixef(m)  # estimates of the fixed-effects parameters
1-element Array{Float64,1}:
 1527.5

julia> show(coef(m))  # another name for fixef
[1527.5]

julia> ranef(m)
1-element Array{Any,1}:
 1x6 Array{Float64,2}:
 -16.6282  0.369516  26.9747  -21.8014  53.5798  -42.4943

julia> ranef(m,true)  # on the u scale
1-element Array{Any,1}:
 1x6 Array{Float64,2}:
 -22.0949  0.490999  35.8429  -28.9689  71.1948  -56.4649

julia> deviance(m)
327.3270598811376
```

## A more substantial example

Fitting a model to the `Dyestuff` data is trivial.  The `InstEval`
data in the `lme4` package is more of a challenge in that there are
nearly 75,000 evaluations by 2972 students on a total of 1128
instructors.

```julia
julia> inst = dataset("lme4","InstEval");

julia> head(inst)
6x7 DataFrame
|-------|---|------|---------|---------|---------|------|---|
| Row # | S | D    | Studage | Lectage | Service | Dept | Y |
| 1     | 1 | 1002 | 2       | 2       | 0       | 2    | 5 |
| 2     | 1 | 1050 | 2       | 1       | 1       | 6    | 2 |
| 3     | 1 | 1582 | 2       | 2       | 0       | 2    | 5 |
| 4     | 1 | 2050 | 2       | 2       | 1       | 3    | 3 |
| 5     | 2 | 115  | 2       | 1       | 0       | 5    | 2 |
| 6     | 2 | 756  | 2       | 1       | 0       | 5    | 4 |

julia> fm2 = fit(lmm(Y ~ 1 + Dept*Service + (1|S) + (1|D), inst))
Linear mixed model fit by maximum likelihood
Formula: Y ~ Dept * Service + (1 | S) + (1 | D)

 logLik: -118792.776709, deviance: 237585.553417

 Variance components:
                Variance    Std.Dev.
 S              0.105422    0.324688
 D              0.258429    0.508359
 Residual       1.384725    1.176744
 Number of obs: 73421; levels of grouping factors: 2972, 1128

  Fixed-effects parameters:
                   Estimate Std.Error   z value
(Intercept)         3.22961 0.0640541     50.42
Dept5              0.129537  0.101295    1.2788
Dept10            -0.176752 0.0881368  -2.00543
Dept12            0.0517089 0.0817538  0.632495
Dept6             0.0347327 0.0856225  0.405649
Dept7              0.145941 0.0998001   1.46233
Dept4              0.151689 0.0816911   1.85686
Dept8              0.104206  0.118752  0.877503
Dept9             0.0440392 0.0963003  0.457312
Dept14            0.0517545 0.0986047  0.524868
Dept1             0.0466714  0.101944  0.457815
Dept3             0.0563455 0.0977943  0.576164
Dept11            0.0596525  0.100235  0.595129
Dept2            0.00556088  0.110869 0.0501574
Service1           0.252024 0.0686508    3.6711
Dept5&Service1    -0.180759  0.123179  -1.46744
Dept10&Service1   0.0186497  0.110017  0.169517
Dept12&Service1   -0.282267 0.0792939  -3.55975
Dept6&Service1    -0.494464  0.079028  -6.25682
Dept7&Service1    -0.392054  0.110313  -3.55402
Dept4&Service1    -0.278546 0.0823729  -3.38152
Dept8&Service1    -0.189526   0.11145  -1.70055
Dept9&Service1    -0.499867 0.0885425   -5.6455
Dept14&Service1   -0.497161 0.0917165  -5.42063
Dept1&Service1    -0.240418 0.0982074  -2.44807
Dept3&Service1    -0.223013  0.089055  -2.50421
Dept11&Service1   -0.516996 0.0809079  -6.38994
Dept2&Service1    -0.384769 0.0918433  -4.18941

julia> gc();@time fit(lmm(Y ~ 1 + Dept*Service + (1|S) + (1|D), inst));
elapsed time: 5.193356844 seconds (327515804 bytes allocated, 4.95% gc time)
```

Models with vector-valued random effects can be fit
```julia
julia> slp = dataset("lme4","sleepstudy")
180x3 DataFrame
|-------|----------|------|---------|
| Row # | Reaction | Days | Subject |
| 1     | 249.56   | 0    | 308     |
| 2     | 258.705  | 1    | 308     |
| 3     | 250.801  | 2    | 308     |
| 4     | 321.44   | 3    | 308     |
| 5     | 356.852  | 4    | 308     |
| 6     | 414.69   | 5    | 308     |
| 7     | 382.204  | 6    | 308     |
| 8     | 290.149  | 7    | 308     |
| 9     | 430.585  | 8    | 308     |
⋮
| 171   | 269.412  | 0    | 372     |
| 172   | 273.474  | 1    | 372     |
| 173   | 297.597  | 2    | 372     |
| 174   | 310.632  | 3    | 372     |
| 175   | 287.173  | 4    | 372     |
| 176   | 329.608  | 5    | 372     |
| 177   | 334.482  | 6    | 372     |
| 178   | 343.22   | 7    | 372     |
| 179   | 369.142  | 8    | 372     |
| 180   | 364.124  | 9    | 372     |

julia> fm3 = fit(lmm(Reaction ~ 1 + Days + (1+Days|Subject), slp))
Linear mixed model fit by maximum likelihood
Formula: Reaction ~ 1 + Days + ((1 + Days) | Subject)

 logLik: -875.969672, deviance: 1751.939344

 Variance components:
                Variance    Std.Dev.  Corr.
 Subject      565.516376   23.780588
               32.682265    5.716840   0.08
 Residual     654.940901   25.591813
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
             Estimate Std.Error z value
(Intercept)   251.405   6.63228 37.9063
Days          10.4673   1.50224 6.96779
```

For models with a single random-effects term a gradient-based
optimization is used, allowing faster and more reliable convergence to
the parameter estimates.

```julia
julia> gc(); @time fit(lmm(Reaction ~ 1 + Days + (1+Days|Subject),slp),true);
f_1: 1784.6423, [1.0,0.0,1.0]
f_2: 1792.09158, [1.04647,-0.384052,0.159046]
f_3: 1759.76629, [1.00506,-0.0847897,0.418298]
f_4: 1787.91236, [1.26209,0.662287,0.0]
f_5: 1770.2265, [1.04773,0.323752,0.0]
f_6: 1755.6188, [1.00967,0.0107469,0.150327]
f_7: 1762.85008, [0.991808,0.14307,0.446863]
f_8: 1753.29754, [1.0048,0.0534958,0.272807]
f_9: 1752.5881, [1.00312,0.0418443,0.252944]
f_10: 1767.54407, [0.99451,0.00122224,0.0940196]
f_11: 1752.21061, [1.00224,0.0373251,0.232541]
f_12: 1758.83812, [0.988744,0.0206109,0.123804]
f_13: 1752.13481, [1.00085,0.0355465,0.220038]
f_14: 1752.02982, [0.980566,0.015964,0.234045]
f_15: 1759.88963, [0.971299,0.0118275,0.120811]
f_16: 1751.98896, [0.979624,0.0155451,0.221269]
f_17: 1751.98436, [0.97888,0.015535,0.224081]
f_18: 1751.96796, [0.968696,0.0144745,0.223608]
f_19: 1752.08826, [0.867754,0.0226905,0.23397]
f_20: 1751.9463, [0.943112,0.0163709,0.226009]
f_21: 1754.13022, [0.834535,0.0100178,0.166328]
f_22: 1751.94908, [0.930934,0.0157232,0.218808]
f_23: 1751.94123, [0.938201,0.0161115,0.223087]
f_24: 1751.96529, [0.894427,0.0196256,0.223832]
f_25: 1751.93978, [0.930555,0.0167127,0.223213]
f_26: 1751.93974, [0.930506,0.019329,0.222173]
f_27: 1751.94419, [0.913941,0.018183,0.222681]
f_28: 1751.93955, [0.927971,0.0191544,0.22225]
f_29: 1751.93979, [0.933502,0.0174017,0.2225]
f_30: 1751.93942, [0.92986,0.0185533,0.222336]
f_31: 1751.9544, [0.903287,0.0173483,0.222935]
f_32: 1751.93944, [0.927141,0.0184317,0.222396]
f_33: 1751.93939, [0.928786,0.0185053,0.222359]
f_34: 1751.93935, [0.929171,0.0180663,0.222656]
f_35: 1751.93935, [0.929702,0.0182395,0.222667]
f_36: 1751.93934, [0.929337,0.0181204,0.222659]
f_37: 1751.93935, [0.928905,0.0183732,0.222647]
f_38: 1751.93934, [0.929269,0.0181603,0.222657]
f_39: 1751.93935, [0.929106,0.0181461,0.222618]
f_40: 1751.93934, [0.929236,0.0181575,0.22265]
f_41: 1751.93934, [0.929197,0.0181927,0.222625]
f_42: 1751.93934, [0.929229,0.018164,0.222645]
f_43: 1751.93934, [0.929146,0.0181729,0.22267]
f_44: 1751.93934, [0.929221,0.0181649,0.222647]
f_45: 1751.93934, [0.929226,0.0181643,0.222646]
FTOL_REACHED
elapsed time: 0.022856979 seconds (1140940 bytes allocated)
```
