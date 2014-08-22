# Linear mixed-effects models in [Julia](http://julialang.org)

[![Build Status](https://travis-ci.org/dmbates/MixedModels.jl.svg?branch=master)](https://travis-ci.org/dmbates/MixedModels.jl)
[![Coverage Status](https://img.shields.io/coveralls/dmbates/MixedModels.jl.svg)](https://coveralls.io/r/dmbates/MixedModels.jl?branch=master)
[![Package Evaluator](http://iainnz.github.io/packages.julialang.org/badges/MixedModels_0.3.svg)](http://iainnz.github.io/packages.julialang.org/?pkg=MixedModels&ver=0.3)

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
julia> m = lmm(Yield ~ 1|Batch, ds)
Linear mixed model fit by maximum likelihood
Formula: Yield ~ 1 | Batch

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

(The formatting of the output will be improved.)

In general the model should be fit through an explicit call to the `fit`
function, which may take a second argument indicating a verbose fit.

```julia
julia> @time fit(lmm(Yield ~ 1|Batch, ds),true);
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
elapsed time: 0.002485569 seconds (172340 bytes allocated)
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

julia> fm2 = fit(lmm(Y ~ Dept*Service + (1|S) + (1|D), inst))
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

julia> @time fit(lmm(Y ~ Dept*Service + (1|S) + (1|D), inst));
elapsed time: 4.454070061 seconds (282382860 bytes allocated, 5.25% gc time)
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

julia> fm3 = fit(lmm(Reaction ~ Days + (Days|Subject), slp))
Linear mixed model fit by maximum likelihood
Formula: Reaction ~ Days + (Days | Subject)

 logLik: -875.969673, deviance: 1751.939345

 Variance components:
                Variance    Std.Dev.  Corr.
 Subject      565.545224   23.781195
               32.692286    5.717717   0.08
 Residual     654.919662   25.591398
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
             Estimate Std.Error z value
(Intercept)   251.405   6.63237 37.9058
Days          10.4673   1.50242 6.96696
```

For models with a single random-effects term a gradient-based
optimization is used, allowing faster and more reliable convergence to
the parameter estimates.

```julia
julia> @time fit(lmm(Reaction ~ Days + (Days|Subject), sleep),true);
f_1: 1784.6423, [1.0,0.0,1.0]
f_2: 1792.09158, [1.04647,-0.384052,0.159046]
f_3: 1759.76629, [1.00506,-0.0847897,0.418298]
f_4: 1784.41102, [1.26209,0.585376,0.0]
f_5: 1766.18565, [1.04773,0.216202,0.0]
f_6: 1757.57929, [1.00967,-0.0297793,0.150327]
f_7: 1762.55317, [1.00916,0.0952413,0.460103]
f_8: 1753.38213, [1.00952,0.0132403,0.289069]
f_9: 1752.71167, [1.00829,0.0111682,0.268488]
f_10: 1767.20917, [0.989955,0.00394333,0.093839]
f_11: 1752.20432, [1.00641,0.0104321,0.245475]
f_12: 1766.28374, [0.996655,0.00871408,0.0948765]
f_13: 1752.04419, [1.00541,0.0102592,0.227167]
f_14: 1752.29042, [0.990437,0.0108813,0.201046]
f_15: 1752.03919, [1.00378,0.0103252,0.224327]
f_16: 1752.21202, [0.981312,0.0109606,0.248824]
f_17: 1752.03339, [1.00056,0.0104138,0.227803]
f_18: 1752.56869, [0.959066,0.0110189,0.190357]
f_19: 1752.02098, [0.99584,0.0104793,0.223568]
f_20: 1752.02236, [0.935085,0.0130369,0.237627]
f_21: 1751.98102, [0.968279,0.0116004,0.229797]
f_22: 1754.94258, [0.782918,0.0187765,0.156352]
f_23: 1751.9537, [0.946424,0.0123312,0.220891]
f_24: 1751.94441, [0.939076,0.0147794,0.225593]
f_25: 1751.94131, [0.936238,0.015371,0.223279]
f_26: 1751.96154, [0.906446,0.0261573,0.223694]
f_27: 1751.93965, [0.931217,0.0171913,0.223348]
f_28: 1751.94851, [0.915681,0.0244214,0.221289]
f_29: 1751.93944, [0.929602,0.0179454,0.223133]
f_30: 1751.93945, [0.928801,0.0182069,0.222154]
f_31: 1751.93935, [0.929249,0.0180608,0.222699]
f_32: 1751.96659, [0.940688,0.0279543,0.222249]
f_33: 1751.93959, [0.930398,0.0190622,0.222654]
f_34: 1751.93935, [0.929364,0.018161,0.222695]
f_35: 1751.93935, [0.929209,0.0181365,0.222689]
f_36: 1751.93935, [0.929265,0.0181483,0.222685]
FTOL_REACHED
elapsed time: 0.024221732 seconds (1768688 bytes allocated)
```

