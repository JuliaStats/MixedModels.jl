# Linear mixed-effects models in [Julia](http://julialang.org)

## Installation

This package requires Steve Johnson's
[NLopt](https://github.com/stevengj/NLopt.jl.git) package for
Julia. Before installing the `NLopt` package be sure to read the
installation instructions as it requires you to have installed the
`nlopt` library of C functions.

Once the `NLopt` package is installed,

```julia
Pkg.add("MixedModels")
```

will install this package.

## Fitting linear mixed-effects models

The `lmer` function is similar to the function of the same name in the
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

julia> ds = data("lme4","Dyestuff");

julia> dump(ds)
DataFrame  30 observations of 2 variables
  Batch: PooledDataArray{ASCIIString,Uint8,1}(30) ["A", "A", "A", "A"]
  Yield: DataArray{Float64,1}(30) [1545.0, 1440.0, 1440.0, 1520.0]
```

The main difference from `R` in a simple call to `lmer` is the need to
pass the formula as an expression, which means enclosing it in `:()`.
Also, this version of `lmer` defaults to maximum likelihood estimates.

```julia
julia> fm1 = fit(lmer(:(Yield ~ 1|Batch), ds))
Linear mixed model fit by maximum likelihood
 logLik: -163.6635299406109, deviance: 327.3270598812218

  Variance components:
    Std. deviation scale:[37.26047449632836,49.51007020929394]
    Variance scale:[1388.342959691536,2451.2470521292157]
  Number of obs: 30; levels of grouping factors:[6]

  Fixed-effects parameters:
        Estimate Std.Error z value
[1,]      1527.5   17.6946 86.3258
```

(At present the formatting of the output is less than wonderful.)

In general the model should be fit through an explicit call to the `fit`
function, which may take a second argument indicating a verbose fit.

```julia
julia> m = fit(lmer(:(Yield ~ 1|Batch), ds),true);
f_1: 327.7670216246145, [1.0]
f_2: 331.0361932224437, [1.75]
f_3: 330.6458314144857, [0.25]
f_4: 327.69511270610866, [0.97619]
f_5: 327.56630914532184, [0.928569]
f_6: 327.3825965130752, [0.833327]
f_7: 327.3531545408492, [0.807188]
f_8: 327.34662982410276, [0.799688]
f_9: 327.34100192001785, [0.792188]
f_10: 327.33252535370985, [0.777188]
f_11: 327.32733056112147, [0.747188]
f_12: 327.3286190977697, [0.739688]
f_13: 327.32706023603697, [0.752777]
f_14: 327.3270681545395, [0.753527]
f_15: 327.3270598812218, [0.752584]
FTOL_REACHED
```

The numeric representation of the model has type
```julia
julia> typeof(m)
LMMGeneral{Int32}
```

It happens that `show`ing an object of this type causes the model to
be fit so it is okay to omit the call to `fit` in the REPL (the
interactive Read-Eval-Print-Loop).

Those familiar with the `lme4` package for `R` will see the usual
suspects.
```julia
julia> fixef(m)
1-element Float64 Array:
 1527.5

julia> ranef(m)
1-element Array{Float64,2} Array:
 1x6 Float64 Array:
 -16.6283  0.369517  26.9747  -21.8015  53.5799  -42.4944

julia> ranef(m,true)  # on the U scale
1-element Array{Float64,2} Array:
 1x6 Float64 Array:
 -22.0949  0.490998  35.8428  -28.9689  71.1947  -56.4647

julia> deviance(m)
327.3270598812218
```

## A more substantial example

Fitting a model to the `Dyestuff` data is trivial.  The `InstEval`
data in the `lme4` package is more of a challenge in that there are
nearly 75,000 evaluations by 2972 students on a total of 1128
instructors.

```julia
julia> inst = data("lme4","InstEval");

julia> dump(inst)
DataFrame  73421 observations of 7 variables
  s: PooledDataArray{ASCIIString,Uint16,1}(73421) ["1", "1", "1", "1"]
  d: PooledDataArray{ASCIIString,Uint16,1}(73421) ["1002", "1050", "1582", "2050"]
  studage: PooledDataArray{ASCIIString,Uint8,1}(73421) ["2", "2", "2", "2"]
  lectage: PooledDataArray{ASCIIString,Uint8,1}(73421) ["2", "1", "2", "2"]
  service: PooledDataArray{ASCIIString,Uint8,1}(73421) ["0", "1", "0", "1"]
  dept: PooledDataArray{ASCIIString,Uint8,1}(73421) ["2", "6", "2", "3"]
  y: DataArray{Int32,1}(73421) [5, 2, 5, 3]

julia> @time fm2 = fit(lmer(:(y ~ dept*service + (1|s) + (1|d)), inst), true)
f_1: 241920.83782176484, [1.0,1.0]
f_2: 244850.35312911027, [1.75,1.0]
f_3: 242983.2665867725, [1.0,1.75]
f_4: 238454.23551272828, [0.25,1.0]
f_5: 241716.05373818372, [1.0,0.25]
f_6: 240026.05964215088, [0.0,0.4459187720602683]
f_7: 241378.58264793456, [0.0,1.2795084971874737]
f_8: 238463.85416880148, [0.3469544534770044,0.9569903039888051]
f_9: 238337.43511332024, [0.25114954419507396,0.9334148476637315]
f_10: 238450.00007949764, [0.1951786674500835,0.8718977150010241]
f_11: 238282.78310872405, [0.2687773664115054,0.9136824143339727]
f_12: 238250.8765641119, [0.2807247154940574,0.8970615428366485]
f_13: 238207.46300521455, [0.3036961805316902,0.8631752348089436]
f_14: 238195.80295309465, [0.31902877663556645,0.8426407031707223]
f_15: 238196.40553488498, [0.3241232917043242,0.8371365305522264]
f_16: 238179.5371083254, [0.31724946862832715,0.8353548220973466]
f_17: 238160.8528644365, [0.31209440956481177,0.8299073120191853]
f_18: 238126.5333151576, [0.3030157611098546,0.817966703063554]
f_19: 238066.4496586597, [0.28962391700452816,0.7911216195014075]
f_20: 237960.37092757496, [0.27529159855792373,0.7328585553188129]
f_21: 237771.6569062529, [0.25940641978995127,0.6139146143402596]
f_22: 237651.67830076427, [0.24270382666956553,0.37449652107441056]
f_23: 240177.28896887854, [0.09414815638115845,0.18599927222137802]
f_24: 237591.38115376089, [0.2830510641605103,0.41035893867971385]
f_25: 237898.58451781687, [0.3305027523447658,0.3001394427252132]
f_26: 237625.00815014305, [0.3123657249746394,0.4094281613913877]
f_27: 237599.45772830598, [0.286273306583993,0.4702723524846733]
f_28: 237599.86006759468, [0.28779623297893586,0.3993369890842638]
f_29: 237586.3576741126, [0.2812673241106126,0.4330171598403483]
f_30: 237586.23506658926, [0.2808546407439123,0.4323909071446664]
f_31: 237586.04586549703, [0.2801074465726599,0.43245572126694304]
f_32: 237585.75792242363, [0.2786085768412473,0.4323975016117133]
f_33: 237585.5570866437, [0.2757877948048254,0.4313761355152525]
f_34: 237585.58865950265, [0.27557472780684533,0.4300519775020271]
f_35: 237585.59987905066, [0.2746377502964964,0.43206620045312394]
f_36: 237585.55356829558, [0.275934855672295,0.4321237270740967]
f_37: 237585.5601067908, [0.27585484291324425,0.43286944683271183]
f_38: 237585.55341702836, [0.27592069761507937,0.43200462733824807]
f_39: 237585.55359755812, [0.27599516940902075,0.4319957418496767]
f_40: 237585.5534506403, [0.27590761293575283,0.43193077755038833]
f_41: 237585.55355291063, [0.2758457838427002,0.43200822271137496]
f_42: 237585.55341516968, [0.2759147806823201,0.43199406219248815]
f_43: 237585.5534158112, [0.27591552268322755,0.4320026906010699]
f_44: 237585.5534151916, [0.27591578046410803,0.431994083082117]
f_45: 237585.55341520355, [0.2759138011000847,0.43199426323637397]
f_46: 237585.55341517975, [0.27591466813986054,0.43199306854557157]
f_47: 237585.55341516944, [0.27591489370984806,0.4319941073765378]
XTOL_REACHED
elapsed time: 11.839522936 seconds
Linear mixed model fit by maximum likelihood
 logLik: -118792.77670758472, deviance: 237585.55341516944

  Variance components:
    Std. deviation scale:[0.32468135128805153,0.508346717517133,1.176744563957777]
    Variance scale:[0.10541797987423512,0.2584163852104438,1.3847277688041786]
  Number of obs: 73421; levels of grouping factors:[2972,1128]

  Fixed-effects parameters:
          Estimate Std.Error   z value
[1,]       3.22961  0.064053   50.4209
[2,]      0.129536  0.101294   1.27882
[3,]     -0.176751 0.0881352  -2.00545
[4,]     0.0517102 0.0817524  0.632522
[5,]     0.0347319  0.085621  0.405647
[6,]       0.14594 0.0997984   1.46235
[7,]      0.151689 0.0816897   1.85689
[8,]      0.104206  0.118751  0.877517
[9,]     0.0440401 0.0962985  0.457329
[10,]    0.0517546 0.0986029  0.524879
[11,]    0.0466719  0.101942  0.457828
[12,]    0.0563461 0.0977925   0.57618
[13,]    0.0596536  0.100233   0.59515
[14,]    0.0055628  0.110867 0.0501756
[15,]     0.252025 0.0686507   3.67112
[16,]    -0.180757  0.123179  -1.46744
[17,]    0.0186492  0.110017  0.169512
[18,]    -0.282269 0.0792937  -3.55979
[19,]    -0.494464 0.0790278  -6.25683
[20,]    -0.392054  0.110313  -3.55403
[21,]    -0.278547 0.0823727  -3.38154
[22,]    -0.189526  0.111449  -1.70056
[23,]    -0.499868 0.0885423  -5.64553
[24,]    -0.497162 0.0917162  -5.42065
[25,]     -0.24042 0.0982071   -2.4481
[26,]    -0.223013 0.0890548  -2.50422
[27,]    -0.516997 0.0809077  -6.38997
[28,]    -0.384773  0.091843  -4.18946
```

Models with vector-valued random effects can be fit
```julia
julia> sleep = data("lme4","sleepstudy");

julia> dump(sleep)
DataFrame  180 observations of 3 variables
  Reaction: DataArray{Float64,1}(180) [249.56, 258.705, 250.801, 321.44]
  Days: DataArray{Float64,1}(180) [0.0, 1.0, 2.0, 3.0]
  Subject: PooledDataArray{ASCIIString,Uint8,1}(180) ["308", "308", "308", "308"]

julia> fm3 = fit(lmer(:(Reaction ~ Days + (Days|Subject)), sleep))
Linear mixed model fit by maximum likelihood
 logLik: -875.9696722444951, deviance: 1751.9393444889902

  Variance components:
    Std. deviation scale:[23.77975960241988,5.716798514149688,25.591907035097925]
    Variance scale:[565.4769667488805,32.68178525138408,654.9457056930947]
    Correlations:
{
2x2 Float64 Array:
 1.0        0.0813211
 0.0813211  1.0      }
  Number of obs: 180; levels of grouping factors:[18]

  Fixed-effects parameters:
        Estimate Std.Error z value
[1,]     251.405   6.63212 37.9072
[2,]     10.4673   1.50223 6.96783
```

## ToDo

Well, obviously I need to incorporate names for the fixed-effects
coefficients and create a coefficient table.

Special cases can be tuned up.  Much more calculation is being done in
the fit for models with a single grouping factor, models with scalar
random-effects terms only, models with strictly nested grouping
factors and models with crossed or nearly crossed grouping factors.

Also, the results of at least `X'X` and `X'y` should be cached for
cases where weights aren't changing.

Incorporating offsets and weights will be important for GLMMs.

Lots of work to be done.
