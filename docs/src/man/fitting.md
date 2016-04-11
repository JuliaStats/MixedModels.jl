# Fitting linear mixed-effects models

The `lmm` function is similar to the `lmer` function in the
[lme4](http://cran.R-project.org/package=lme4) package for
[R](http://www.R-project.org).  The first two arguments for in the `R`
version are `formula` and `data`.  The principle method for the
`Julia` version takes these arguments.

## A simple example

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

    {meta}
    DocTestSetup = quote
        using DataFrames, MixedModels
        include(Pkg.dir("MixedModels", "test", "data.jl"))
    end

These `Dyestuff` data are available through `RCall` but to run the doctests we use a stored copy of the dataframe.
```julia
julia> using DataFrames, MixedModels

julia> ds
30x2 DataFrames.DataFrame
│ Row │ Yield  │ Batch │
┝━━━━━┿━━━━━━━━┿━━━━━━━┥
│ 1   │ 1545.0 │ 'A'   │
│ 2   │ 1440.0 │ 'A'   │
│ 3   │ 1440.0 │ 'A'   │
│ 4   │ 1520.0 │ 'A'   │
│ 5   │ 1580.0 │ 'A'   │
│ 6   │ 1540.0 │ 'B'   │
│ 7   │ 1555.0 │ 'B'   │
│ 8   │ 1490.0 │ 'B'   │
⋮
│ 22  │ 1630.0 │ 'E'   │
│ 23  │ 1515.0 │ 'E'   │
│ 24  │ 1635.0 │ 'E'   │
│ 25  │ 1625.0 │ 'E'   │
│ 26  │ 1520.0 │ 'F'   │
│ 27  │ 1455.0 │ 'F'   │
│ 28  │ 1450.0 │ 'F'   │
│ 29  │ 1480.0 │ 'F'   │
│ 30  │ 1445.0 │ 'F'   │
```

`lmm` defaults to maximum likelihood estimation whereas `lmer` in `R`
defaults to REML estimation.

```julia
julia> m = fit!(lmm(Yield ~ 1 + (1 | Batch), ds))
Linear mixed model fit by maximum likelihood
 logLik: -163.663530, deviance: 327.327060, AIC: 333.327060, BIC: 337.530652

Variance components:
           Variance  Std.Dev.
 Batch    1388.3332 37.260344
 Residual 2451.2500 49.510100
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value
(Intercept)    1527.5   17.6946  86.326
```

In general the model should be fit through an explicit call to the `fit!`
function, which may take a second argument indicating a verbose fit.

```julia
julia> fit!(lmm(Yield ~ 1 + (1 | Batch), ds), true);
f_1: 327.76702, [1.0]
f_2: 331.03619, [1.75]
f_3: 330.64583, [0.25]
f_4: 327.69511, [0.97619]
f_5: 327.56631, [0.928569]
f_6: 327.3826, [0.833327]
f_7: 327.35315, [0.807188]
f_8: 327.34663, [0.799688]
f_9: 327.341, [0.792188]
f_10: 327.33253, [0.777188]
f_11: 327.32733, [0.747188]
f_12: 327.32862, [0.739688]
f_13: 327.32706, [0.752777]
f_14: 327.32707, [0.753527]
f_15: 327.32706, [0.752584]
f_16: 327.32706, [0.752509]
f_17: 327.32706, [0.752591]
f_18: 327.32706, [0.752581]
FTOL_REACHED
```

The numeric representation of the model has type
```julia
julia> typeof(fit!(lmm(Yield ~ 1 + (1 | Batch), ds)))
MixedModels.LinearMixedModel{Float64}
```
Those familiar with the `lme4` package for `R` will see the usual
suspects.
```julia
julia> m = fit!(lmm(Yield ~ 1 + (1 | Batch), ds));

julia> fixef(m)  # estimates of the fixed-effects parameters
1-element Array{Float64,1}:
 1527.5

julia> coef(m)  # another name for fixef
1-element Array{Float64,1}:
 1527.5

julia> ranef(m)
1-element Array{Array{Float64,2},1}:
 1x6 Array{Float64,2}:
 -16.6282  0.369516  26.9747  -21.8014  53.5798  -42.4943

julia> ranef(m, true)  # on the u scale
1-element Array{Array{Float64,2},1}:
 1x6 Array{Float64,2}:
 -22.0949  0.490999  35.8429  -28.9689  71.1948  -56.4648

julia> deviance(m)
327.3270598811394

julia> objective(m)
327.3270598811394
```

We prefer `objective` to `deviance` because the value returned is
`-2loglikelihood(m)`, without the correction for the null deviance.
It is not clear how the null deviance should be defined for these models.

## More substantial examples

Fitting a model to the `Dyestuff` data is trivial.  The `InstEval`
data in the `lme4` package is more of a challenge in that there are
nearly 75,000 evaluations by 2972 students on a total of 1128
instructors.

```
julia> head(inst)
6x7 DataFrames.DataFrame
│ Row │ s   │ d      │ studage │ lectage │ service │ dept │ y │
┝━━━━━┿━━━━━┿━━━━━━━━┿━━━━━━━━━┿━━━━━━━━━┿━━━━━━━━━┿━━━━━━┿━━━┥
│ 1   │ "1" │ "1002" │ "2"     │ "2"     │ "0"     │ "2"  │ 5 │
│ 2   │ "1" │ "1050" │ "2"     │ "1"     │ "1"     │ "6"  │ 2 │
│ 3   │ "1" │ "1582" │ "2"     │ "2"     │ "0"     │ "2"  │ 5 │
│ 4   │ "1" │ "2050" │ "2"     │ "2"     │ "1"     │ "3"  │ 3 │
│ 5   │ "2" │ "115"  │ "2"     │ "1"     │ "0"     │ "5"  │ 2 │
│ 6   │ "2" │ "756"  │ "2"     │ "1"     │ "0"     │ "5"  │ 4 │

julia> m2 = fit!(lmm(y ~ 1 + dept*service + (1|s) + (1|d), inst))
Linear mixed model fit by maximum likelihood
 logLik: -118792.776708, deviance: 237585.553415, AIC: 237647.553415, BIC: 237932.876339

Variance components:
            Variance   Std.Dev.
 s        0.105417971 0.32468134
 d        0.258416394 0.50834673
 Residual 1.384727771 1.17674456
 Number of obs: 73421; levels of grouping factors: 2972, 1128

  Fixed-effects parameters:
                           Estimate Std.Error   z value
(Intercept)                 3.22961  0.064053   50.4209
dept - 5                   0.129536  0.101294   1.27882
dept - 10                 -0.176751 0.0881352  -2.00545
dept - 12                 0.0517102 0.0817524  0.632522
dept - 6                  0.0347319  0.085621  0.405647
dept - 7                    0.14594 0.0997984   1.46235
dept - 4                   0.151689 0.0816897   1.85689
dept - 8                   0.104206  0.118751  0.877517
dept - 9                  0.0440401 0.0962985  0.457329
dept - 14                 0.0517546 0.0986029  0.524879
dept - 1                  0.0466719  0.101942  0.457828
dept - 3                  0.0563461 0.0977925   0.57618
dept - 11                 0.0596536  0.100233   0.59515
dept - 2                 0.00556281  0.110867 0.0501757
service - 1                0.252025 0.0686507   3.67112
dept - 5 & service - 1    -0.180757  0.123179  -1.46744
dept - 10 & service - 1   0.0186492  0.110017  0.169512
dept - 12 & service - 1   -0.282269 0.0792937  -3.55979
dept - 6 & service - 1    -0.494464 0.0790278  -6.25683
dept - 7 & service - 1    -0.392054  0.110313  -3.55403
dept - 4 & service - 1    -0.278547 0.0823727  -3.38154
dept - 8 & service - 1    -0.189526  0.111449  -1.70056
dept - 9 & service - 1    -0.499868 0.0885423  -5.64553
dept - 14 & service - 1   -0.497162 0.0917162  -5.42065
dept - 1 & service - 1     -0.24042 0.0982071   -2.4481
dept - 3 & service - 1    -0.223013 0.0890548  -2.50422
dept - 11 & service - 1   -0.516997 0.0809077  -6.38997
dept - 2 & service - 1    -0.384773  0.091843  -4.18946
```

Models with vector-valued random effects can be fit
```julia
julia> slp
180x3 DataFrames.DataFrame
│ Row │ Reaction │ Days │ Subject │
┝━━━━━┿━━━━━━━━━━┿━━━━━━┿━━━━━━━━━┥
│ 1   │ 249.56   │ 0    │ 1       │
│ 2   │ 258.705  │ 1    │ 1       │
│ 3   │ 250.801  │ 2    │ 1       │
│ 4   │ 321.44   │ 3    │ 1       │
│ 5   │ 356.852  │ 4    │ 1       │
│ 6   │ 414.69   │ 5    │ 1       │
│ 7   │ 382.204  │ 6    │ 1       │
│ 8   │ 290.149  │ 7    │ 1       │
⋮
│ 172 │ 273.474  │ 1    │ 18      │
│ 173 │ 297.597  │ 2    │ 18      │
│ 174 │ 310.632  │ 3    │ 18      │
│ 175 │ 287.173  │ 4    │ 18      │
│ 176 │ 329.608  │ 5    │ 18      │
│ 177 │ 334.482  │ 6    │ 18      │
│ 178 │ 343.22   │ 7    │ 18      │
│ 179 │ 369.142  │ 8    │ 18      │
│ 180 │ 364.124  │ 9    │ 18      │

julia> fm3 = fit!(lmm(Reaction ~ 1 + Days + (1+Days|Subject), slp))
Linear mixed model fit by maximum likelihood
 logLik: -875.969672, deviance: 1751.939344, AIC: 1763.939344, BIC: 1783.097086

Variance components:
           Variance  Std.Dev.   Corr.
 Subject  565.51066 23.780468
           32.68212  5.716828  0.08
 Residual 654.94145 25.591824
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
             Estimate Std.Error z value
(Intercept)   251.405   6.63226 37.9064
Days          10.4673   1.50224 6.96781
```
