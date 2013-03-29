# Linear mixed-effects models in [Julia](http://julialang.org)

## Installation

This package requires Steve Johnson's
[NLopt](https://github.com/stevengj/NLopt.jl.git) package.  If this
package is not already installed, check it's installation
instructions.  It requires you to have installed the `nlopt` library.
Once the `NLopt` package is installed,

```julia
Pkg.add("MixedModels")
```

will install this package.

## Fitting simple mixed-effects models

At present the `MixedModels` package supports only _simple linear
mixed-effects models_, in which all the random-effects terms are
simple, scalar terms.  That is, no vector valued random effects or
interactions between random factors and fixed-effects terms are
allowed.  In the formula notation used in the
[lme4](https://github.com/lme4/lme4) package for
[R](http://r-project.org) this means that all random-effects terms are
of the form ```(1|g)```.

Such a model is represented by a `LMMsimple` object that is
created from the indices for the random effects grouping factors, the
fixed-effects model matrix, and the response.

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

To obtain such a fit using Julia we attach the `Distributions` and
`MixedModels` packages

```julia
using Distributions, MixedModels
```

and read in the data.  We could use the `RDatasets` package to obtain
the data but it is small enough to create it inline

```julia
julia> Yield = [1545., 1440, 1440, 1520, 1580, 1540, 1555, 1490, 1560,
         1495, 1595, 1550, 1605, 1510, 1560, 1445, 1440, 1595,
         1465, 1545, 1595, 1630, 1515, 1635, 1625, 1520, 1455,
         1450, 1480, 1445];
```

To get the `Batch` factor we could use the `gl` function from Julia's
`DataFrames` package (patterned after R's `gl` function)

```
julia> using DataFrames
julia> println(gl(6,5))
[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]
```

or, we could reshape the result of a comprehension.  Comprehensions
are pretty neat and astonishingly fast so we will show that approach.

```
julia> indm = [j for i in 1:5, j in 1:6]
5x6 Int64 Array:
 1  2  3  4  5  6
 1  2  3  4  5  6
 1  2  3  4  5  6
 1  2  3  4  5  6
 1  2  3  4  5  6
julia> println(vec(m))
[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]
```

A comprehension generates arrays in a kind of inverted, nested `for`
loop.  As you can see the expression generating `indm` produces a 5 by
6 matrix with `indm[i,j] = j`.  Like `R` and `Matlab/Octave`, Julia
stores matrices in column-major order so the `vec` function applied to
a matrix concatenates the columns.  Currently `LMMsimple` expects
matrices with `n` rows for the indices and the fixed-effects model
matrix and we must construct the model as 

```julia
julia> n = length(Yield);
julia> fm = LMMsimple(reshape(m, (n,1)), ones(n,1), Yield); 
julia> fm.ZXt
7 by 30 regular sparse column matrix Row indices:
1x30 Int64 Array: 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5 6 6 6 6 6
Non-zero values: 2x30 Float64 Array:
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  …  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0     1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
```

The `fit` method optimizes the parameter estimates.  Setting the optional
optional `verbose` argument to `true` provides a trace of the optimization

```julia
julia> fit(fm, true)  # verbose fit
f_1: 327.7670216246145, [1.0]
f_2: 331.0361932224437, [1.75]
f_3: 330.6458314144857, [0.25]
f_4: 327.69511270610866, [0.97619]
f_5: 327.5663091453218, [0.928569]
f_6: 327.3825965130752, [0.833327]
f_7: 327.3531545408492, [0.807188]
f_8: 327.3466298241027, [0.799688]
f_9: 327.34100192001785, [0.792188]
f_10: 327.33252535370985, [0.777188]
f_11: 327.3273305611214, [0.747188]
f_12: 327.32861909776966, [0.739688]
f_13: 327.3270602360369, [0.752777]
f_14: 327.32706815453946, [0.753527]
f_15: 327.3270598812219, [0.752584]
FTOL_REACHED
true
```

Parameter estimates are provided by `fixef` and `VarCorr`

```julia
julia> println(fixef(fm))
[1527.5]
julia> println(VarCorr(fm))
1x2 Float64 Array:
 1388.34  2451.25
```

The conditional modes of the random effects are returned by `ranef`

```julia
julia> println(ranef(fm))
[-16.6283, 0.369517, 26.9747, -21.8015, 53.5799, -42.4944]
```

For a REML fit, set the REML field to `true`.  Whenever you change
properties of the model directly you should also set the `fit` field
to `false`

```julia
julia> fm.REML = true; fm.fit = false
false

julia> fit(fm, true)
f_1: 319.72580924655597, [0.752584]
f_2: 320.6230489677262, [1.31702]
f_3: 324.62180212210484, [0.188146]
f_4: 319.7164973657527, [0.947384]
f_5: 319.65462477204755, [0.855379]
f_6: 319.65880164730163, [0.874022]
f_7: 319.6543502846913, [0.8451]
f_8: 319.6542768539961, [0.848283]
f_9: 319.6542787718305, [0.848847]
f_10: 319.65427684226245, [0.848325]
FTOL_REACHED
true
```
