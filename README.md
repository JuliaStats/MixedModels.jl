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

## A substantial example

Natually the `Dyestuff` example does not show the power of the Julia
version.  I fit a model to a data set with 2,552,112 observations and
a total of 1,760,188 random effects for which the regular sparse
compressed column model matrix has the form

```julia
julia> mm.ZXt
1760188 by 2552112 regular sparse column matrix
Row indices: 3x2552112 Int32 Array:
  655840   680816   657696   657697   657698  …   125238   527019   125137   125496   862549
 1535708  1560684  1537564  1537565  1537566     1005106  1406887  1005005  1005364  1742417
 1759786  1759797  1759797  1759797  1759797     1760158  1760158  1760158  1760158  1760158
Non-zero values: 4x2552112 Float64 Array:
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  …  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0     1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0     1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0     1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
```

It is not obvious here but those grouping factors are partially
crossed.  The system matrix is 

```julia
julia> mm.A

CHOLMOD sparse:  :  1760188-by-1760188, nz 6318981, upper.
  nzmax 6318981, sorted, packed, 
  scalar types: int, real, double
  col 0: nz 1 start 0 end 1:
         0: 6.5948
  col 1: nz 1 start 1 end 2:
         1: 5.4758
  col 2: nz 1 start 2 end 3:
         2: 6.5948
  col 3: nz 1 start 3 end 4:
         3: 6.5948
    ...
  col 1760187: nz 1760188 start 4558793 end 6318981:
         0: 5.289
         1: 4.2312
         2: 5.289
    ...
   1760184: 27.324
   1760185: 105.06
   1760186: 44.258
   1760187: 2.5521e+06
  nnz on diagonal: 1760188
  OK
```

and the sparse Cholesky factor is

```julia
julia> mm.L

CHOLMOD factor:  :  1760188-by-1760188
  scalar types: int, real, double
  simplicial, LDL'.
  ordering method used: AMD
         0:3389
         1:883257
         2:5344
         3:885212
         4:5828
         5:885696
         6:8748
         7:888616
    ...
   1760184:1760136
   1760185:1760139
   1760186:1760140
   1760187:1760187
  col: 0 colcount: 4
  col: 1 colcount: 3
  col: 2 colcount: 4
  col: 3 colcount: 3
  col: 4 colcount: 4
  col: 5 colcount: 3
  col: 6 colcount: 4
  col: 7 colcount: 3
    ...
  col: 1760184 colcount: 4
  col: 1760185 colcount: 3
  col: 1760186 colcount: 2
  col: 1760187 colcount: 1
monotonic: 1
 nzmax 6372007.
  col 0: nz 4 start 0 end 4 space 4 free 0:
         0: 6.5948
         1: 0.95058
   1760186: 0.30865
   1760187: 0.802
  col 1: nz 3 start 4 end 7 space 3 free 0:
         1: 2.0651
   1760186: 0.16747
   1760187: 0.43515
  col 2: nz 4 start 7 end 11 space 4 free 0:
         2: 3.2379
    ...
  col 1760185: nz 3 start 6372001 end 6372004 space 3 free 0:
   1760185: 681.42
   1760186: -0.0059217
   1760187: 2.5736
  col 1760186: nz 2 start 6372004 end 6372006 space 2 free 0:
   1760186: 463.94
   1760187: 2.5868
  col 1760187: nz 1 start 6372006 end 6372007 space 1 free 0:
   1760187: 2962
  nz 6372007  OK
```

The fact that the number of nonzeros in `L`, 6,372,007, is larger than
the number of nonzeros in `A`, 6,318,981, indicates that the grouping
factors are not nested.

I can't identify these data other than to say that they are derived
from annual test scores of students in one of the United States.

I have fit such a model in `R` using the `lme4` package.  In fact this
model is my stress test when comparing methods in `lme4`.  Generally
each evaluation of the deviance requires 20 seconds or more in various
incarnations of `lme4`.  On the same computer the Julia code performs
an evaluation in about 1.4 seconds.  We had difficulty with several of
the optimizers in R and eventually resorted to using a Nelder-Mead
simplex method that allows for box constraints.  In Julia I can use
Steve Johnson's implementation of a derivative-free optimizer BOBYQA.
The Nelder-Mead version took over 300 evaluations to converge on this
3-dimensional optimization problem in R.  The total time to fit such a
model was 7000 seconds, during which time R used up all the memory (8
GB) on this computer resulting in swap thrashing.  Essentially
response on all other processes died.  The Julia version converged in
73 iterations while using less than 1 GB of memory. Here's the
trace

```julia
julia> @elapsed fit(mm, true)
f_1: 2.64313337640267e7, [1.0, 1.0, 1.0]
f_2: 2.646530790370047e7, [1.75, 1.0, 1.0]
f_3: 2.646530790370086e7, [1.0, 1.75, 1.0]
f_4: 2.6431792205251433e7, [1.0, 1.0, 1.75]
f_5: 2.6548597211884834e7, [0.25, 1.0, 1.0]
f_6: 2.6548597211894397e7, [1.0, 0.25, 1.0]
f_7: 2.6431054281238183e7, [1.0, 1.0, 0.25]
f_8: 2.6518651372547925e7, [1.20652, 1.20652, 0.0]
f_9: 2.6542694444043886e7, [0.736752, 0.736752, 0.204994]
f_10: 2.642547553170733e7, [1.075, 1.0, 0.325]
f_11: 2.6424596212055616e7, [0.972194, 1.11541, 0.431144]
f_12: 2.6428614541411452e7, [0.974207, 1.05276, 0.472326]
f_13: 2.6425128228043392e7, [0.911593, 1.15648, 0.447445]
f_14: 2.6421780805168476e7, [1.01277, 1.15118, 0.483095]
f_15: 2.6420943237139884e7, [0.998956, 1.21356, 0.522377]
f_16: 2.6420981871568996e7, [0.974296, 1.27818, 0.551386]
f_17: 2.6420893374648757e7, [0.999481, 1.2457, 0.541684]
f_18: 2.642087219743117e7, [0.984028, 1.24682, 0.541771]
f_19: 2.6420862702375993e7, [0.991782, 1.23884, 0.528627]
f_20: 2.6420865891771503e7, [0.991001, 1.24125, 0.533825]
f_21: 2.6420856469631426e7, [0.9946, 1.23842, 0.52169]
f_22: 2.6420855632944528e7, [0.997869, 1.23282, 0.517915]
f_23: 2.6420874723336898e7, [0.993077, 1.22858, 0.514001]
f_24: 2.6420857780514594e7, [0.997125, 1.23694, 0.523435]
f_25: 2.6420856570992395e7, [0.996609, 1.23445, 0.520053]
f_26: 2.642085863926388e7, [0.997971, 1.231, 0.518337]
f_27: 2.6420853220310677e7, [0.998583, 1.23342, 0.516285]
f_28: 2.642085102338043e7, [0.999941, 1.23206, 0.513063]
f_29: 2.642084869252778e7, [1.00273, 1.22817, 0.507289]
f_30: 2.6420843818743285e7, [1.00541, 1.22602, 0.500623]
f_31: 2.6420833254012857e7, [1.01086, 1.22246, 0.487108]
f_32: 2.6420819186332628e7, [1.02165, 1.21192, 0.461179]
f_33: 2.6420802290913634e7, [1.03131, 1.20618, 0.433363]
f_34: 2.642079308871887e7, [1.05406, 1.18449, 0.382254]
f_35: 2.642090466267342e7, [1.0543, 1.16553, 0.325329]
f_36: 2.642083001091622e7, [1.05934, 1.19926, 0.378647]
f_37: 2.6420796779363126e7, [1.06168, 1.18198, 0.353348]
f_38: 2.6420793218270306e7, [1.06833, 1.18004, 0.383511]
f_39: 2.6420791444902476e7, [1.06432, 1.18058, 0.372029]
f_40: 2.6420800189956754e7, [1.0717, 1.18002, 0.370817]
f_41: 2.6420793098414853e7, [1.06298, 1.18459, 0.378225]
f_42: 2.642079133428025e7, [1.06481, 1.17872, 0.371367]
f_43: 2.642079078809407e7, [1.06464, 1.17916, 0.375086]
f_44: 2.6420790281789716e7, [1.06314, 1.18088, 0.382232]
f_45: 2.6420790275522258e7, [1.05802, 1.18567, 0.384891]
f_46: 2.6420790256668385e7, [1.059, 1.18417, 0.387537]
f_47: 2.6420790222395737e7, [1.05904, 1.18421, 0.383787]
f_48: 2.6420790322209913e7, [1.06045, 1.18287, 0.380583]
f_49: 2.64207903720455e7, [1.05999, 1.18426, 0.384139]
f_50: 2.6420790215797003e7, [1.05801, 1.18509, 0.385087]
f_51: 2.6420790230582e7, [1.05822, 1.18485, 0.38324]
f_52: 2.6420790237310737e7, [1.05757, 1.18514, 0.385694]
f_53: 2.6420790219062142e7, [1.05763, 1.18542, 0.385645]
f_54: 2.6420790216793273e7, [1.05806, 1.18509, 0.385205]
f_55: 2.642079021680278e7, [1.05791, 1.18508, 0.385013]
f_56: 2.6420790215588965e7, [1.05793, 1.18516, 0.385025]
f_57: 2.6420790215490576e7, [1.05783, 1.18523, 0.384974]
f_58: 2.642079021538983e7, [1.05776, 1.1853, 0.384892]
f_59: 2.6420790215423398e7, [1.05771, 1.18535, 0.384784]
f_60: 2.6420790215511013e7, [1.05769, 1.18533, 0.384914]
f_61: 2.6420790215370934e7, [1.05781, 1.18526, 0.384861]
f_62: 2.6420790215388946e7, [1.05793, 1.18516, 0.384843]
f_63: 2.642079021541716e7, [1.05783, 1.18524, 0.384791]
f_64: 2.6420790215375423e7, [1.05783, 1.18524, 0.384871]
f_65: 2.6420790215476643e7, [1.05782, 1.18527, 0.384863]
f_66: 2.6420790215396754e7, [1.0578, 1.18527, 0.384867]
f_67: 2.64207902153635e7, [1.05781, 1.18526, 0.384853]
f_68: 2.642079021538987e7, [1.05781, 1.18526, 0.384844]
f_69: 2.6420790215382434e7, [1.05781, 1.18526, 0.384852]
f_70: 2.6420790215373155e7, [1.05781, 1.18526, 0.384852]
f_71: 2.6420790215365566e7, [1.05781, 1.18526, 0.384853]
f_72: 2.6420790215393037e7, [1.05781, 1.18526, 0.384854]
f_73: 2.6420790215388436e7, [1.05781, 1.18526, 0.384852]
SUCCESS
98.352053755
```

In Julia it took less than 100 seconds and did not freeze the computer
due to swapping.

Overall, a very satisfactory first cut at mixed-models fitting in
Julia.

## ToDo

Lots. Extension to models with vector-valued random effects,
GLMMs, NLMMs, etc.  But this is a good start!
