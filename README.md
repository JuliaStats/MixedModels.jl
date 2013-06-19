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
julia> m = lmer(:(Yield ~ 1|Batch),ds)
Linear mixed model fit by maximum likelihood
 logLik: -163.6635299406109, deviance: 327.3270598812218

  Variance components: [1x1 Float64 Array:
 1388.34, 1x1 Float64 Array:
 2451.25]
  Number of obs: 30; levels of grouping factors: [6]
  Fixed-effects parameters: [1527.5]
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
julia> VarCorr(m)
2-element Array{Float64,2} Array:
 1x1 Float64 Array:
 1388.34
 1x1 Float64 Array:
 2451.25

julia> fixef(m)
1-element Float64 Array:
 1527.5

julia> ranef(m)
1-element Array{Float64,2} Array:
 1x6 Float64 Array:
 -16.6283  0.369517  26.9747  -21.8015  53.5799  -42.4944

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

julia> @time fm2 = fit(lmer(:(y ~ dept*service + (1|s) + (1|d)), inst),true)
f_1: 241920.83782176487, [1.0, 1.0]
f_2: 244850.35312911033, [1.75, 1.0]
f_3: 242983.2665867725, [1.0, 1.75]
f_4: 238454.23551272828, [0.25, 1.0]
f_5: 241716.05373818372, [1.0, 0.25]
f_6: 240026.0596421509, [0.0, 0.445919]
f_7: 241378.58264793456, [0.0, 1.27951]
f_8: 238463.85416880157, [0.346954, 0.95699]
f_9: 238337.4351133204, [0.25115, 0.933415]
f_10: 238450.00007949758, [0.195179, 0.871898]
f_11: 238282.78310872425, [0.268777, 0.913682]
f_12: 238250.876564112, [0.280725, 0.897062]
f_13: 238207.46300521464, [0.303696, 0.863175]
f_14: 238195.80295309465, [0.319029, 0.842641]
f_15: 238196.40553488504, [0.324123, 0.837137]
f_16: 238179.53710832534, [0.317249, 0.835355]
f_17: 238160.8528644366, [0.312094, 0.829907]
f_18: 238126.53331515726, [0.303016, 0.817967]
f_19: 238066.44965865996, [0.289624, 0.791122]
f_20: 237960.3709275751, [0.275292, 0.732859]
f_21: 237771.6569062523, [0.259406, 0.613915]
f_22: 237651.6783007646, [0.242704, 0.374497]
f_23: 240177.2889688969, [0.0941482, 0.185999]
f_24: 237591.38115376432, [0.283051, 0.410359]
f_25: 237898.58451783203, [0.330503, 0.300139]
f_26: 237625.00815015173, [0.312366, 0.409428]
f_27: 237599.4577282987, [0.286273, 0.470272]
f_28: 237599.86006760105, [0.287796, 0.399337]
f_29: 237586.3576741112, [0.281267, 0.433017]
f_30: 237586.23506658932, [0.280855, 0.432391]
f_31: 237586.0458654961, [0.280107, 0.432456]
f_32: 237585.75792242354, [0.278609, 0.432398]
f_33: 237585.55708664245, [0.275788, 0.431376]
f_34: 237585.58865949293, [0.275575, 0.430052]
f_35: 237585.59987904236, [0.274638, 0.432066]
f_36: 237585.55356829462, [0.275935, 0.432124]
f_37: 237585.56010679065, [0.275855, 0.432869]
f_38: 237585.55341702755, [0.275921, 0.432005]
f_39: 237585.5535975567, [0.275995, 0.431996]
f_40: 237585.55345063927, [0.275908, 0.431931]
f_41: 237585.55355290952, [0.275846, 0.432008]
f_42: 237585.55341517035, [0.275915, 0.431994]
f_43: 237585.5534158103, [0.275916, 0.432003]
f_44: 237585.5534151921, [0.275916, 0.431994]
f_45: 237585.55341520358, [0.275914, 0.431994]
f_46: 237585.5534151795, [0.275915, 0.431993]
f_47: 237585.55341517017, [0.275915, 0.431994]
XTOL_REACHED
elapsed time: 12.078582378 seconds
Linear mixed model fit by maximum likelihood
 logLik: -118792.77670758509, deviance: 237585.55341517017

  Variance components: [1x1 Float64 Array:
 0.105418, 1x1 Float64 Array:
 0.258416, 1x1 Float64 Array:
 1.38473]
  Number of obs: 73421; levels of grouping factors: [2972, 1128]
  Fixed-effects parameters: [3.22961, 0.129536, -0.176751, 0.0517102, 0.0347319  …  -0.24042, -0.223013, -0.516997, -0.384773]
```

Models with vector-valued random effects can be fit
```julia
julia> sleep = data("lme4","sleepstudy");

julia> dump(sleep)
DataFrame  180 observations of 3 variables
  Reaction: DataArray{Float64,1}(180) [249.56, 258.705, 250.801, 321.44]
  Days: DataArray{Float64,1}(180) [0.0, 1.0, 2.0, 3.0]
  Subject: PooledDataArray{ASCIIString,Uint8,1}(180) ["308", "308", "308", "308"]

julia> fm3 = fit(lmer(:(Reaction ~ Days + (Days|Subject)), sleep), true)
f_1: 1784.6422961924472, [1.0, 0.0, 1.0]
f_2: 1790.1256369894443, [1.75, 0.0, 1.0]
f_3: 1798.9996244965714, [1.0, 1.0, 1.0]
f_4: 1803.8532002843879, [1.0, 0.0, 1.75]
f_5: 1800.6139807455356, [0.25, 0.0, 1.0]
f_6: 1798.6046308389216, [1.0, -1.0, 1.0]
f_7: 1752.2607369909074, [1.0, 0.0, 0.25]
f_8: 1797.587692019767, [1.18326, -0.00866189, 0.0]
f_9: 1754.9541095798577, [1.075, 0.0, 0.325]
f_10: 1753.6956816567858, [0.816632, 0.0111673, 0.288238]
f_11: 1754.8169985163663, [1.0, -0.0707107, 0.196967]
f_12: 1753.1067335474627, [0.943683, 0.0638354, 0.262696]
f_13: 1752.9393767190031, [0.980142, -0.0266568, 0.274743]
f_14: 1752.2568790228843, [0.984343, -0.0132347, 0.247191]
f_15: 1752.057448057069, [0.97314, 0.00253785, 0.23791]
f_16: 1752.0223889106287, [0.954526, 0.00386421, 0.235892]
f_17: 1752.0227280154613, [0.935929, 0.0013318, 0.234445]
f_18: 1751.9716865442101, [0.954965, 0.00790664, 0.229046]
f_19: 1751.9526031249716, [0.953313, 0.0166274, 0.225768]
f_20: 1751.948524268375, [0.946929, 0.0130761, 0.222871]
f_21: 1751.9871762805533, [0.933418, 0.00613767, 0.218951]
f_22: 1751.9832131209412, [0.951544, 0.005789, 0.220618]
f_23: 1751.951971276329, [0.952809, 0.0190332, 0.224178]
f_24: 1751.9462759266778, [0.946322, 0.0153739, 0.225088]
f_25: 1751.9466979622896, [0.947124, 0.0148894, 0.224892]
f_26: 1751.9475678612077, [0.946497, 0.0154643, 0.225814]
f_27: 1751.9453118381468, [0.946086, 0.0157934, 0.224449]
f_28: 1751.944179614325, [0.945304, 0.0166902, 0.223361]
f_29: 1751.943532886748, [0.944072, 0.0172106, 0.222716]
f_30: 1751.942440721589, [0.941271, 0.0163099, 0.222523]
f_31: 1751.9421698853885, [0.939, 0.015899, 0.222132]
f_32: 1751.942370107883, [0.938979, 0.016548, 0.221562]
f_33: 1751.942278038317, [0.938863, 0.0152466, 0.222683]
f_34: 1751.942203968235, [0.938269, 0.015733, 0.222024]
f_35: 1751.9413093339792, [0.938839, 0.0166373, 0.222611]
f_36: 1751.9409310154442, [0.938397, 0.0173965, 0.222817]
f_37: 1751.9405672034432, [0.937006, 0.0180445, 0.222534]
f_38: 1751.9401790031193, [0.934109, 0.0187354, 0.22195]
f_39: 1751.940081850155, [0.932642, 0.0189242, 0.221726]
f_40: 1751.9402696242064, [0.931357, 0.0190082, 0.221309]
f_41: 1751.9415006286824, [0.932821, 0.0206454, 0.221367]
f_42: 1751.9394886779307, [0.931867, 0.0179574, 0.222564]
f_43: 1751.9393923344423, [0.929167, 0.0177824, 0.222534]
f_44: 1751.939397613947, [0.929659, 0.0177721, 0.222508]
f_45: 1751.939425025923, [0.929193, 0.0187806, 0.22257]
f_46: 1751.9393548581595, [0.928986, 0.0182366, 0.222484]
f_47: 1751.9394897720788, [0.928697, 0.0182937, 0.223175]
f_48: 1751.9393630911852, [0.928243, 0.0182695, 0.222584]
f_49: 1751.939344829649, [0.929113, 0.0181791, 0.222624]
f_50: 1751.9393444889902, [0.929191, 0.0181658, 0.222643]
FTOL_REACHED
Linear mixed model fit by maximum likelihood
 logLik: -875.9696722444951, deviance: 1751.9393444889902

  Variance components: [2x2 Float64 Array:
 565.477   11.0551
  11.0551  32.6818, 1x1 Float64 Array:
 654.946]
  Number of obs: 180; levels of grouping factors: [18]
  Fixed-effects parameters: [251.405, 10.4673]
```

## ToDo

Well, obviously I need to incorporate names for the fixed-effects
coefficients and create a coefficient table.  Also there should be a
type for the value of `VarCorr` with its own `show` method.

Special cases can be tuned up.  Much more calculation is being done in
the fit for models with a single grouping factor, models with scalar
random-effects terms only, models with strictly nested grouping
factors and models with crossed or nearly crossed grouping factors.

Also, the results of at least `X'X` and `X'y` should be cached for
cases where weights aren't changing.

Incorporating offsets and weights will be important for GLMMs.

Lots of work to be done.
