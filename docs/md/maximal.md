---
title: Simulation of Subject-Item mixed models
author: Douglas Bates
date: 2019-02-21
---
[Note: The "terms2.0 - Son of Terms" release of the `StatsModels` package is imminent.
After that release and a corresponding release of `"MixedModels"` these calculations can be reproduced.
To reproduce the results in this document now requires the `"terms2.0"` branch of MixedModels and the `"dfk/terms2.0"` branch of StatsModels.]

Following the publication of Barr et al. (2011) there has been considerable interest in simulation of subject-item types of data from mixed-effects models to assess the effect of the choice of random-effects structure on the Type I error of tests on the fixed-effects.
Here we show how such simulations can be carried out efficiently using the [`MixedModels`](https://github.com/dmbates/MixedModels.jl) package for [`Julia`](https://julialang.org).

## Data characteristics

The data characteristics for this simulation are those from the paper _Maybe maximal: Good enough mixed models optimize power while controlling Type I error_ by Seedorff, Oleson and McMurray, which is just one example of such a study.
There are 50 subjects, 25 from each of two age groups, which are denoted by `'O'` (older) and `'Y'` (younger).
Each subject's response is measured on 5 different occasions on each of 20 different items under two noise conditions, `'Q'` (quiet) and `'N'` (noisy).
Such an experimental design yields a total of 10,000 measurements.

In the data for this experimental design, the 25 younger subjects are labelled `'a'` to `'y'` while the older subjects are `'A'` to `'Y'` and the items are `'A'` to `'T'`.
````julia
using DataFrames, FreqTables, MixedModels, Random, StatsModels, Tables
df = (S = repeat(['A':'Y'; 'a':'y'], inner=40, outer=5),
    Age = repeat(['O','Y'], inner=1000, outer=5),
    I = repeat('A':'T', inner=2, outer=250),
    Noise = repeat(['Q','N'], outer=5000),
    Y = ones(10000))
````




The response column, `Y`, is added as a placeholder.

#### ColumnTable versus DataFrame

`df` is a `NamedTuple`, which is similar to a `list` in `R` except that the names are `Symbol`s, not `String`s,
````julia
typeof(df)
````


````
NamedTuple{(:S, :Age, :I, :Noise, :Y),Tuple{Array{Char,1},Array{Char,1},Arr
ay{Char,1},Array{Char,1},Array{Float64,1}}}
````




It is easily converted to a `DataFrame` if desired.
````julia
DataFrame(df)
````



<table class="data-frame"><thead><tr><th></th><th>S</th><th>Age</th><th>I</th><th>Noise</th><th>Y</th></tr><tr><th></th><th>Char</th><th>Char</th><th>Char</th><th>Char</th><th>Float64</th></tr></thead><tbody><p>10,000 rows × 5 columns</p><tr><th>1</th><td>'A'</td><td>'O'</td><td>'A'</td><td>'Q'</td><td>1.0</td></tr><tr><th>2</th><td>'A'</td><td>'O'</td><td>'A'</td><td>'N'</td><td>1.0</td></tr><tr><th>3</th><td>'A'</td><td>'O'</td><td>'B'</td><td>'Q'</td><td>1.0</td></tr><tr><th>4</th><td>'A'</td><td>'O'</td><td>'B'</td><td>'N'</td><td>1.0</td></tr><tr><th>5</th><td>'A'</td><td>'O'</td><td>'C'</td><td>'Q'</td><td>1.0</td></tr><tr><th>6</th><td>'A'</td><td>'O'</td><td>'C'</td><td>'N'</td><td>1.0</td></tr><tr><th>7</th><td>'A'</td><td>'O'</td><td>'D'</td><td>'Q'</td><td>1.0</td></tr><tr><th>8</th><td>'A'</td><td>'O'</td><td>'D'</td><td>'N'</td><td>1.0</td></tr><tr><th>9</th><td>'A'</td><td>'O'</td><td>'E'</td><td>'Q'</td><td>1.0</td></tr><tr><th>10</th><td>'A'</td><td>'O'</td><td>'E'</td><td>'N'</td><td>1.0</td></tr><tr><th>11</th><td>'A'</td><td>'O'</td><td>'F'</td><td>'Q'</td><td>1.0</td></tr><tr><th>12</th><td>'A'</td><td>'O'</td><td>'F'</td><td>'N'</td><td>1.0</td></tr><tr><th>13</th><td>'A'</td><td>'O'</td><td>'G'</td><td>'Q'</td><td>1.0</td></tr><tr><th>14</th><td>'A'</td><td>'O'</td><td>'G'</td><td>'N'</td><td>1.0</td></tr><tr><th>15</th><td>'A'</td><td>'O'</td><td>'H'</td><td>'Q'</td><td>1.0</td></tr><tr><th>16</th><td>'A'</td><td>'O'</td><td>'H'</td><td>'N'</td><td>1.0</td></tr><tr><th>17</th><td>'A'</td><td>'O'</td><td>'I'</td><td>'Q'</td><td>1.0</td></tr><tr><th>18</th><td>'A'</td><td>'O'</td><td>'I'</td><td>'N'</td><td>1.0</td></tr><tr><th>19</th><td>'A'</td><td>'O'</td><td>'J'</td><td>'Q'</td><td>1.0</td></tr><tr><th>20</th><td>'A'</td><td>'O'</td><td>'J'</td><td>'N'</td><td>1.0</td></tr><tr><th>21</th><td>'A'</td><td>'O'</td><td>'K'</td><td>'Q'</td><td>1.0</td></tr><tr><th>22</th><td>'A'</td><td>'O'</td><td>'K'</td><td>'N'</td><td>1.0</td></tr><tr><th>23</th><td>'A'</td><td>'O'</td><td>'L'</td><td>'Q'</td><td>1.0</td></tr><tr><th>24</th><td>'A'</td><td>'O'</td><td>'L'</td><td>'N'</td><td>1.0</td></tr><tr><th>&vellip;</th><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td></tr></tbody></table>



The trend in Julia packages supporting data science, like the `StatsModels` package, is towards data representations as "column tables" (a `NamedTuple` of arrays) or "row tables" (a vector of `NamedTuple`s).
Sometimes it is convenient to work on individual columns, sometimes it makes more sense to iterate over rows.
The `columntable` and `rowtable` functions allow for conversion back and forth between the two representations.

````julia
rowtable(df)
````


````
10000-element Array{NamedTuple{(:S, :Age, :I, :Noise, :Y),Tuple{Char,Char,C
har,Char,Float64}},1}:
 (S = 'A', Age = 'O', I = 'A', Noise = 'Q', Y = 1.0)
 (S = 'A', Age = 'O', I = 'A', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'B', Noise = 'Q', Y = 1.0)
 (S = 'A', Age = 'O', I = 'B', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'C', Noise = 'Q', Y = 1.0)
 (S = 'A', Age = 'O', I = 'C', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'D', Noise = 'Q', Y = 1.0)
 (S = 'A', Age = 'O', I = 'D', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'E', Noise = 'Q', Y = 1.0)
 (S = 'A', Age = 'O', I = 'E', Noise = 'N', Y = 1.0)
 ⋮                                                  
 (S = 'y', Age = 'Y', I = 'P', Noise = 'N', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'Q', Noise = 'Q', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'Q', Noise = 'N', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'R', Noise = 'Q', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'R', Noise = 'N', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'S', Noise = 'Q', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'S', Noise = 'N', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'T', Noise = 'Q', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'T', Noise = 'N', Y = 1.0)
````





`DataFrames.describe` provides a convenient summary of a `DataFrame`.
````julia
describe(DataFrame(df))
````



<table class="data-frame"><thead><tr><th></th><th>variable</th><th>mean</th><th>min</th><th>median</th><th>max</th><th>nunique</th><th>nmissing</th><th>eltype</th></tr><tr><th></th><th>Symbol</th><th>Union…</th><th>Any</th><th>Union…</th><th>Any</th><th>Union…</th><th>Nothing</th><th>DataType</th></tr></thead><tbody><p>5 rows × 8 columns</p><tr><th>1</th><td>S</td><td></td><td>'A'</td><td></td><td>'y'</td><td>50</td><td></td><td>Char</td></tr><tr><th>2</th><td>Age</td><td></td><td>'O'</td><td></td><td>'Y'</td><td>2</td><td></td><td>Char</td></tr><tr><th>3</th><td>I</td><td></td><td>'A'</td><td></td><td>'T'</td><td>20</td><td></td><td>Char</td></tr><tr><th>4</th><td>Noise</td><td></td><td>'N'</td><td></td><td>'Q'</td><td>2</td><td></td><td>Char</td></tr><tr><th>5</th><td>Y</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td></td><td></td><td>Float64</td></tr></tbody></table>



#### Checking properties of the design

It is worthwhile checking that the design has the desired properties.
`S` (subject) and `I` (item) should be balanced, which can be checked in a cross-tabulation
````julia

freqtable(df, :I, :S)
````



```
20×50 Named Array{Int64,2}
I ╲ S │ 'A'  'B'  'C'  'D'  'E'  'F'  'G'  'H'  'I'  'J'  'K'  'L'  'M'  'N'  …  'l'  'm'  'n'  'o'  'p'  'q'  'r'  's'  't'  'u'  'v'  'w'  'x'  'y'
──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
'A'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10  …   10   10   10   10   10   10   10   10   10   10   10   10   10   10
'B'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'C'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'D'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'E'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'F'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'G'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'H'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'I'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'J'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'K'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'L'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'M'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'N'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'O'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'P'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'Q'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'R'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'S'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10      10   10   10   10   10   10   10   10   10   10   10   10   10   10
'T'   │  10   10   10   10   10   10   10   10   10   10   10   10   10   10  …   10   10   10   10   10   10   10   10   10   10   10   10   10   10
```
or, more compactly,
````julia
all(freqtable(df, :I, :S) .== 10)
````


````
true
````





Checking on the experimental variables, `Age` does not vary within levels of `S`
````julia
freqtable(df, :Age, :S)
````


````
2×50 Named Array{Int64,2}
Age ╲ S │ 'A'  'B'  'C'  'D'  'E'  'F'  …  't'  'u'  'v'  'w'  'x'  'y'
────────┼──────────────────────────────────────────────────────────────
'O'     │ 200  200  200  200  200  200  …    0    0    0    0    0    0
'Y'     │   0    0    0    0    0    0  …  200  200  200  200  200  200
````




However, `Age` does vary within levels of `I`
````julia
freqtable(df, :Age, :I)
````


````
2×20 Named Array{Int64,2}
Age ╲ I │ 'A'  'B'  'C'  'D'  'E'  'F'  …  'O'  'P'  'Q'  'R'  'S'  'T'
────────┼──────────────────────────────────────────────────────────────
'O'     │ 250  250  250  250  250  250  …  250  250  250  250  250  250
'Y'     │ 250  250  250  250  250  250  …  250  250  250  250  250  250
````




and `Noise` varies within levels of `S`
````julia
freqtable(df, :Noise, :S)
````


````
2×50 Named Array{Int64,2}
Noise ╲ S │ 'A'  'B'  'C'  'D'  'E'  'F'  …  't'  'u'  'v'  'w'  'x'  'y'
──────────┼──────────────────────────────────────────────────────────────
'N'       │ 100  100  100  100  100  100  …  100  100  100  100  100  100
'Q'       │ 100  100  100  100  100  100  …  100  100  100  100  100  100
````




and within levels of `I`
````julia
freqtable(df, :Noise, :I)
````


````
2×20 Named Array{Int64,2}
Noise ╲ I │ 'A'  'B'  'C'  'D'  'E'  'F'  …  'O'  'P'  'Q'  'R'  'S'  'T'
──────────┼──────────────────────────────────────────────────────────────
'N'       │ 250  250  250  250  250  250  …  250  250  250  250  250  250
'Q'       │ 250  250  250  250  250  250  …  250  250  250  250  250  250
````





## Creating a LinearMixedModel

A `LinearMixedModel` with fixed-effects for `Age` and `Noise` and for their interaction and with random intercepts for `S` and `I` is created as
````julia
const hc = HelmertCoding();
m1 = LinearMixedModel(@formula(Y ~ 1 + Age * Noise + (1|S) + (1|I)), df, Dict(:Age => hc, :Noise => hc));
m1.X                               # model matrix for fixed-effects terms
````


````
10000×4 Array{Float64,2}:
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 ⋮                    
 1.0   1.0  -1.0  -1.0
 1.0   1.0   1.0   1.0
 1.0   1.0  -1.0  -1.0
 1.0   1.0   1.0   1.0
 1.0   1.0  -1.0  -1.0
 1.0   1.0   1.0   1.0
 1.0   1.0  -1.0  -1.0
 1.0   1.0   1.0   1.0
 1.0   1.0  -1.0  -1.0
````





#### HelmertCoding of contrasts

The third argument in the call to `LinearMixedModel` is a dictionary of "contrasts" to use when forming the contrasts for categorical covariates.
`HelmertCoding` applied to a 2-level factor creates a  `±1` coding of the levels, as shown in the display of the model matrix.
With this coding the `(Intercept)` coefficient will be a "typical" response level without regard to `Age` and `Noise`.
In other words, the `(Intercept)` is not defined with respect to an arbitrary reference level for the categorical covariates.
Note that when 2-level factors are coded as `±1` the interaction terms also have a `±1` coding.

Sometimes coefficient estimates are called the "effect" of the condition in the covariate, e.g. "Noise" versus "Quiet".
For the `HelmertCoding` the "effect" of changing from the lower level to the higher level is twice the coefficient, because the distance between the `±1` values in the model matrix is 2.

## Simulating a response and fitting the model

The `MixedModels.simulate!` function installs a simulated response in the model object, given values of the parameters.

````julia
rng = Random.MersenneTwister(2052162715);  # repeatable random number generator
refit!(simulate!(rng, m1, β = [1000., 0, 0, 0], σ = 200., θ = [0.5, 0.5]))
````


````
Linear mixed model fit by maximum likelihood
 Y ~ 1 + Age + Noise + Age & Noise + (1 | S) + (1 | I)
     logLik        -2 logLik          AIC             BIC       
 -6.72750605×10⁴  1.34550121×10⁵  1.34564121×10⁵  1.34614593×10⁵

Variance components:
            Variance  Std.Dev.
 S          9606.815  98.01436
 I         10335.779 101.66503
 Residual  39665.426 199.16181
 Number of obs: 10000; levels of grouping factors: 50, 20

  Fixed-effects parameters:
                   Estimate Std.Error  z value P(>|z|)
(Intercept)         993.065      26.7  37.1934  <1e-99
Age: Y             -27.0764   14.0037 -1.93352  0.0532
Noise: Q           -2.21268   1.99162 -1.11099  0.2666
Age: Y & Noise: Q  0.257456   1.99162  0.12927  0.8971
````





The parameters are `β`, the fixed-effects coefficients, `σ`, the standard deviation of the per-observation, or "residual", noise term and `θ`, the parameters in the lower Cholesky factor of the relative covariance matrices for the random effects terms.

In this case, both random-effects terms are simple, scalar random effects with standard deviations of $100 = 200 * 0.5$.

Notice that the estimated standard deviations, 98.014 and 101.665 for the random effects and 199.16 for the residual noise, are very close to the values in the simulation.

Similarly, the estimates of the fixed-effects are quite close to the values in the simulation.

#### REML estimates

To use the REML criterion instead of maximum likelihood for parameter optimization, add the optional `REML` argument.
````julia
refit!(m1, REML=true)
````


````
Linear mixed model fit by REML
 Y ~ 1 + Age + Noise + Age & Noise + (1 | S) + (1 | I)
 REML criterion at convergence: 134528.135222923

Variance components:
            Variance  Std.Dev.
 S          9867.226  99.33391
 I         10736.171 103.61550
 Residual  39673.394 199.18181
 Number of obs: 10000; levels of grouping factors: 50, 20

  Fixed-effects parameters:
                   Estimate Std.Error  z value P(>|z|)
(Intercept)         993.065   27.1684  36.5522  <1e-99
Age: Y             -27.0764   14.1884 -1.90834  0.0563
Noise: Q           -2.21268   1.99182 -1.11088  0.2666
Age: Y & Noise: Q  0.257456   1.99182 0.129257  0.8972
````





Because the experimental design is balanced across subjects, items, age and noise, the fixed-effects parameter estimates are the same under ML or under REML.
This does not need to be the case for unbalanced designs.

The REML standard errors of the fixed-effects parameter estimates and the estimated variance components are all somewhat larger than those from ML, as would be expected.
Because the standard errors are larger for REML estimates, the p-values for the null hypothesis of no effect for a covariate or interaction are also larger.

The REML estimates are preferred for evaluating p-values for the fixed-effects coefficients, because they are more conservative.
However, REML estimates should not be used for comparing models in likelihood ratio tests or using various information criteria because the log-likelihood is not explicitly optimized.

In this package the `loglikelihood` extractor function does not return a value for a model fit by REML,
````julia
loglikelihood(m1)
````


<pre class="julia-error">
ERROR: ArgumentError: loglikelihood not available for models fit by REML
</pre>



hence all the other information criteria extractors also fail
````julia
aicc(m1)    # the corrected Akaike's Information Criterion
````


<pre class="julia-error">
ERROR: ArgumentError: loglikelihood not available for models fit by REML
</pre>




## Fitting alternative models to the same response

Define another model with random slopes with respect to `Noise` and random intercepts for both `S` and `I`.

````julia
m2 = LinearMixedModel(@formula(Y ~ 1 + Age * Noise + (1+Noise|S) + (1+Noise|I)), df,
     Dict(:Age => hc, :Noise => hc));
refit!(m2, response(m1))
````


````
Linear mixed model fit by maximum likelihood
 Y ~ 1 + Age + Noise + Age & Noise + (1 + Noise | S) + (1 + Noise | I)
     logLik       -2 logLik         AIC            BIC      
 -6.7274890×10⁴  1.3454978×10⁵  1.3457178×10⁵ 1.34651094×10⁵

Variance components:
              Variance      Std.Dev.     Corr.
 S          9607.09559138  98.01579256
               7.46097821   2.73147912 -0.25
 I         10335.72689027 101.66477704
               0.78373263   0.88528675 -1.00
 Residual  39657.13460277 199.14099177
 Number of obs: 10000; levels of grouping factors: 50, 20

  Fixed-effects parameters:
                   Estimate Std.Error  z value P(>|z|)
(Intercept)         993.065   26.7001  37.1933  <1e-99
Age: Y             -27.0764   14.0038  -1.9335  0.0532
Noise: Q           -2.21268   2.03817 -1.08562  0.2776
Age: Y & Noise: Q  0.257456   2.02853 0.126917  0.8990
````
