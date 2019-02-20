---
title: Simulation of Subject-Item mixed models
author: Douglas Bates
date: 2019-02-20
---

Following the publication of Barr et al. (2011) there has been considerable interest in simulation of subject-item types of data from mixed-effects models to assess the effect of the choice of random-effects structure on the Type I error of tests on the fixed-effects.
Here we show how such simulations can be carried out efficiently using the [`MixedModels`](https://github.com/dmbates/MixedModels.jl) package for [`Julia`](https://julialang.org).

## Data characteristics

The data characteristics are those from _Maybe maximal: Good enough mixed models optimize power while controlling Type I error_ by Seedorff, Oleson and McMurray, which is just one example of such a study.
There are 50 subjects, 25 from each of two age groups which we will denote by `'Y'` (younger) and `'O'` (older).
Each subject's response is measured on 5 different occasions on each of 20 different items under two noise conditions, `'N'` (no noise) and `'Y'` (noise).
Such an experimental design yields a total of 10,000 measurements.

In the data for this experimental design, the 25 younger subjects are labelled `'a'` to `'y'` while the older subjects are `'A'` to `'Y'` and the items are `'A'` to `'T'`.
````julia
julia> using DataFrames, FreqTables, MixedModels, Random, StatsModels, Tables

julia> df = (S = repeat(['A':'Y'; 'a':'y'], inner=40, outer=5),
    Age = repeat(['O','Y'], inner=1000, outer=5),
    I = repeat('A':'T', inner=2, outer=250),
    Noise = repeat(['N','Y'], outer=5000),
    Y = ones(10000));

julia> describe(DataFrame(df))
5×8 DataFrame
│ Row │ variable │ mean   │ min │ median │ max │ nunique │ nmissing │ eltype   │
│     │ Symbol   │ Union… │ Any │ Union… │ Any │ Union…  │ Nothing  │ DataType │
├─────┼──────────┼────────┼─────┼────────┼─────┼─────────┼──────────┼──────────┤
│ 1   │ S        │        │ 'A' │        │ 'y' │ 50      │          │ Char     │
│ 2   │ Age      │        │ 'O' │        │ 'Y' │ 2       │          │ Char     │
│ 3   │ I        │        │ 'A' │        │ 'T' │ 20      │          │ Char     │
│ 4   │ Noise    │        │ 'N' │        │ 'Y' │ 2       │          │ Char     │
│ 5   │ Y        │ 1.0    │ 1.0 │ 1.0    │ 1.0 │         │          │ Float64  │

julia> freqtable(df, :I, :S)
20×50 Named Array{Int64,2}
I ╲ S │ 'A'  'B'  'C'  'D'  'E'  'F'  'G'  …  's'  't'  'u'  'v'  'w'  'x'  'y'
──────┼────────────────────────────────────────────────────────────────────────
'A'   │  10   10   10   10   10   10   10  …   10   10   10   10   10   10   10
'B'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'C'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'D'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'E'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'F'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'G'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'H'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'I'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
⋮         ⋮    ⋮    ⋮    ⋮    ⋮    ⋮    ⋮  ⋱    ⋮    ⋮    ⋮    ⋮    ⋮    ⋮    ⋮
'L'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'M'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'N'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'O'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'P'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'Q'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'R'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'S'   │  10   10   10   10   10   10   10      10   10   10   10   10   10   10
'T'   │  10   10   10   10   10   10   10  …   10   10   10   10   10   10   10

julia> freqtable(df, :Age, :S)    # Age does not vary within levels of S
2×50 Named Array{Int64,2}
Age ╲ S │ 'A'  'B'  'C'  'D'  'E'  'F'  …  't'  'u'  'v'  'w'  'x'  'y'
────────┼──────────────────────────────────────────────────────────────
'O'     │ 200  200  200  200  200  200  …    0    0    0    0    0    0
'Y'     │   0    0    0    0    0    0  …  200  200  200  200  200  200

julia> freqtable(df, :Age, :I)    # Age does vary within levels of I
2×20 Named Array{Int64,2}
Age ╲ I │ 'A'  'B'  'C'  'D'  'E'  'F'  …  'O'  'P'  'Q'  'R'  'S'  'T'
────────┼──────────────────────────────────────────────────────────────
'O'     │ 250  250  250  250  250  250  …  250  250  250  250  250  250
'Y'     │ 250  250  250  250  250  250  …  250  250  250  250  250  250

julia> freqtable(df, :Noise, :S)  # Noise does vary within levels of S
2×50 Named Array{Int64,2}
Noise ╲ S │ 'A'  'B'  'C'  'D'  'E'  'F'  …  't'  'u'  'v'  'w'  'x'  'y'
──────────┼──────────────────────────────────────────────────────────────
'N'       │ 100  100  100  100  100  100  …  100  100  100  100  100  100
'Y'       │ 100  100  100  100  100  100  …  100  100  100  100  100  100

julia> freqtable(df, :Noise, :I)  # and within levels of I
2×20 Named Array{Int64,2}
Noise ╲ I │ 'A'  'B'  'C'  'D'  'E'  'F'  …  'O'  'P'  'Q'  'R'  'S'  'T'
──────────┼──────────────────────────────────────────────────────────────
'N'       │ 250  250  250  250  250  250  …  250  250  250  250  250  250
'Y'       │ 250  250  250  250  250  250  …  250  250  250  250  250  250

````




The response column, `Y`, is added as a placeholder.

As an aside, although `df` is a `NamedTuple`, which is similar to a `list` in `R` except that the names are `Symbol`s, not `String`s,
````julia
julia> typeof(df)
NamedTuple{(:S, :Age, :I, :Noise, :Y),Tuple{Array{Char,1},Array{Char,1},Array{Char,1},Array{Char,1},Array{Float64,1}}}

````




it is easily converted to a `DataFrame` if desired (as shown in the call to `describe`).

The trend in Julia packages supporting data science, like the `StatsModels` package, is towards data representations as "column tables" (a `NamedTuple` of arrays) or "row tables" (a vector of `NamedTuple`s).
Sometimes it is convenient to work on individual columns, sometimes it makes more sense to iterate over rows.
The `columntable` and `rowtable` functions allow for conversion back and forth between the two representations.

````julia
julia> rowtable(df)
10000-element Array{NamedTuple{(:S, :Age, :I, :Noise, :Y),Tuple{Char,Char,Char,Char,Float64}},1}:
 (S = 'A', Age = 'O', I = 'A', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'A', Noise = 'Y', Y = 1.0)
 (S = 'A', Age = 'O', I = 'B', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'B', Noise = 'Y', Y = 1.0)
 (S = 'A', Age = 'O', I = 'C', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'C', Noise = 'Y', Y = 1.0)
 (S = 'A', Age = 'O', I = 'D', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'D', Noise = 'Y', Y = 1.0)
 (S = 'A', Age = 'O', I = 'E', Noise = 'N', Y = 1.0)
 (S = 'A', Age = 'O', I = 'E', Noise = 'Y', Y = 1.0)
 ⋮                                                  
 (S = 'y', Age = 'Y', I = 'P', Noise = 'Y', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'Q', Noise = 'N', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'Q', Noise = 'Y', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'R', Noise = 'N', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'R', Noise = 'Y', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'S', Noise = 'N', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'S', Noise = 'Y', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'T', Noise = 'N', Y = 1.0)
 (S = 'y', Age = 'Y', I = 'T', Noise = 'Y', Y = 1.0)

````





## Creating a LinearMixedModel

A `LinearMixedModel` with fixed-effects for `Age` and `Noise` and for their interaction and with random intercepts for `S` and `I` is created as
````julia
julia> const hc = HelmertCoding()
HelmertCoding(nothing, nothing)

julia> m1 = LinearMixedModel(@formula(Y ~ 1 + Age * Noise + (1|S) + (1|I)), df,
    Dict(:Age => hc, :Noise => hc));

julia> first(m1.feterms).x               # model matrix for fixed-effects terms
10000×4 Array{Float64,2}:
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 1.0  -1.0  -1.0   1.0
 1.0  -1.0   1.0  -1.0
 ⋮                    
 1.0   1.0   1.0   1.0
 1.0   1.0  -1.0  -1.0
 1.0   1.0   1.0   1.0
 1.0   1.0  -1.0  -1.0
 1.0   1.0   1.0   1.0
 1.0   1.0  -1.0  -1.0
 1.0   1.0   1.0   1.0
 1.0   1.0  -1.0  -1.0
 1.0   1.0   1.0   1.0

````




The third argument in the call to `LinearMixedModel` is a dictionary of "hints" to use when forming the contrasts for categorical covariates.
The `HelmertCoding` uses levels of `-1` and `+1`, as shown in the display of the model matrix.
With this coding the `(Intercept)` coefficient will be a "typical" response level without regard to `Age` and `Noise`.
In other words, the `(Intercept)` is not defined with respect to an arbitrary reference level for the categorical covariates.
Note that when 2-level factors are coded as `±1` the interaction terms also have a `±1` coding.

Sometimes coefficient estimates are called the "effect" of the condition in the covariate, e.g. "Noise" versus "no Noise".
For this encoding the "effect" of changing from the lower level to the higher level is half the coefficient, because the distance between the `±1` values in the model matrix is 2. 

## Simulating a response and fitting the model

The `MixedModels.simulate!` function installs a simulated response in the model object given values of the parameters.

````julia
julia> rng = Random.MersenneTwister(2052162715);  # repeatable random number generator

julia> refit!(simulate!(rng, m1, β = [1000., 0, 0, 0], σ = 200., θ = [0.5, 0.5]))
Linear mixed model fit by maximum likelihood
 Y ~ 1 + Age + Noise + Age & Noise + (1 | S) + (1 | I)
     logLik        -2 logLik          AIC             BIC       
 -6.72750605×10⁴  1.34550121×10⁵  1.34564121×10⁵  1.34614593×10⁵

Variance components:
            Variance  Std.Dev.  
 S          9606.817  98.014371
 I         10335.783 101.665055
 Residual  39665.426 199.161809
 Number of obs: 10000; levels of grouping factors: 50, 20

  Fixed-effects parameters:
                    Estimate Std.Error  z value P(>|z|)
(Intercept)          993.065      26.7  37.1934  <1e-99
Age: Y              -27.0764   14.0037 -1.93352  0.0532
Noise: Y             2.21268   1.99162  1.11099  0.2666
Age: Y & Noise: Y  -0.257456   1.99162 -0.12927  0.8971


````




To use the REML criterion instead of maximum likelihood for parameter optimization, add the optional `REML` argument.

````julia
julia> fit!(m1, REML=true)
Linear mixed model fit by REML
 Y ~ 1 + Age + Noise + Age & Noise + (1 | S) + (1 | I)
 REML criterion at convergence: 134528.13522292292

Variance components:
            Variance  Std.Dev.  
 S          9867.247  99.334016
 I         10736.193 103.615603
 Residual  39673.394 199.181811
 Number of obs: 10000; levels of grouping factors: 50, 20

  Fixed-effects parameters:
                    Estimate Std.Error   z value P(>|z|)
(Intercept)          993.065   27.1684   36.5522  <1e-99
Age: Y              -27.0764   14.1885  -1.90834  0.0563
Noise: Y             2.21268   1.99182   1.11088  0.2666
Age: Y & Noise: Y  -0.257456   1.99182 -0.129257  0.8972


````





## Fitting alternative models to the same response

Define another model with random slopes with respect to `Noise` and random intercepts for both `S` and `I`.

````julia
julia> m2 = LinearMixedModel(@formula(Y ~ 1 + Age * Noise + (1+Noise|S) + (1+Noise|I)), df,
     Dict(:Age => HelmertCoding(), :Noise => HelmertCoding()));

julia> refit!(m2, response(m1))
Linear mixed model fit by maximum likelihood
 Y ~ 1 + Age + Noise + Age & Noise + (1 + Noise | S) + (1 + Noise | I)
     logLik       -2 logLik         AIC            BIC      
 -6.7274890×10⁴  1.3454978×10⁵  1.3457178×10⁵ 1.34651094×10⁵

Variance components:
              Variance      Std.Dev.    Corr.
 S          9606.86459184  98.0146142
               7.46527102   2.7322648  0.25
 I         10335.57901585 101.6640498
               0.78408368   0.8854850  1.00
 Residual  39657.13610404 199.1409955
 Number of obs: 10000; levels of grouping factors: 50, 20

  Fixed-effects parameters:
                    Estimate Std.Error   z value P(>|z|)
(Intercept)          993.065   26.6998   37.1937  <1e-99
Age: Y              -27.0764   14.0037  -1.93352  0.0532
Noise: Y             2.21268   2.03819   1.08561  0.2777
Age: Y & Noise: Y  -0.257456   2.02855 -0.126916  0.8990


````


