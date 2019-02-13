# Singular covariance estimates in random regression models

This notebook explores the occurrence of singularity in the estimated covariance matrix of random regression models.
These are mixed-effects models with vector-valued random effects.

First, fit a model to the `sleepstudy` data from [`lme4`](https://github.com/lme4/lme4).

## Fitting a linear mixed-model to the sleepstudy data

Load the required packages

````julia
julia> using DataFrames, FreqTables, LinearAlgebra, MixedModels, Random, RData

julia> using Gadfly

julia> using Gadfly.Geom: density, histogram, point

julia> using Gadfly.Guide: xlabel, ylabel

julia> const dat = Dict(Symbol(k)=>v for (k,v) in 
    load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")));

````





However, the `LinearMixedModel` constructor only creates a model structure but does not fit it.
An explicit call to `fit!` is required to fit the model.
As is customary (though not required) in Julia, a function whose name ends in `!` is a _mutating function_ that modifies one or more of its arguments.

An optional second argument of `true` in the call to `fit!` produces verbose output from the optimization.

````julia
julia> sleepm = fit!(LinearMixedModel(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy]), verbose=true)
f_1: 1784.6423 [1.0, 0.0, 1.0]
f_2: 1790.12564 [1.75, 0.0, 1.0]
f_3: 1798.99962 [1.0, 1.0, 1.0]
f_4: 1803.8532 [1.0, 0.0, 1.75]
f_5: 1800.61398 [0.25, 0.0, 1.0]
f_6: 1798.60463 [1.0, -1.0, 1.0]
f_7: 1752.26074 [1.0, 0.0, 0.25]
f_8: 1797.58769 [1.18326, -0.00866189, 0.0]
f_9: 1754.95411 [1.075, 0.0, 0.325]
f_10: 1753.69568 [0.816632, 0.0111673, 0.288238]
f_11: 1754.817 [1.0, -0.0707107, 0.196967]
f_12: 1753.10673 [0.943683, 0.0638354, 0.262696]
f_13: 1752.93938 [0.980142, -0.0266568, 0.274743]
f_14: 1752.25688 [0.984343, -0.0132347, 0.247191]
f_15: 1752.05745 [0.97314, 0.00253785, 0.23791]
f_16: 1752.02239 [0.954526, 0.00386421, 0.235892]
f_17: 1752.02273 [0.935929, 0.0013318, 0.234445]
f_18: 1751.97169 [0.954965, 0.00790664, 0.229046]
f_19: 1751.9526 [0.953313, 0.0166274, 0.225768]
f_20: 1751.94852 [0.946929, 0.0130761, 0.222871]
f_21: 1751.98718 [0.933418, 0.00613767, 0.218951]
f_22: 1751.98321 [0.951544, 0.005789, 0.220618]
f_23: 1751.95197 [0.952809, 0.0190332, 0.224178]
f_24: 1751.94628 [0.946322, 0.0153739, 0.225088]
f_25: 1751.9467 [0.947124, 0.0148894, 0.224892]
f_26: 1751.94757 [0.946497, 0.0154643, 0.225814]
f_27: 1751.94531 [0.946086, 0.0157934, 0.224449]
f_28: 1751.94418 [0.945304, 0.0166902, 0.223361]
f_29: 1751.94353 [0.944072, 0.0172106, 0.222716]
f_30: 1751.94244 [0.941271, 0.0163099, 0.222523]
f_31: 1751.94217 [0.939, 0.015899, 0.222132]
f_32: 1751.94237 [0.938979, 0.016548, 0.221562]
f_33: 1751.94228 [0.938863, 0.0152466, 0.222683]
f_34: 1751.9422 [0.938269, 0.015733, 0.222024]
f_35: 1751.94131 [0.938839, 0.0166373, 0.222611]
f_36: 1751.94093 [0.938397, 0.0173965, 0.222817]
f_37: 1751.94057 [0.937006, 0.0180445, 0.222534]
f_38: 1751.94018 [0.934109, 0.0187354, 0.22195]
f_39: 1751.94008 [0.932642, 0.0189242, 0.221726]
f_40: 1751.94027 [0.931357, 0.0190082, 0.221309]
f_41: 1751.9415 [0.932821, 0.0206454, 0.221367]
f_42: 1751.93949 [0.931867, 0.0179574, 0.222564]
f_43: 1751.93939 [0.929167, 0.0177824, 0.222534]
f_44: 1751.9394 [0.929659, 0.0177721, 0.222508]
f_45: 1751.93943 [0.929193, 0.0187806, 0.22257]
f_46: 1751.93935 [0.928986, 0.0182366, 0.222484]
f_47: 1751.93949 [0.928697, 0.0182937, 0.223175]
f_48: 1751.93936 [0.928243, 0.0182695, 0.222584]
f_49: 1751.93934 [0.929113, 0.0181791, 0.222624]
f_50: 1751.93934 [0.929191, 0.0181658, 0.222643]
f_51: 1751.93935 [0.929254, 0.0182093, 0.222621]
f_52: 1751.93935 [0.929189, 0.0181298, 0.222573]
f_53: 1751.93934 [0.929254, 0.0181676, 0.22265]
f_54: 1751.93934 [0.929214, 0.0181717, 0.222647]
f_55: 1751.93934 [0.929208, 0.0181715, 0.222646]
f_56: 1751.93934 [0.929209, 0.018173, 0.222652]
f_57: 1751.93934 [0.929221, 0.0181684, 0.222645]
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + U + ((1 + U) | G)
   logLik   -2 logLik     AIC        BIC    
 -875.96967 1751.93934 1763.93934 1783.09709

Variance components:
              Column    Variance   Std.Dev.    Corr.
 G        (Intercept)  565.510660 23.7804680
          U             32.682124  5.7168281  0.08
 Residual              654.941449 25.5918239
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   251.405   6.63226 37.9064  <1e-99
U             10.4673   1.50224 6.96781  <1e-11


````





The variables in the optimization are the elements of a lower triangular matrix, $\Lambda$, which is the relative covariance factor of the random effects.
The corresponding parameter vector is called $\theta$.

````julia
julia> Λ = sleepm.λ[1]
2×2 LowerTriangular{Float64,Array{Float64,2}}:
 0.929221    ⋅      
 0.0181684  0.222645

````





The matrix $\Lambda$ is the left (or lower) Cholesky factor of the covariance matrix of the unconditional distribution of the vector-valued random effects, relative to the variance, $\sigma^2$, of the per-observation noise.
That is
```math
\begin{equation}
    \Sigma = \sigma^2\Lambda\Lambda'
\end{equation}
```
In terms of the estimates,

````julia
julia> s² = varest(sleepm)    # estimate of the residual variance
654.941448668924

````



````julia
julia> s² * Λ * Λ'   # unconditional covariance matrix of the random effects
2×2 Array{Float64,2}:
 565.511  11.057 
  11.057  32.6821

````





The estimated correlation of the random effects can, of course, be evaluated from the covariance matrix.
Writing out the expressions for the elements of the covariance matrix in terms of the elements of Λ shows that many terms cancel in the evaluation of the correlation, resulting in the simpler formula.

````julia
julia> Λ[2, 1] / sqrt(Λ[2, 1]^2 + Λ[2, 2]^2)
0.08133212094454777

````





For a $2\times 2$ covariance matrix it is not terribly important to perform this calculation in an efficient and numerically stable way.
However, it is a good idea to pay attention to stability and efficiency in a calculation that can be repeated tens of thousands of times in a simulation or a parametric bootstrap.
The `norm` function evaluates the (geometric) length of a vector in a way that controls round-off better than the naive calculation.
The `view` function provides access to a subarray, such as the second row of $\Lambda$, without generating a copy.
Thus the estimated correlation can be written

````julia
julia> Λ[2, 1] / norm(view(Λ, 2, :))
0.08133212094454777

````





### Optimization with respect to θ

As described in section 3 of the 2015 _Journal of Statistical Software_ [paper](https://www.jstatsoft.org/index.php/jss/article/view/v067i01/v67i01.pdf) by Bates, Maechler, Bolker and Walker, using the relative covariance factor, $\Lambda$, in the formulation of mixed-effects models in the [`lme4`](https://github.com/lme4/lme4) and [`MixedModels`](https://github.com/dmbates/MixedModels) packages and using the vector $\theta$ as the optimization variable was a conscious choice.
Indeed, a great deal of effort went into creating this form so that the profiled log-likelihood can be easily evaluated and so that the constraints on the parameters, $\theta$, are simple "box" constraints.
In fact, the constraints are simple lower bounds.

````julia
julia> show(sleepm.lowerbd)
[0.0, -Inf, 0.0]
````





In contrast, trying to optimize the log-likelihood with respect to standard deviations and correlations of the random effects
would be quite difficult because the constraints on the correlations when the covariance matrix is larger than $2\times 2$ are quite complicated.
Also, the correlation itself can be unstable.
Consider what happens to the expression for the correlation if both $\Lambda_{2,1}$ and $\Lambda_{2,2}$ are small in magnitude.
Small perturbations in $\Lambda_{2,1}$ that result in sign changes can move the correlation from near $-1$ to near $+1$ or vice-versa.

Some details on the optimization process are available in an `OptSummary` object stored as the `optsum` field of the model.

````julia
julia> sleepm.optsum
Initial parameter vector: [1.0, 0.0, 1.0]
Initial objective value:  1784.6422961924686

Optimizer (from NLopt):   LN_BOBYQA
Lower bounds:             [0.0, -Inf, 0.0]
ftol_rel:                 1.0e-12
ftol_abs:                 1.0e-8
xtol_rel:                 0.0
xtol_abs:                 [1.0e-10, 1.0e-10, 1.0e-10]
initial_step:             [0.75, 1.0, 0.75]
maxfeval:                 -1

Function evaluations:     57
Final parameter vector:   [0.929221, 0.0181684, 0.222645]
Final objective value:    1751.9393444646948
Return code:              FTOL_REACHED


````





### Convergence on the boundary

Determining if an estimated covariance matrix is singular is easy when using the $\theta$  parameters because singularity corresponds to points on the boundary of the allowable parameter space.
In other words, if the optimization converges to a vector in which either or both of $\theta_1$ or $\theta_3$ are zero, the covariance matrix is singular.
Otherwise it is non-singular.

The $\theta_1$ parameter is the estimated relative standard deviation of the random intercepts.
If this is zero then the correlation is undefined and reported as `NaN`.
If $\theta_3$ is zero and $\theta_2$ is non-zero then the estimated correlation is $\pm 1$ with the sign determined by the sign of $\theta_2$.
If both $\theta_2$ and $\theta_3$ are zero the correlation is `NaN` because the standard deviation of the random slopes will be zero.

Singular covariance matrices larger than $2\times 2$ do not necessarily result in particular values, like ±1, for the correlations.

Users of `lmer` or `lmm` are sometimes taken aback by convergence on the boundary if this produces correlations of `NaN` or $\pm 1$.
Some feel that this is a sign of model failure.
Others consider such estimates as a sign that Bayesian methods with priors that pull singular covariance matrices away from the boundary should be used.

This type of value judgement seems peculiar.
An important property of maximum likelihood estimates is that these estimates are well-defined once the probability model for the data has been specified.
It may be difficult to determine numerical values of the estimates but the definition itself is straightforward.
If there is a direct method of evaluating the log-likelihood at a particular value of the parameters, then, by definition, the mle's are the parameter values that maximize this log-likelihood.
Bates et al. (2015) provide such a method of evaluating the log-likelihood for a linear mixed-effects model.
Indeed they go further and describe how the fixed-effects parameters and one of the variance components can be profiled out of the log-likelihood evaluation, thereby reducing the dimension of the nonlinear, constrained optimization problem to be solved.

If the mle's correspond to a singular covariance matrix, this is a property of the model and the data.
It is not a mistake in some way.
It is just the way things are.
It reflects the fact that often the distribution of the estimator of a covariance matrix is diffuse.
It is difficult to estimate variances and covariances precisely.
A search for papers or books on "covariance estimation" will produce many results, often describing ways of regularizing the estimating process because the data themselves do not provide precise estimates.

For the example at hand a parametric bootstrap is one way of evaluating the precision of the estimates.

## The bootstrap function

The `MixedModels` package provides a `bootstrap` method to create a parametric bootstrap sample from a fitted model.

For reproducibility, set the random number seed to some arbitrary value.

````julia
julia> Random.seed!(1234321);

````





Arguments to the `bootstrap` function are the number of samples to generate and the model from which to generate them.
By default the converged parameter estimates are those used to generate the samples.
Addition, named arguments can be used to override these parameter values, allowing `bootstrap` to be used for simulation.

`bootstrap` returns a `DataFrame` with columns:
- `obj`: the objective (-2 loglikelihood)
- `σ`: the standard deviation of the per-observation noise
- `β₁` to `βₚ`: the fixed-effects coefficients
- `θ₁` to `θₖ`: the covariance parameter elements
- `σ₁` to `σₛ`: the estimates standard deviations of random effects.
- `ρ₁` to `ρₜ`: the estimated correlations of random effects

The `ρᵢ` and `σᵢ` values are derived from the `θᵢ` and `σ` values.

````julia
julia> sleepmbstrp = bootstrap(10000, sleepm);

julia> show(names(sleepmbstrp))
Symbol[:obj, :σ, :β₁, :β₂, :θ₁, :θ₂, :θ₃, :σ₁, :σ₂, :ρ₁]
````





Recall that the constrained parameters are $\theta_1$ and $\theta_3$ which both must be non-negative.
If either or both of these are zero (in practice the property to check is if they are "very small", which here is arbitrarily defined as less than 0.00001) then the covariance matrix is singular.

````julia
julia> issmall(x) = x < 0.00001   # defines a one-liner function in Julia
issmall (generic function with 1 method)

````



````julia
julia> freqtable(issmall.(sleepmbstrp[:θ₁]), issmall.(sleepmbstrp[:θ₃]))
2×2 Named Array{Int64,2}
Dim1 ╲ Dim2 │ false   true
────────────┼─────────────
false       │  9687    306
true        │     7      0

````





Here the covariance matrix estimate is non-singular in 9,686 of the 10,000 samples, has an zero estimated intercept variance in 6 samples and is otherwise singular (i.e. correlation estimate of $\pm 1$) in 308 samples.

Empirical densities of the θ components are:

![](./assets/SingularCovariance_14_1.svg)

![](./assets/SingularCovariance_15_1.svg)



A density plot is typically a good way to visualize such a large sample.
However, when there is a spike such as the spike at zero here, a histogram provides a more informative plot.

![](./assets/SingularCovariance_16_1.svg)

![](./assets/SingularCovariance_17_1.svg)



### Reciprocal condition number

The definitve way to assess singularity of the estimated covariance matrix is by its _condition number_ or, alternatively, its _reciprocal condition number_.
In general the condition number, $\kappa$, of a matrix is the ratio of the largest singular value to the smallest.
For singular matrices it is $\infty$, which is why it is often more convenient to evaluate and plot $\kappa^{-1}$.
Because $\kappa$ is a ratio of singular values it is unaffected by nonzero scale factors.
Thus
```math
\begin{equation}
\kappa^{-1}(s^2\Lambda\Lambda') = \kappa^{-1}(\Lambda\Lambda') =
[\kappa^{-1}(\Lambda)]^2
\end{equation}
```
````julia
function recipcond(bstrp::DataFrame)
    T = eltype(bstrp[:θ₁])
    val = sizehint!(T[], size(bstrp, 1))
    d = Matrix{T}(undef, 2, 2)
    for (t1, t2, t3) in zip(bstrp[:θ₁], bstrp[:θ₂], bstrp[:θ₃])
        d[1, 1] = t1
        d[1, 2] = t2
        d[2, 2] = t3
        v = svdvals!(d)
        push!(val, v[2] / v[1])
    end
    val
end
rc = recipcond(sleepmbstrp)
````



![](./assets/SingularCovariance_19_1.svg)



$\kappa^{-1}$ is small if either or both of $\theta_1$ or $\theta_3$ is small.

````julia
julia> sum(issmall, rc)
313

````





The density of the estimated correlation

![](./assets/SingularCovariance_21_1.svg)

````julia
julia> sum(isfinite, sleepmbstrp[:ρ₁])  # recall that ρ = NaN in 7 cases
9993

````



````julia
julia> sum(x -> x == -1, sleepmbstrp[:ρ₁])  # number of cases of rho == -1
2

````



````julia
julia> sum(x -> x == +1, sleepmbstrp[:ρ₁])  # number of cases of rho == +1
304

````





In this case the bootstrap simulations that resulted in $\rho = -1$ were not close to being indeterminant with respect to sign.
That is, the values of $\theta_2$ were definitely negative.

````julia
julia> sleepmbstrp[:θ₂][findall(x -> x == -1, sleepmbstrp[:ρ₁])]
2-element Array{Float64,1}:
 -0.26586204029006416
 -0.25449503128677364

````





## The Oxboys data

In the `nlme` package for R there are several data sets to which random regression models are fit.
The `RCall` package for Julia provides the ability to run an embedded R process and communicate with it.
The simplest form of writing R code within Julia is to use character strings prepended with `R`.
In Julia strings are delimited by `"` or by `"""`.
With `"""` multi-line strings are allowed.

````julia
julia> using RCall

julia> R"""
library(nlme)
plot(Oxboys)
"""
RObject{NilSxp}
NULL


````



````julia
julia> oxboys = rcopy(R"Oxboys");

julia> show(names(oxboys))
Symbol[:Subject, :age, :height, :Occasion]
````



````julia
julia> oxboysm = fit(LinearMixedModel, @formula(height ~ 1 + age + (1+age | Subject)), oxboys)
Linear mixed model fit by maximum likelihood
 Formula: height ~ 1 + age + ((1 + age) | Subject)
   logLik   -2 logLik     AIC        BIC    
 -362.98384  725.96769  737.96769  758.69962

Variance components:
              Column     Variance   Std.Dev.    Corr.
 Subject  (Intercept)  62.78894329 7.92394746
          age           2.71159677 1.64669268  0.64
 Residual               0.43545647 0.65989126
 Number of obs: 234; levels of grouping factors: 26

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   149.372   1.55461 96.0829  <1e-99
age           6.52547  0.329763 19.7884  <1e-86


````



````julia
julia> show(getθ(oxboysm))
[12.008, 1.60158, 1.91362]
````





As seen in the plot and by the estimate $\widehat{\theta_1} = 12.0$, the estimated standard deviation of the random intercepts is much greater than the residual standard deviation.
It is unlikely that bootstrap samples will include singular covariance estimates.

````julia
julia> Random.seed!(4321234);

julia> oxboysmbtstrp = bootstrap(10000, oxboysm);

````





In this bootstrap sample, there are no singular estimated covariance matrices for the random effects.

````julia
julia> freqtable(issmall.(oxboysmbtstrp[:θ₁]), issmall.(oxboysmbtstrp[:θ₃]))
1×1 Named Array{Int64,2}
Dim1 ╲ Dim2 │ false
────────────┼──────
false       │ 10000

````





The empirical density of the correlation estimates shows that even in this case the correlation is not precisely estimated.

![](./assets/SingularCovariance_32_1.svg)

````julia
julia> extrema(oxboysmbtstrp[:ρ₁])
(0.033623235242859324, 0.9352618141809922)

````





The reciprocal condition number

````julia
julia> rc = recipcond(oxboysmbtstrp);

julia> extrema(rc)
(0.06198775211313265, 0.3625146203502122)

````





does not get very close to zero.

![](./assets/SingularCovariance_35_1.svg)



## The Orthodont data

````julia
julia> R"plot(Orthodont)"
RObject{VecSxp}


````





The subject labels distinguish between the male and the female subjects.  Consider first the female subjects only.

````julia
julia> orthfemale = rcopy(R"subset(Orthodont, Sex == 'Female', -Sex)");

julia> orthfm = fit!(LinearMixedModel(@formula(distance ~ 1 + age + (1 + age | Subject)), orthfemale))
Linear mixed model fit by maximum likelihood
 Formula: distance ~ 1 + age + ((1 + age) | Subject)
   logLik   -2 logLik     AIC        BIC    
  -67.25463  134.50927  146.50927  157.21441

Variance components:
              Column     Variance   Std.Dev.    Corr.
 Subject  (Intercept)  2.970884485 1.72362539
          age          0.021510368 0.14666413 -0.30
 Residual              0.446615832 0.66829322
 Number of obs: 44; levels of grouping factors: 27

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   17.3727  0.725169 23.9568  <1e-99
age          0.479545 0.0631313   7.596  <1e-13


````



````julia
julia> Random.seed!(1234123)
MersenneTwister(UInt32[0x0012d4cb], Random.DSFMT.DSFMT_state(Int32[1849428804, 1072710534, 1722234079, 1073299110, 2058053067, 1072801015, 18044541, 1072957251, 668716466, 1073001711  …  -1153221639, 1073553062, 1653158638, 1073411494, 780501209, -2117144994, -394908522, -1446490633, 382, 0]), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], UInt128[0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000  …  0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000], 1002, 0)

julia> orthfmbtstrp = bootstrap(10000, orthfm);

````



````julia
julia> freqtable(issmall.(orthfmbtstrp[:θ₁]), issmall.(orthfmbtstrp[:θ₃]))
2×2 Named Array{Int64,2}
Dim1 ╲ Dim2 │ false   true
────────────┼─────────────
false       │  6771   3194
true        │    35      0

````





For this model almost 1/3 of the bootstrap samples converge to singular covariance estimates for the vector-valued random effects.
A histogram of the estimated correlations of the random effects is dominated by the boundary values.

![](./assets/SingularCovariance_40_1.svg)



Even though the estimated correlation in the model is -0.30, more of the boundary values are at +1 than at -1.
This may be an artifact of the optimization routine used.
In some cases there may be multiple optima on the boundary.
It is difficult to determine the global optimum in these cases.

A histogram of the reciprocal condition number is also dominated by the boundary values.




## Early childhood cognitive study

This example from Singer and Willett (2003), *Applied Longitudinal Data Analysis* was the motivation for reformulating the estimation methods to allow for singular covariance matrices. Cognitive scores (`cog`) were recorded at `age` 1, 1.5 and 2 years on 103 infants, of whom 58 were in the treatment group and 45 in the control group.  The treatment began at age 6 months (0.5 years).  The data are available as the `Early` data set in the `mlmRev` package for R.  In the model, time on study (`tos`) is used instead of age because the zero point on the time scale should be when the treatment begins.

````julia
julia> R"""
suppressMessages(library(mlmRev))
library(lattice)
Early$tos <- Early$age - 0.5
Early$trttos <- Early$tos * (Early$trt == "Y")
xyplot(cog ~ tos | reorder(id, cog, min), Early, 
    type = c("p","l","g"), aspect="xy")
"""
RObject{VecSxp}


````





Notice that most of these plots within subjects have a negative slope and that the scores at 1 year of age (`tos = 0.5`) are frequently greater than would be expected on an age-adjusted scale.

````julia
julia> R"print(xtabs(cog ~ age + trt, Early) / xtabs(~ age + trt, Early))";
     trt
age           N         Y
  1   108.53333 112.93103
  1.5  95.88889 110.29310
  2    87.40000  97.06897

````





When the time origin is the beginning of the treatment there is not generally a "main effect" for the treatment but only an interaction of `trt` and `tos`.

````julia
julia> early = rcopy(R"subset(Early, select = c(cog, tos, id, trt, trttos))");

julia> earlym = fit(LinearMixedModel, @formula(cog ~ 1 + tos + trttos + (1 + tos | id)), early)
Linear mixed model fit by maximum likelihood
 Formula: cog ~ 1 + tos + trttos + ((1 + tos) | id)
   logLik   -2 logLik     AIC        BIC    
 -1185.6369  2371.2738  2385.2738  2411.4072

Variance components:
              Column    Variance   Std.Dev.    Corr.
 id       (Intercept)  165.476297 12.8637590
          tos           10.744812  3.2779279 -1.00
 Residual               74.946887  8.6571870
 Number of obs: 309; levels of grouping factors: 103

  Fixed-effects parameters:
             Estimate Std.Error  z value P(>|z|)
(Intercept)   120.783    1.8178  66.4447  <1e-99
tos           -22.474    1.4878 -15.1055  <1e-50
trttos        7.65206   1.43609  5.32841   <1e-7


````





The model converges to a singular covariance matrix for the random effects.

````julia
julia> getθ(earlym)
3-element Array{Float64,1}:
  1.4859051786273185 
 -0.37863660770421764
  0.0                

````





The conditional (on the observed responses) means of the random effects fall along a line.

![](./assets/SingularCovariance_46_1.svg)
