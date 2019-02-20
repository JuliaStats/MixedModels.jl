
# A Simple, Linear, Mixed-effects Model

In this book we describe the theory behind a type of statistical model called *mixed-effects* models and the practice of fitting and analyzing such models using the [`MixedModels`](https://github.com/dmbates/MixedModels.jl) package for [`Julia`](https://julialang.org).
These models are used in many different disciplines.
Because the descriptions of the models can vary markedly between disciplines, we begin by describing what mixed-effects models are and by exploring a very simple example of one type of mixed model, the *linear mixed model*.

This simple example allows us to illustrate the use of the `lmm` function in the `MixedModels` package for fitting such models and other functions 
for analyzing the fitted model.
We also describe methods of assessing the precision of the parameter estimates and of visualizing the conditional distribution of the random effects, given the observed data.

## Mixed-effects Models

Mixed-effects models, like many other types of statistical models, describe a relationship between a *response* variable and some of the *covariates* that have been measured or observed along with the response.
In mixed-effects models at least one of the covariates is a *categorical* covariate representing experimental or observational “units” in the data set.
In the example from the chemical industry discussed in this chapter, the observational unit is the batch of an intermediate product used in production of a dye.
In medical and social sciences the observational units are often the human or animal subjects in the study.
In agriculture the experimental units may be the plots of land or the specific plants being studied.

In all of these cases the categorical covariate or covariates are observed at a set of discrete *levels*.
We may use numbers, such as subject identifiers, to designate the particular levels that we observed but these numbers are simply labels.
The important characteristic of a categorical covariate is that, at each observed value of the response, the covariate takes on the value of one of a set of distinct levels.

Parameters associated with the particular levels of a covariate are sometimes called the “effects” of the levels.
If the set of possible levels of the covariate is fixed and reproducible we model the covariate using *fixed-effects* parameters.
If the levels that were observed represent a random sample from the set of all possible levels we incorporate *random effects* in the model.

There are two things to notice about this distinction between fixed-effects parameters and random effects.
First, the names are misleading because the distinction between fixed and random is more a property of the levels of the categorical covariate than a property of the effects associated with them.
Secondly, we distinguish between “fixed-effects parameters”, which are indeed parameters in the statistical model, and “random effects”, which, strictly speaking, are not parameters.
As we will see shortly, random effects are unobserved random variables.

To make the distinction more concrete, suppose the objective is to model the annual reading test scores for students in a school district and that the covariates recorded with the score include a student identifier and the student’s gender.
Both of these are categorical covariates.
The levels of the gender covariate, *male* and *female*, are fixed.
If we consider data from another school district or we incorporate scores from earlier tests, we will not change those levels.
On the other hand, the students whose scores we observed would generally be regarded as a sample from the set of all possible students whom we could have observed.
Adding more data, either from more school districts or from results on previous or subsequent tests, will increase the number of distinct levels of the student identifier.

*Mixed-effects models* or, more simply, *mixed models* are statistical models that incorporate both fixed-effects parameters and random effects.
Because of the way that we will define random effects, a model with random effects always includes at least one fixed-effects parameter.
Thus, any model with random effects is a mixed model.

We characterize the statistical model in terms of two random variables: a $q$-dimensional vector of random effects represented by the random variable $\mathcal{B}$ and an $n$-dimensional response vector represented by the random variable $\mathcal{Y}$.
(We use upper-case “script” characters to denote random variables.
The corresponding lower-case upright letter denotes a particular value of the random variable.)
We observe the value, $\bf{y}$, of $\mathcal{Y}$. We do not observe the value, $\bf{b}$, of $\mathcal{B}$.

When formulating the model we describe the unconditional distribution of $\mathcal{B}$ and the conditional distribution, $(\mathcal{Y}|\mathcal{B}=\bf{b})$.
The descriptions of the distributions involve the form of the distribution and the values of certain parameters.
We use the observed values of the response and the covariates to estimate these parameters and to make inferences about them.

That’s the big picture.
Now let’s make this more concrete by describing a particular, versatile class of mixed models called *linear mixed models* and by studying a simple example of such a model.
First we describe the data in the example.

## The `Dyestuff` and `Dyestuff2` Data

Models with random effects have been in use for a long time.
The first edition of the classic book, *Statistical Methods in Research and Production*, edited by O.L. Davies, was published in 1947 and contained examples of the use of random effects to characterize batch-to-batch variability in chemical processes.
The data from one of these examples are available as `Dyestuff` in the `MixedModels` package.
In this section we describe and plot these data and introduce a second example, the `Dyestuff2` data, described in Box and Tiao (1973).

### The `Dyestuff` Data

The data are described in Davies (), the fourth edition of the book mentioned above, as coming from

> an investigation to find out how much the variation from batch to batch in the quality of an intermediate product (H-acid) contributes to the variation in the yield of the dyestuff (Naphthalene Black 12B) made from it. In the experiment six samples of the intermediate, representing different batches of works manufacture, were obtained, and five preparations of the dyestuff were made in the laboratory from each sample. The equivalent yield of each preparation as grams of standard colour was determined by dye-trial.

First attach the packages to be used
````julia
julia> using DataFrames, Distributions, GLM, LinearAlgebra, MixedModels

julia> using Random, RCall, RData, SparseArrays

julia> using Gadfly

````




and allow for unqualified names for some graphics functions
````julia
julia> using Gadfly.Geom: point, line, histogram, density, vline

julia> using Gadfly.Guide: xlabel, ylabel, yticks

````





The `Dyestuff` data are available in the [`lme4`](https://github.com/lme4/lme4) package for [`R`](http://r-project.org).
This data frame and others have been stored in saved RData format in the `test` directory within the `MixedModels` package.

Access the `Dyestuff` data
````julia
julia> const dat = Dict(Symbol(k)=>v for (k,v) in 
    load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")));

julia> dyestuff = dat[:Dyestuff];

julia> describe(dyestuff)
2×8 DataFrame. Omitted printing of 1 columns
│ Row │ variable │ mean   │ min    │ median │ max    │ nunique │ nmissing │
│     │ Symbol   │ Union… │ Any    │ Union… │ Any    │ Union…  │ Nothing  │
├─────┼──────────┼────────┼────────┼────────┼────────┼─────────┼──────────┤
│ 1   │ G        │        │ A      │        │ F      │ 6       │          │
│ 2   │ Y        │ 1527.5 │ 1440.0 │ 1530.0 │ 1635.0 │         │          │

````




and plot it
![Yield versus Batch for the Dyestuff data](./assets/SimpleLMM_4_1.svg)



In the dotplot we can see that there is considerable variability in yield, even for preparations from the same batch, but there is also noticeable batch-to-batch variability.
For example, four of the five preparations from batch F provided lower yields than did any of the preparations from batches B, C, and E.

Recall that the labels for the batches are just labels and that their ordering is arbitrary.
In a plot, however, the order of the levels influences the perception of the pattern.
Rather than providing an arbitrary pattern it is best to order the levels according to some criterion for the plot.
In this case a good choice is to order the batches by increasing mean yield, which can be easily done in R.

````julia
julia> dyestuffR = rcopy(R"within(lme4::Dyestuff, Batch <- reorder(Batch, Yield, mean))");

julia> plot(dyestuffR, x = :Yield, y = :Batch, point, xlabel("Yield of dyestuff (g)"), ylabel("Batch"))
Plot(...)

````





In Sect. [sec:DyestuffLMM] we will use mixed models to quantify the variability in yield between batches.
For the time being let us just note that the particular batches used in this experiment are a selection or sample from the set of all batches that we wish to consider.
Furthermore, the extent to which one particular batch tends to increase or decrease the mean yield of the process — in other words, the “effect” of that particular batch on the yield — is not as interesting to us as is the extent of the variability between batches.
For the purposes of designing, monitoring and controlling a process we want to predict the yield from future batches, taking into account the batch-to-batch variability and the within-batch variability.
Being able to estimate the extent to which a particular batch in the past increased or decreased the yield is not usually an important goal for us.
We will model the effects of the batches as random effects rather than as fixed-effects parameters.

### The `Dyestuff2` Data

The data are simulated data presented in Box and Tiao (1973), where the authors state

> These data had to be constructed for although examples of this sort undoubtedly occur in practice they seem to be rarely published.

The structure and summary are intentionally similar to those of the `Dyestuff` data.
````julia
julia> dyestuff2 = dat[:Dyestuff2];

julia> describe(dyestuff2)
2×8 DataFrame. Omitted printing of 1 columns
│ Row │ variable │ mean   │ min    │ median │ max    │ nunique │ nmissing │
│     │ Symbol   │ Union… │ Any    │ Union… │ Any    │ Union…  │ Nothing  │
├─────┼──────────┼────────┼────────┼────────┼────────┼─────────┼──────────┤
│ 1   │ G        │        │ A      │        │ F      │ 6       │          │
│ 2   │ Y        │ 5.6656 │ -0.892 │ 5.365  │ 13.434 │         │          │

````




As can be seen in a data plot
![](./assets/SimpleLMM_7_1.svg)


the batch-to-batch variability in these data is small compared to the within-batch variability.
In some approaches to mixed models it can be difficult to fit models to such data.
Paradoxically, small “variance components” can be more difficult to estimate than large variance components.

The methods we will present are not compromised when estimating small variance components.

## Fitting Linear Mixed Models

Before we formally define a linear mixed model, let’s go ahead and fit models to these data sets using `lmm` which takes, as its first two arguments, a *formula* specifying the model and the *data* with which to evaluate the formula. 

The structure of the formula will be explained after showing the example.

### A Model For the `Dyestuff` Data

A model allowing for an overall level of the `Yield` and for an additive random effect for each level of `Batch` can be fit as
````julia
julia> mm1 = fit(LinearMixedModel, @formula(Y ~ 1 + (1 | G)), dyestuff)
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -163.66353  327.32706  333.32706  337.53065

Variance components:
              Column    Variance  Std.Dev. 
 G        (Intercept)  1388.3334 37.260347
 Residual              2451.2500 49.510100
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)    1527.5   17.6946  86.326  <1e-99


````





As shown in the summary of the model fit, the default estimation criterion is maximum likelihood.
The summary provides several other model-fit statistics such as Akaike’s Information Criterion (`AIC`), Schwarz’s Bayesian Information Criterion (`BIC`), the log-likelihood at the parameter estimates, and the objective function (negative twice the log-likelihood) at the parameter estimates.
These are all statistics related to the model fit and are used to compare different models fit to the same data.

The third section is the table of estimates of parameters associated with the random effects.
There are two sources of variability in this model, a batch-to-batch variability in the level of the response and the residual or per-observation variability — also called the within-batch variability.
The name “residual” is used in statistical modeling to denote the part of the variability that cannot be explained or modeled with the other terms.
It is the variation in the observed data that is “left over” after determining the estimates of the parameters in the other parts of the model.

Some of the variability in the response is associated with the fixed-effects terms.
In this model there is only one such term, labeled the `(Intercept)`.
The name “intercept”, which is better suited to models based on straight lines written in a slope/intercept form, should be understood to represent an overall “typical” or mean level of the response in this case.
(For those wondering about the parentheses around the name, they are included so that a user cannot accidentally name a variable in conflict with this name.)
The line labeled `Batch` in the random effects table shows that the random effects added to the intercept term, one for each level of the factor, are modeled as random variables whose unconditional variance is estimated as 1388.33 g$^2$.
The corresponding standard deviations is 37.26 g for the ML fit.

Note that the last column in the random effects summary table is the estimate of the variability expressed as a standard deviation rather than as a variance.
These are provided because it is usually easier to visualize the variability in standard deviations, which are on the scale of the response, than it is to visualize the magnitude of a variance.
The values in this column are a simple re-expression (the square root) of the estimated variances.
Do not confuse them with the standard errors of the variance estimators, which are not given here.
As described in later sections, standard errors of variance estimates are generally not useful because the distribution of the estimator of a variance is skewed - often badly skewed.

The line labeled `Residual` in this table gives the estimate, 2451.25 g$^2$, of the variance of the residuals and the corresponding standard deviation, 49.51 g.
In written descriptions of the model the residual variance parameter is written $\sigma^2$ and the variance of the random effects is $\sigma_1^2$.
Their estimates are $\widehat{\sigma^2}$ and $\widehat{\sigma_1^2}$

The last line in the random effects table states the number of observations to which the model was fit and the number of levels of any “grouping factors” for the random effects.
In this case we have a single random effects term, `(1 | Batch)`, in the model formula and the grouping factor for that term is `Batch`.
There will be a total of six random effects, one for each level of `Batch`.

The final part of the printed display gives the estimates and standard errors of any fixed-effects parameters in the model.
The only fixed-effects term in the model formula is the `(Intercept)`.
The estimate of this parameter is 1527.5 g.
The standard error of the intercept estimate is 17.69 g.

### A Model For the `Dyestuff2` Data

Fitting a similar model to the `dyestuff2` data produces an estimate $\widehat{\sigma_1^2}=0$.

````julia
julia> mm2 = fit(LinearMixedModel, @formula(Y ~ 1 + (1 | G)), dyestuff2)
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -81.436518 162.873037 168.873037 173.076629

Variance components:
              Column    Variance  Std.Dev. 
 G        (Intercept)   0.000000 0.0000000
 Residual              13.346099 3.6532314
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)    5.6656  0.666986 8.49433  <1e-16


````





An estimate of `0` for $\sigma_1$ does not mean that there is no variation between the groups.
Indeed Fig. [fig:Dyestuff2dot] shows that there is some small amount of variability between the groups.
The estimate, $\widehat{\sigma_1}$, is a measure of the “between-group” variability that is **in excess of** the variability induced by the "within-group" or residual variability in the responses.  

If 30 observations were simulated from a "normal" (also called "Gaussian") distribution and divided arbitrarily into 6 groups of 5, a plot of the data would look very much like Fig. [fig:Dyestuff2dot]. 
(In fact, it is likely that this is how that data set was generated.)
It is only where there is excess variability between the groups that $\widehat{\sigma_1}>0$.
Obtaining $\widehat{\sigma_1}=0$ is not a mistake; it is simply a statement about the data and the model.

The important point to take away from this example is the need to allow for the estimates of variance components that are zero.
Such a model is said to be *singular*, in the sense that it corresponds to a linear model in which we have removed the random effects associated with `Batch`.
Singular models can and do occur in practice.
Even when the final fitted model is not singular, we must allow for such models to be expressed when determining the parameter estimates through numerical optimization.

It happens that this model corresponds to the linear model (i.e. a model with fixed-effects only)
````julia
julia> lm1 = lm(@formula(Y ~ 1), dyestuff2)
StatsModels.DataFrameRegressionModel{LinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: Y ~ +1

Coefficients:
             Estimate Std.Error t value Pr(>|t|)
(Intercept)    5.6656  0.678388 8.35156    <1e-8


````





The log-likelihood for this model
````julia
julia> loglikelihood(lm1)
-81.43651832691287

````




is the same as that of `fm2`.
The standard error of the intercept in `lm1` is a bit larger than that of `fm2` because the estimate of the residual variance is evaluated differently in the linear model.

### Further Assessment of the Fitted Models

The parameter estimates in a statistical model represent our “best guess” at the unknown values of the model parameters and, as such, are important results in statistical modeling. However, they are not the whole story. Statistical models characterize the variability in the data and we must assess the effect of this variability on the parameter estimates and on the precision of predictions made from the model.

In Sect. [sec:variability] we introduce a method of assessing variability in parameter estimates using the “profiled log-likelihood” and in Sect. [sec:assessRE] we show methods of characterizing the conditional distribution of the random effects given the data. Before we get to these sections, however, we should state in some detail the probability model for linear mixed-effects and establish some definitions and notation. In particular, before we can discuss profiling the log-likelihood, we should define the log-likelihood. We do that in the next section.

## The Linear Mixed-effects Probability Model

In explaining some of parameter estimates related to the random effects we have used terms such as “unconditional distribution” from the theory of probability.
Before proceeding further we clarify the linear mixed-effects probability model and define several terms and concepts that will be used throughout the book.
Readers who are more interested in practical results than in the statistical theory should feel free to skip this section.

### Definitions and Results

In this section we provide some definitions and formulas without derivation and with minimal explanation, so that we can use these terms in what follows.
In Chap. [chap:computational] we revisit these definitions providing derivations and more explanation.

As mentioned in Sect. [sec:memod], a mixed model incorporates two random variables: $\mathcal{B}$, the $q$-dimensional vector of random effects, and $\mathcal{Y}$, the $n$-dimensional response vector. In a linear mixed model the unconditional distribution of $\mathcal{B}$ and the conditional distribution, $(\mathcal{Y} | \mathcal{B}=\bf{b})$, are both multivariate Gaussian distributions,

\begin{aligned}
  (\mathcal{Y} | \mathcal{B}=\bf{b}) &\sim\mathcal{N}(\bf{ X\beta + Z b},\sigma^2\bf{I})\\\\
  \mathcal{B}&\sim\mathcal{N}(\bf{0},\Sigma_\theta) .
\end{aligned}

The *conditional mean* of $\mathcal Y$, given $\mathcal B=\bf b$, is the *linear predictor*, $\bf X\bf\beta+\bf Z\bf b$, which depends on the $p$-dimensional *fixed-effects parameter*, $\bf \beta$, and on $\bf b$. The *model matrices*, $\bf X$ and $\bf Z$, of dimension $n\times p$ and $n\times q$, respectively, are determined from the formula for the model and the values of covariates. Although the matrix $\bf Z$ can be large (i.e. both $n$ and $q$ can be large), it is sparse (i.e. most of the elements in the matrix are zero).

The *relative covariance factor*, $\Lambda_\theta$, is a $q\times q$ lower-triangular matrix, depending on the *variance-component parameter*, $\bf\theta$, and generating the symmetric $q\times q$ variance-covariance matrix, $\Sigma_\theta$, as
```math
\begin{equation}
  \Sigma_\theta=\sigma^2\Lambda_\theta\Lambda_\theta'
\end{equation}
```
The *spherical random effects*, $\mathcal{U}\sim\mathcal{N}({\bf 0},\sigma^2{\bf I}_q)$, determine $\mathcal B$ according to
```math
\begin{equation}
  \mathcal{B}=\Lambda_\theta\mathcal{U}.
\end{equation}
```
The *penalized residual sum of squares* (PRSS),
```math
\begin{equation}
  r^2(\theta,\beta,{\bf u})=\|{\bf y} -{\bf X}\beta -{\bf Z}\Lambda_\theta{\bf u}\|^2+\|{\bf u}\|^2,
\end{equation}
```
is the sum of the residual sum of squares, measuring fidelity of the model to the data, and a penalty on the size of $\bf u$, measuring the complexity of the model.
Minimizing $r^2$ with respect to $\bf u$,
```math
\begin{equation}
  r^2_{\beta,\theta} =\min_{\bf u}\left\{\|{\bf y} -{\bf X}{\beta} -{\bf Z}\Lambda_\theta{\bf u}\|^2+\|{\bf u}\|^2\right\}
\end{equation}
```
is a direct (i.e. non-iterative) computation.
The particular method used to solve this generates a *blocked Choleksy factor*, ${\bf L}_\theta$, which is an lower triangular $q\times q$ matrix satisfying
```math
\begin{equation}
  {\bf L}_\theta{\bf L}_\theta'=\Lambda_\theta'{\bf Z}'{\bf Z}\Lambda_\theta+{\bf I}_q .
\end{equation}
```
where ${\bf I}_q$ is the $q\times q$ *identity matrix*.

Negative twice the log-likelihood of the parameters, given the data, $\bf y$, is
```math
\begin{equation}
d({\bf\theta},{\bf\beta},\sigma|{\bf y})
=n\log(2\pi\sigma^2)+\log(|{\bf L}_\theta|^2)+\frac{r^2_{\beta,\theta}}{\sigma^2}.
\end{equation}
```
where $|{\bf L}_\theta|$ denotes the *determinant* of ${\bf L}_\theta$.
Because ${\bf L}_\theta$ is triangular, its determinant is the product of its diagonal elements.

Negative twice the log-likelihood will be called the *objective* in what follows.
It is the value to be minimized by the parameter estimates.  It is, up to an additive factor, the *deviance* of the parameters.  Unfortunately, it is not clear what the additive factor should be in the case of linear mixed models.  In many applications, however, it is not the deviance of the model that is of interest as much the change in the deviance between two fitted models.  When calculating the change in the deviance the additive factor will cancel out so the change in the deviance when comparing models is the same as the change in this objective.

Because the conditional mean, $\bf\mu_{\mathcal Y|\mathcal B=\bf b}=\bf
X\bf\beta+\bf Z\Lambda_\theta\bf u$, is a linear function of both $\bf\beta$ and $\bf u$, minimization of the PRSS with respect to both $\bf\beta$ and $\bf u$ to produce
```math
\begin{equation}
r^2_\theta =\min_{{\bf\beta},{\bf u}}\left\{\|{\bf y} -{\bf X}{\bf\beta} -{\bf Z}\Lambda_\theta{\bf u}\|^2+\|{\bf u}\|^2\right\}
\end{equation}
```
is also a direct calculation.
The values of $\bf u$ and $\bf\beta$ that provide this minimum are called, respectively, the *conditional mode*, $\tilde{\bf u}_\theta$, of the spherical random effects and the conditional estimate, $\widehat{\bf\beta}_\theta$, of the fixed effects.
At the conditional estimate of the fixed effects the objective is
```math
\begin{equation}
d({\bf\theta},\widehat{\beta}_\theta,\sigma|{\bf y})
=n\log(2\pi\sigma^2)+\log(|{\bf L}_\theta|^2)+\frac{r^2_\theta}{\sigma^2}.
\end{equation}
```
Minimizing this expression with respect to $\sigma^2$ produces the conditional estimate
```math
\begin{equation}
\widehat{\sigma^2}_\theta=\frac{r^2_\theta}{n}
\end{equation}
```
which provides the *profiled log-likelihood* on the deviance scale as
```math
\begin{equation}
\tilde{d}(\theta|{\bf y})=d(\theta,\widehat{\beta}_\theta,\widehat{\sigma}_\theta|{\bf y})
=\log(|{\bf L}_\theta|^2)+n\left[1+\log\left(\frac{2\pi r^2_\theta}{n}\right)\right],
\end{equation}
```
a function of $\bf\theta$ alone.

The MLE of $\bf\theta$, written $\widehat{\bf\theta}$, is the value that minimizes this profiled objective.
We determine this value by numerical optimization.
In the process of evaluating $\tilde{d}(\widehat{\theta}|{\bf y})$ we determine $\widehat{\beta}=\widehat{\beta}_{\widehat\theta}$, $\tilde{\bf u}_{\widehat{\theta}}$ and $r^2_{\widehat{\theta}}$, from which we can evaluate $\widehat{\sigma}=\sqrt{r^2_{\widehat{\theta}}/n}$.

The elements of the conditional mode of $\mathcal B$, evaluated at the parameter estimates,
```math
\begin{equation}
\tilde{\bf b}_{\widehat{\theta}}=
\Lambda_{\widehat{\theta}}\tilde{\bf u}_{\widehat{\theta}}
\end{equation}
```
are sometimes called the *best linear unbiased predictors* or BLUPs of the random effects.
Although BLUPs an appealing acronym, I don’t find the term particularly instructive (what is a “linear unbiased predictor” and in what sense are these the “best”?) and prefer the term “conditional modes”, because these are the values of $\bf b$ that maximize the density of the conditional distribution $\mathcal{B} | \mathcal{Y} = {\bf y}$.
For a linear mixed model, where all the conditional and unconditional distributions are Gaussian, these values are also the *conditional means*.

### Fields of a `LinearMixedModel` object

The optional second argument, `verbose`, in a call to `fit!` of a `LinearMixedModel` object produces output showing the progress of the iterative optimization of $\tilde{d}(\bf\theta|\bf y)$.

````julia
julia> mm1 = fit!(LinearMixedModel(@formula(Y ~ 1 + (1 | G)), dyestuff), verbose = true);
f_1: 327.76702 [1.0]
f_2: 331.03619 [1.75]
f_3: 330.64583 [0.25]
f_4: 327.69511 [0.97619]
f_5: 327.56631 [0.928569]
f_6: 327.3826 [0.833327]
f_7: 327.35315 [0.807188]
f_8: 327.34663 [0.799688]
f_9: 327.341 [0.792188]
f_10: 327.33253 [0.777188]
f_11: 327.32733 [0.747188]
f_12: 327.32862 [0.739688]
f_13: 327.32706 [0.752777]
f_14: 327.32707 [0.753527]
f_15: 327.32706 [0.752584]
f_16: 327.32706 [0.752509]
f_17: 327.32706 [0.752591]
f_18: 327.32706 [0.752581]

````





The algorithm converges after 18 function evaluations to a profiled deviance of 327.32706 at $\theta=0.752581$. In this model the parameter $\theta$ is of length 1, the single element being the ratio $\sigma_1/\sigma$.

Whether or not verbose output is requested, the `optsum` field of a `LinearMixedModel` contains information on the optimization.  The various tolerances or the optimizer name can be changed between creating a `LinearMixedModel` and calling `fit!` on it to exert finer control on the optimization process.

````julia
julia> mm1.optsum
Initial parameter vector: [1.0]
Initial objective value:  327.76702162461663

Optimizer (from NLopt):   LN_BOBYQA
Lower bounds:             [0.0]
ftol_rel:                 1.0e-12
ftol_abs:                 1.0e-8
xtol_rel:                 0.0
xtol_abs:                 [1.0e-10]
initial_step:             [0.75]
maxfeval:                 -1

Function evaluations:     18
Final parameter vector:   [0.752581]
Final objective value:    327.3270598811424
Return code:              FTOL_REACHED


````





The full list of fields in a `LinearMixedModel` object is

````julia
julia> fieldnames(LinearMixedModel)
(:formula, :trms, :sqrtwts, :A, :L, :optsum)

````





The `formula` field is a copy of the model formula

````julia
julia> mm1.formula
Formula: Y ~ 1 + (1 | G)

````





The `mf` field is a `ModelFrame` constructed from the `formula` and `data` arguments to `lmm`.
It contains a `DataFrame` formed by reducing the `data` argument to only those columns needed to evaluate the `formula` and only those rows that have no missing data.
It also contains information on the terms in the model formula and on any "contrasts" associated with categorical variables in the fixed-effects terms.

The `trms` field is a vector of numerical objects representing the terms in the model, including the response vector.
As the names imply, the `sqrtwts` and `wttrms` fields are for incorporating case weights.
These fields are not often used when fitting linear mixed models but are vital to the process of fitting a generalized linear mixed model, described in Chapter [sec:glmm].  When used, `sqrtwts` is a diagonal matrix of size `n`. A size of `0` indicates weights are not used.

````julia
julia> mm1.sqrtwts
0-element Array{Float64,1}

````





The `trms` field is a vector of length $\ge 3$.

````julia
julia> length(mm1.trms)
3

````





The last two elements are $\bf X$, the $n\times p$ model matrix for the fixed-effects parameters, $\bf\beta$, and $\bf y$, the response vector stored as a matrix of size $n\times 1$.  In `mm1`, $\bf X$ consists of a single column of 1's

````julia
julia> mm1.trms[end - 1]
MatrixTerm{Float64,Array{Float64,2}}([1.0; 1.0; … ; 1.0; 1.0], [1.0; 1.0; … ; 1.0; 1.0], [1], 1, ["(Intercept)"])

````



````julia
julia> mm1.trms[end]
MatrixTerm{Float64,Array{Float64,2}}([1545.0; 1440.0; … ; 1480.0; 1445.0], [1545.0; 1440.0; … ; 1480.0; 1445.0], [1], 0, [""])

````





The elements of `trms` before the last two represent vertical sections of $\bf Z$ associated with the random effects terms in the model.  In `mm1` there is only one random effects term, `(1 | Batch)`, and $\bf Z$ has only one section, the one generated by this term, of type `ScalarReMat`.

````julia
julia> mm1.trms[1]
ScalarFactorReTerm{Float64,UInt8}(UInt8[0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02  …  0x05, 0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x06, 0x06, 0x06], ["A", "B", "C", "D", "E", "F"], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], :G, ["(Intercept)"], 0.75258072175308)

````





In practice these matrices are stored in a highly condensed form because, in some models, they can be very large.
In small examples the structure is more obvious when the `ScalarReMat` is converted to a sparse or a dense matrix.

````julia
julia> sparse(mm1.trms[1])
30×6 SparseMatrixCSC{Float64,Int32} with 30 stored entries:
  [1 ,  1]  =  1.0
  [2 ,  1]  =  1.0
  [3 ,  1]  =  1.0
  [4 ,  1]  =  1.0
  [5 ,  1]  =  1.0
  [6 ,  2]  =  1.0
  [7 ,  2]  =  1.0
  [8 ,  2]  =  1.0
  [9 ,  2]  =  1.0
  ⋮
  [22,  5]  =  1.0
  [23,  5]  =  1.0
  [24,  5]  =  1.0
  [25,  5]  =  1.0
  [26,  6]  =  1.0
  [27,  6]  =  1.0
  [28,  6]  =  1.0
  [29,  6]  =  1.0
  [30,  6]  =  1.0

````



````julia
julia> Matrix(mm1.trms[1])
30×6 Array{Float64,2}:
 1.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 ⋮                        ⋮  
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  1.0

````





The `A` field is a representation of the blocked, square, symmetric matrix $\bf A = [Z : X : y]'[Z : X : y]$.
Only the upper triangle of `A` is stored.
The number of blocks of the rows and columns of `A` is the number of vertical sections of $\bf Z$ (i.e. the number of random-effects terms) plus 2.

````julia
julia> nblocks(mm1.A)
(3, 3)

````



````julia
julia> mm1.A[Block(1, 1)]
6×6 Diagonal{Float64,Array{Float64,1}}:
 5.0   ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅   5.0   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   5.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅   5.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   5.0   ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   5.0

````



````julia
julia> mm1.A[Block(2, 1)]
1×6 Array{Float64,2}:
 5.0  5.0  5.0  5.0  5.0  5.0

````



````julia
julia> mm1.A[Block(2, 2)]
1×1 Array{Float64,2}:
 30.0

````



````julia
julia> mm1.A[Block(3, 1)]
1×6 Array{Float64,2}:
 7525.0  7640.0  7820.0  7490.0  8000.0  7350.0

````



````julia
julia> mm1.A[Block(3, 2)]
1×1 Array{Float64,2}:
 45825.0

````



````julia
julia> mm1.A[Block(3, 3)]
1×1 Array{Float64,2}:
 7.0112875e7

````





## Fields modified during the optimization

Changing the value of $\theta$ changes the $\Lambda$ field.
(Note: to input a symbol like $\Lambda$ in a Jupyter code cell or in the Julia read-eval-print loop (REPL), type `\Lambda` followed by a tab character.  Such "latex completions" are available for many UTF-8 characters used in Julia.)  The matrix $\Lambda$ has a special structure.  It is a block diagonal matrix where the diagonal blocks are [`Kronecker product`](https://en.wikipedia.org/wiki/Kronecker_product)s of an identity matrix and a (small) lower triangular matrix.  The diagonal blocks correspond to the random-effects terms.  For a scalar random-effects term, like `(1 | Batch)` the diagonal block is the Kronecker product of an identity matrix and a $1\times 1$ matrix.  This result in this case is just a multiple of the identity matrix.

It is not necessary to store the full $\Lambda$ matrix.  Storing the small lower-triangular matrices is sufficient.

````julia
julia> getΛ(mm1)
1-element Array{Float64,1}:
 0.75258072175308

````





The `L` field is a blocked matrix like the `A` field containing the upper Cholesky factor of

\begin{bmatrix}
  \bf{\Lambda'Z'Z\Lambda + I} & \bf{\Lambda'Z'X} & \bf{\Lambda'Z'y} \\
  \bf{X'Z\Lambda} & \bf{X'X} & \bf{X'y} \\
  \bf{y'Z\Lambda} & \bf{y'Z} & \bf{y'y}
\end{bmatrix}   

````julia
julia> mm1.L
8×8 LowerTriangular{Float64,BlockArrays.BlockArray{Float64,2,AbstractArray{Float64,2}}}:
    1.95752      ⋅           ⋅       …      ⋅           ⋅          ⋅   
    0.0         1.95752      ⋅              ⋅           ⋅          ⋅   
    0.0         0.0         1.95752         ⋅           ⋅          ⋅   
    0.0         0.0         0.0             ⋅           ⋅          ⋅   
    0.0         0.0         0.0             ⋅           ⋅          ⋅   
    0.0         0.0         0.0    
 ──────────────────────────────────  …     1.95752      ⋅          ⋅   
 ───────────────────────────────
    1.92228     1.92228     1.92228
 ──────────────────────────────────        1.92228     2.79804     ⋅   
 ───────────────────────────────
 2893.03     2937.24     3006.45        2825.75     4274.01     271.178

````



````julia
julia> mm1.L.data[Block(1, 1)]
6×6 Diagonal{Float64,Array{Float64,1}}:
 1.95752   ⋅        ⋅        ⋅        ⋅        ⋅     
  ⋅       1.95752   ⋅        ⋅        ⋅        ⋅     
  ⋅        ⋅       1.95752   ⋅        ⋅        ⋅     
  ⋅        ⋅        ⋅       1.95752   ⋅        ⋅     
  ⋅        ⋅        ⋅        ⋅       1.95752   ⋅     
  ⋅        ⋅        ⋅        ⋅        ⋅       1.95752

````



````julia
julia> mm1.L.data[Block(2, 1)]
1×6 Array{Float64,2}:
 1.92228  1.92228  1.92228  1.92228  1.92228  1.92228

````



````julia
julia> mm1.L.data[Block(2, 2)]
1×1 Array{Float64,2}:
 2.7980417055919244

````



````julia
julia> mm1.L.data[Block(3, 1)]
1×6 Array{Float64,2}:
 2893.03  2937.24  3006.45  2879.58  3075.65  2825.75

````



````julia
julia> mm1.L.data[Block(3, 2)]
1×1 Array{Float64,2}:
 4274.008705291662

````



````julia
julia> mm1.L.data[Block(3, 3)]
1×1 Array{Float64,2}:
 271.177984264646

````





All the information needed to evaluate the profiled log-likelihood is available in the `R` field; $\log(|\bf R_\theta|^2)$ is

````julia
julia> 2 * sum(log.(diag(mm1.L.data[Block(1,1)])))
8.060146910373954

````





It can also be evaluated as `logdet(mm1)` or `2 * logdet(mm1.R[1, 1])`

````julia
julia> logdet(mm1) == (2*logdet(mm1.L.data[Block(1, 1)])) == (2*sum(log.(diag(mm1.L.data[Block(1, 1)]))))
true

````





The penalized residual sum of squares is the square of the single element of the lower-right block, `R[3, 3]` in this case

````julia
julia> abs2(mm1.L.data[Block(3, 3)][1, 1])
73537.4991498366

````



````julia
julia> pwrss(mm1)
73537.4991498366

````





The objective is

````julia
julia> logdet(mm1) + nobs(mm1) * (1 + log(2π * pwrss(mm1) / nobs(mm1)))
327.3270598811424

````





## Assessing variability of parameter estimates

### Parametric bootstrap samples

One way to assess the variability of the parameter estimates is to generate a parametric bootstrap sample from the model.  The technique is to simulate response vectors from the model at the estimated parameter values and refit the model to each of these simulated responses, recording the values of the parameters.  The `bootstrap` method for these models performs these simulations and returns 4 arrays: a vector of objective (negative twice the log-likelihood) values, a vector of estimates of $\sigma^2$, a matrix of estimates of the fixed-effects parameters and a matrix of the estimates of the relative covariance parameters.  In this case there is only one fixed-effects parameter and one relative covariance parameter, which is the ratio of the standard deviation of the random effects to the standard deviation of the per-sample noise.

First set the random number seed for reproducibility.

````julia
julia> Random.seed!(1234321);

julia> mm1bstp = bootstrap(100000, mm1);

julia> size(mm1bstp)
(100000, 5)

````



````julia
julia> describe(mm1bstp)
5×8 DataFrame. Omitted printing of 2 columns
│ Row │ variable │ mean     │ min     │ median   │ max     │ nunique │
│     │ Symbol   │ Float64  │ Float64 │ Float64  │ Float64 │ Nothing │
├─────┼──────────┼──────────┼─────────┼──────────┼─────────┼─────────┤
│ 1   │ obj      │ 324.03   │ 284.104 │ 324.346  │ 353.114 │         │
│ 2   │ σ        │ 48.8259  │ 23.0686 │ 48.6577  │ 81.1291 │         │
│ 3   │ β₁       │ 1527.41  │ 1442.75 │ 1527.47  │ 1598.74 │         │
│ 4   │ θ₁       │ 0.611039 │ 0.0     │ 0.610791 │ 2.9935  │         │
│ 5   │ σ₁       │ 28.9064  │ 0.0     │ 29.6869  │ 94.001  │         │

````





#### Histograms, kernel density plots and quantile-quantile plots

I am a firm believer in the value of plotting results **before** summarizing them.
Well chosen plots can provide insights not available from a simple numerical summary.
It is common to visualize the distribution of a sample using a *histogram*, which approximates the shape of the probability density function.
The density can also be approximated more smoothly using a *kernel density* plot.
Finally, the extent to which the distribution of a sample can be approximated by a particular distribution or distribution family can be assessed by a *quantile-quantile (qq) plot*, the most common of which is the *normal probability plot*.

The [`Gadfly`](https://github.com/GiovineItalia/Gadfly.jl) package for Julia uses a "grammar of graphics" specification, similar to the [`ggplot2`](http://ggplot2.org/) package for R.  A histogram or a kernel density plot are describes as *geometries* and specified by `Geom.histogram` and `Geom.density`, respectively.

````julia
julia> plot(mm1bstp, x = :β₁, histogram)
Plot(...)

````



````julia
julia> plot(mm1bstp, x = :σ, histogram)
Plot(...)

````



````julia
julia> plot(mm1bstp, x = :σ₁, histogram)
Plot(...)

````





The last two histograms show that, even if the models are defined in terms of variances, the variance is usually not a good scale on which to assess the variability of the parameter estimates.  The standard deviation or, in some cases, the logarithm of the standard deviation is a more suitable scale.

The histogram of $\sigma_1^2$ has a "spike" at zero.  Because the value of $\sigma^2$ is never zero, a value of $\sigma_1^2=0$ must correspond to $\theta=0$.  There are several ways to count the zeros in `theta1`.  The simplest is to use `count` on the nonzeros, and subtrack that value from the total number of values in `theta1`.

````julia
julia> length(mm1bstp[:θ₁]) - count(!iszero, mm1bstp[:θ₁])
10090

````





That is, nearly 1/10 of the `theta1` values are zeros.  Because such a spike or pulse will be spread out or diffused in a kernel density plot,

````julia
julia> plot(mm1bstp, density, x = :θ₁)
Plot(...)

````





such a plot is not suitable for a sample of a bounded parameter that includes values on the boundary.

The density of the estimates of the other two parameters, $\beta_1$ and $\sigma$, are depicted well in kernel density plots.

````julia
julia> plot(mm1bstp, density, x = :β₁)
Plot(...)

````



````julia
julia> plot(mm1bstp, density, x = :σ)
Plot(...)

````





The standard approach of summarizing a sample by its mean and standard deviation, or of constructing a confidence interval using the sample mean, the standard error of the mean and quantiles of a *t* or normal distribution, are based on the assumption that the sample is approximately normal (also called Gaussian) in shape.  A *normal probability plot*, which plots sample quantiles versus quantiles of the standard normal distribution, $\mathcal{N}(0,1)$, can be used to assess the validity of this assumption.  If the points fall approximately along a straight line, the assumption of normality should be valid.  Systematic departures from a straight line are cause for concern.

In Gadfly a normal probability plot can be  constructed by specifying the statistic to be generated as `Stat.qq` and either `x` or `y` as the distribution `Normal()`. For the present purposes it is an advantage to put the theoretical quantiles on the `x` axis.

This approach is suitable for small to moderate sample sizes, but not for sample sizes of 10,000.  To smooth the plot and to reduce the size of the plot files, we plot quantiles defined by a sequence of `n` "probability points".  These are constructed by partitioning the interval (0, 1) into `n` equal-width subintervals and returning the midpoint of each of the subintervals.

````julia
julia> function ppoints(n)
    if n ≤ 0
        throw(ArgumentError("n = $n should be positive"))
    end
    width = inv(n)
    width / 2 : width : one(width)
end
ppoints (generic function with 1 method)

julia> const ppt250 = ppoints(250)
0.002:0.004:0.998

````





The kernel density estimate of $\sigma$ is more symmetric

![](./assets/SimpleLMM_53_1.svg)



and the normal probability plot of $\sigma$ is also reasonably straight.

![](./assets/SimpleLMM_54_1.svg)



The normal probability plot of $\sigma_1$ has a flat section at $\sigma_1 = 0$.

![](./assets/SimpleLMM_55_1.svg)



In terms of the variances, $\sigma^2$ and $\sigma_1^2$, the normal probability plots are

![](./assets/SimpleLMM_56_1.svg)

![](./assets/SimpleLMM_57_1.svg)



### Confidence intervals based on bootstrap samples

When the distribution of a parameter estimator is close to normal or to a T distribution, symmetric confidence intervals are an appropriate representation of the uncertainty in the parameter estimates.  However, they are not appropriate for skewed and/or bounded estimator distributions, such as those for $\sigma^2$ and $\sigma_2^1$ shown above.

The fact that a symmetric confidence interval is not appropriate for $\sigma^2$ should not be surprising.  In an introductory statistics course the calculation of a confidence interval on $\sigma^2$ from a random sample of a normal distribution using quantiles of a $\chi^2$ distribution is often introduced. So a symmetric confidence interval on $\sigma^2$ is inappropriate in the simplest case but is often expected to be appropriate in much more complicated cases, as shown by the fact that many statistical software packages include standard errors of variance component estimates in the output from a mixed model fitting procedure.  Creating confidence intervals in this way is optimistic at best. Completely nonsensical would be another way of characterizing this approach.

A more reasonable alternative for constructing a $1 - \alpha$ confidence interval from a bootstrap sample is to report a contiguous interval that contains a $1 - \alpha$ proportion of the sample values.

But there are many such intervals.  Suppose that a 95% confidence interval was to be constructed from one of the samples of size 10,000
of bootstrapped values.  To get a contigous interval the sample should be sorted. The sorted sample values, also called the *order statistics* of the sample, are denoted by a bracketed subscript.  That is, $\sigma_{[1]}$ is the smallest value in the sample, $\sigma_{[2]}$ is the second smallest, up to $\sigma_{[10,000]}$, which is the largest.

One possible interval containing 95% of the sample is $(\sigma_{[1]}, \sigma_{[9500]})$.  Another is $(\sigma_{[2]}, \sigma_{[9501]})$ and so on up to $(\sigma_{[501]},\sigma_{[10000]})$.  There needs to be a method of choosing one of these intervals.  On approach would be to always choose the central 95% of the sample.  That is, cut off 2.5% of the sample on the left side and 2.5% on the right side.  

````julia
julia> sigma95 = quantile(mm1bstp[:σ], [0.025, 0.975])
2-element Array{Float64,1}:
 35.58365989114112 
 63.098968467157626

````





This approach has the advantage that the endpoints of a 95% interval on $\sigma^2$ are the squares of the endpoints of a 95% interval on $\sigma$.

````julia
julia> isapprox(abs2.(sigma95), quantile(abs2.(mm1bstp[:σ]), [0.025, 0.975]))
true

````





The intervals are compared with `isapprox` rather than exact equality because, in floating point arithmetic, it is not always the case that $\left(\sqrt{x}\right)^2 = x$.  This comparison can also be expressed
in Julia as

````julia
julia> abs2.(sigma95) ≈ quantile(abs2.(mm1bstp[:σ]), [0.025, 0.975])
true

````





An alternative approach is to evaluate all of the contiguous intervals containing, say, 95% of the sample and return the shortest shortest such interval.  This is the equivalent of a *Highest Posterior Density (HPD)* interval sometimes used in Bayesian analysis.  If the procedure is applied to a unimodal (i.e. one that has only one peak or *mode*) theoretical probability density the resulting interval has the property that the density at the left endpoint is equal to the density at the right endpoint and that the density at any point outside the interval is less than the density at any point inside the interval.  Establishing this equivalence is left as an exercise for the mathematically inclined reader.  (Hint: Start with the interval defined by the "equal density at the endpoints" property and consider what happens if you shift that interval while maintaining the same area under the density curve.  You will be replacing a region of higher density by one with a lower density and the interval must become wider to maintain the same area.)

With large samples a brute-force enumeration approach works.

````julia
julia> function hpdinterval(v, level=0.95)
    n = length(v)
    if !(0 < level < 1)
        throw(ArgumentError("level = $level must be in (0, 1)"))
    end
    if (lbd = floor(Int, (1 - level) * n)) < 2
        throw(ArgumentError(
            "level = $level is too large from sample size $n"))
    end
    ordstat = sort(v)
    leftendpts = ordstat[1:lbd]
    rtendpts = ordstat[(1 + n - lbd):n]
    (w, ind) = findmin(rtendpts - leftendpts)
    return [leftendpts[ind], rtendpts[ind]]
end
hpdinterval (generic function with 2 methods)

````





For example, the 95% HPD interval calculated from the sample of $\beta_1$ values is

````julia
julia> hpdinterval(mm1bstp[:β₁])
2-element Array{Float64,1}:
 1493.0095069080169
 1562.0769567557636

````





which is very close to the central probability interval of

````julia
julia> quantile(mm1bstp[:β₁], [0.025, 0.975])
2-element Array{Float64,1}:
 1492.848222804122 
 1561.9248094116124

````





because the empirical distribution of the $\beta_1$ sample is very similar to a normal distribution.  In particular, it is more-or-less symmetric and also unimodal.

The HPD interval on $\sigma^2$ is 

````julia
julia> hpdinterval(abs2.(mm1bstp[:σ]))
2-element Array{Float64,1}:
 1162.8123674714316
 3834.319048668444 

````





which is shifted to the left relative to the central probability interval

````julia
julia> quantile(abs2.(mm1bstp[:σ]), [0.025, 0.975])
2-element Array{Float64,1}:
 1266.19685124846  
 3981.4798217228795

````





because the distribution of the $\sigma^2$ sample is skewed to the right.  The HPD interval will truncate the lower density, long, right tail and include more of the higher density, short, left tail.

The HPD interval does not have the property that the endpoints of the interval on $\sigma^2$ are the squares of the endpoints of the intervals on $\sigma$, because "shorter" on the scale of $\sigma$ does not necessarily correspond to shorter on the scale of $\sigma^2$.

````julia
julia> sigma95hpd = hpdinterval(mm1bstp[:σ])
2-element Array{Float64,1}:
 35.425425171170225
 62.887489728787415

````



````julia
julia> abs2.(sigma95hpd)
2-element Array{Float64,1}:
 1254.960748558181 
 3954.8363643883426

````





Finally, a 95% HPD interval on $\sigma_1$ includes the boundary value $\sigma_1=0$.

````julia
julia> hpdinterval(mm1bstp[:σ₁])
2-element Array{Float64,1}:
  0.0             
 54.59857004471835

````





In fact, the confidence level or coverage probability must be rather small before the boundary value is excluded

````julia
julia> hpdinterval(mm1bstp[:σ₁], 0.798)
2-element Array{Float64,1}:
  0.0              
 42.337106279435886

````



````julia
julia> hpdinterval(mm1bstp[:σ₁], 0.799)
2-element Array{Float64,1}:
  0.0             
 42.39434829198112

````





### Empirical cumulative distribution function

The empirical cumulative distribution function (ecdf) of a sample maps the range of the sample onto `[0,1]` by `x → proportion of sample ≤ x`.  In general this is a "step function", which takes jumps of size `1/length(samp)` at each observed sample value.  For large samples, we can plot it as a qq plot where the theoretical quantiles are the probability points and are on the vertical axis.

![](./assets/SimpleLMM_71_1.svg)



The orange lines added to the plot show the construction of the central probability 80% confidence interval on $\sigma_1$ and the red lines show the 80% HPD interval.  Comparing the spacing of the left end points to that of the right end points shows that the HPD interval is shorter, because, in switching from the orange to the red lines, the right end point moves further to the left than does the left end point.

The differences in the widths becomes more dramatic on the scale of $\sigma_1^2$

![](./assets/SimpleLMM_72_1.svg)



ADD: similar analysis for mm2

## Assessing the Random Effects

In Sect. [sec:definitions] we mentioned that what are sometimes called the BLUPs (or best linear unbiased predictors) of the random effects, $\mathcal B$, are the conditional modes evaluated at the parameter estimates, calculated as $\tilde{b}_{\widehat{\theta}}=\Lambda_{\widehat{\theta}}\tilde{u}_{\widehat{\theta}}$.

These values are often considered as some sort of “estimates” of the random effects. It can be helpful to think of them this way but it can also be misleading. As we have stated, the random effects are not, strictly speaking, parameters—they are unobserved random variables. We don’t estimate the random effects in the same sense that we estimate parameters. Instead, we consider the conditional distribution of $\mathcal B$ given the observed data, $(\mathcal B|\mathcal Y=\mathbf  y)$.

Because the unconditional distribution, $\mathcal B\sim\mathcal{N}(\mathbf 
0,\Sigma_\theta)$ is continuous, the conditional distribution, $(\mathcal
B|\mathcal Y=\mathbf  y)$ will also be continuous. In general, the mode of a probability density is the point of maximum density, so the phrase “conditional mode” refers to the point at which this conditional density is maximized. Because this definition relates to the probability model, the values of the parameters are assumed to be known. In practice, of course, we don’t know the values of the parameters (if we did there would be no purpose in forming the parameter estimates), so we use the estimated values of the parameters to evaluate the conditional modes.

Those who are familiar with the multivariate Gaussian distribution may recognize that, because both $\mathcal B$ and $(\mathcal Y|\mathcal B=\mathbf  b)$ are multivariate Gaussian, $(\mathcal B|\mathcal Y=\mathbf  y)$ will also be multivariate Gaussian and the conditional mode will also be the conditional mean of $\mathcal B$, given $\mathcal Y=\mathbf  y$. This is the case for a linear mixed model but it does not carry over to other forms of mixed models. In the general case all we can say about $\tilde{\mathbf 
  u}$ or $\tilde{\mathbf  b}$ is that they maximize a conditional density, which is why we use the term “conditional mode” to describe these values. We will only use the term “conditional mean” and the symbol, $\mathbf \mu$, in reference to $\mathrm{E}(\mathcal Y|\mathcal B=\mathbf  b)$, which is the conditional mean of $\mathcal Y$ given $\mathcal B$, and an important part of the formulation of all types of mixed-effects models.
  
The `ranef` extractor returns the conditional modes.

````julia
julia> ranef(mm1)  # FIXME return an ordered dict
1-element Array{Array{Float64,2},1}:
 [-16.6282 0.369516 … 53.5798 -42.4943]

````





The result is an array of matrices, one for each random effects term in the model.  In this case there is only one matrix because there is only one random-effects term, `(1 | Batch)`, in the model. There is only one row in this matrix because the random-effects term, `(1 | Batch)`, is a simple, scalar term.

To make this more explicit, random-effects terms in the model formula are those that contain the vertical bar () character. The variable is the grouping factor for the random effects generated by this term. An expression for the grouping factor, usually just the name of a variable, occurs to the right of the vertical bar. If the expression on the left of the vertical bar is , as it is here, we describe the term as a _simple, scalar, random-effects term_. The designation “scalar” means there will be exactly one random effect generated for each level of the grouping factor. A simple, scalar term generates a block of indicator columns — the indicators for the grouping factor — in $\mathbf Z$. Because there is only one random-effects term in this model and because that term is a simple, scalar term, the model matrix, 𝐙, for this model is the indicator matrix for the levels of `Batch`.

In the next chapter we fit models with multiple simple, scalar terms and, in subsequent chapters, we extend random-effects terms beyond simple, scalar terms. When we have only simple, scalar terms in the model, each term has a unique grouping factor and the elements of the list returned by can be considered as associated with terms or with grouping factors. In more complex models a particular grouping factor may occur in more than one term, in which case the elements of the list are associated with the grouping factors, not the terms.

Given the data, 𝐲, and the parameter estimates, we can evaluate a measure of the dispersion of $(\mathcal B|\mathcal Y=\mathbf y)$. In the case of a linear mixed model, this is the conditional standard deviation, from which we can obtain a prediction interval. The extractor is named `condVar`.

````julia
julia> condVar(mm1)
1-element Array{Array{Float64,3},1}:
 [362.31]

[362.31]

[362.31]

[362.31]

[362.31]

[362.31]

````





## Chapter Summary

A considerable amount of material has been presented in this chapter, especially considering the word “simple” in its title (it’s the model that is simple, not the material). A summary may be in order.

A mixed-effects model incorporates fixed-effects parameters and random effects, which are unobserved random variables, $\mathcal B$. In a linear mixed model, both the unconditional distribution of $\mathcal B$ and the conditional distribution, $(\mathcal Y|\mathcal B=\mathbf b)$, are multivariate Gaussian distributions. Furthermore, this conditional distribution is a spherical Gaussian with mean, $\mathbf\mu$, determined by the linear predictor, $\mathbf Z\mathbf b+\mathbf X\mathbf\beta$. That is,

\begin{equation*}(\mathcal Y|\mathcal B=\mathbf b)\sim
  \mathcal{N}(\mathbf Z\mathbf b+\mathbf X\mathbf\beta, \sigma^2\mathbf I_n) .\end{equation*}

The unconditional distribution of $\mathcal B$ has mean $\mathbf 0$ and a parameterized $q\times q$ variance-covariance matrix, $\Sigma_\theta$.

In the models we considered in this chapter, $\Sigma_\theta$, is a simple multiple of the identity matrix, $\mathbf I_6$. This matrix is always a multiple of the identity in models with just one random-effects term that is a simple, scalar term. The reason for introducing all the machinery that we did is to allow for more general model specifications.

The maximum likelihood estimates of the parameters are obtained by minimizing the deviance. For linear mixed models we can minimize the profiled deviance, which is a function of $\mathbf\theta$ only, thereby considerably simplifying the optimization problem.

To assess the precision of the parameter estimates, we profile the deviance function with respect to each parameter and apply a signed square root transformation to the likelihood ratio test statistic, producing a profile zeta function for each parameter. These functions provide likelihood-based confidence intervals for the parameters. Profile zeta plots allow us to visually assess the precision of individual parameters. Density plots derived from the profile zeta function provide another way of examining the distribution of the estimators of the parameters.

Prediction intervals from the conditional distribution of the random effects, given the observed data, allow us to assess the precision of the random effects.

# Notation

#### Random Variables

- $\mathcal Y$: The responses ($n$-dimensional Gaussian)

- $\mathcal B$: The random effects on the original scale ($q$-dimensional Gaussian with mean $\mathbf 0$)

- $\mathcal U$: The orthogonal random effects ($q$-dimensional spherical Gaussian)

Values of these random variables are denoted by the corresponding bold-face, lower-case letters: $\mathbf y$, $\mathbf b$ and $\mathbf u$. We observe $\mathbf y$. We do not observe $\mathbf b$ or $\mathbf u$.

#### Parameters in the Probability Model

- $\mathbf\beta$: The $p$-dimension fixed-effects parameter vector.

- $\mathbf\theta$: The variance-component parameter vector. Its (unnamed) dimension is typically very small. Dimensions of 1, 2 or 3 are common in practice.

- $\sigma$: The (scalar) common scale parameter, $\sigma>0$. It is called the common scale parameter because it is incorporated in the variance-covariance matrices of both $\mathcal Y$ and $\mathcal U$.

- $\mathbf\theta$ The "covariance" parameter vector which determines the $q\times q$ lower triangular matrix $\Lambda_\theta$, called the *relative covariance factor*, which, in turn, determines the $q\times q$ sparse, symmetric semidefinite variance-covariance matrix $\Sigma_\theta=\sigma^2\Lambda_\theta\Lambda_\theta'$ that defines the distribution of $\mathcal B$.

#### Model Matrices

- $\mathbf X$: Fixed-effects model matrix of size $n\times p$.

- $\mathbf Z$: Random-effects model matrix of size $n\times q$.

#### Derived Matrices

- $\mathbf L_\theta$: The sparse, lower triangular Cholesky factor of $\Lambda_\theta'\mathbf Z'\mathbf Z\Lambda_\theta+\mathbf I_q$

#### Vectors

In addition to the parameter vectors already mentioned, we define

- $\mathbf y$: the $n$-dimensional observed response vector

- $\mathbf\eta$: the $n$-dimension linear predictor,

\begin{equation*}
  \mathbf{\eta=X\beta+Zb=Z\Lambda_\theta u+X\beta}
\end{equation*}

- $\mathbf\mu$: the $n$-dimensional conditional mean of $\mathcal Y$ given $\mathcal B=\mathbf b$ (or, equivalently, given $\mathcal U=\mathbf u$)

\begin{equation*}
  \mathbf\mu=\mathrm{E}[\mathcal Y|\mathcal B=\mathbf b]=\mathrm{E}[\mathcal Y|\mathcal U=\mathbf u]
\end{equation*}

- $\tilde{u}_\theta$: the $q$-dimensional conditional mode (the value at which the conditional density is maximized) of $\mathcal U$ given $\mathcal Y=\mathbf y$.

## Exercises

These exercises and several others in this book use data sets from the package for . You will need to ensure that this package is installed before you can access the data sets.

To load a particular data set,

or load just the one data set


1. Check the documentation, the structure () and a summary of the `Rail` data (Fig. [fig:Raildot]).

1. Fit a model with as the response and a simple, scalar random-effects term for the variable `Rail`. Create a dotplot of the conditional modes of the random effects.

1. Create a bootstrap simulation from the model and construct 95% bootstrap-based confidence intervals on the parameters. Is the confidence interval on $\sigma_1$ close to being symmetric about the estimate? Is the corresponding interval on $\log(\sigma_1)$ close to being symmetric about its estimate?

1. Create the profile zeta plot for this model. For which parameters are there good normal approximations?

1. Plot the prediction intervals on the random effects from this model. Do any of these prediction intervals contain zero? Consider the relative magnitudes of $\widehat{\sigma_1}$ and $\widehat{\sigma}$ in this model compared to those in model for the data. Should these ratios of $\sigma_1/\sigma$ lead you to expect a different pattern of prediction intervals in this plot than those in Fig. [fig:fm01preddot]?
