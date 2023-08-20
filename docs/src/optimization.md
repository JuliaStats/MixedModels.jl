# Details of the parameter estimation

## The probability model

Maximum likelihood estimates are based on the probability model for the observed responses.
In the probability model the distribution of the responses is expressed as a function of one or more *parameters*.

For a continuous distribution the probability density is a function of the responses, given the parameters.
The *likelihood* function is the same expression as the probability density but regarding the observed values as fixed and the parameters as varying.

In general a mixed-effects model incorporates two random variables: $\mathcal{B}$, the $q$-dimensional vector of random effects, and $\mathcal{Y}$, the $n$-dimensional response vector.
The value, $\bf y$, of $\mathcal{Y}$ is observed; the value, $\bf b$, of $\mathcal{B}$ is not.

## Linear Mixed-Effects Models

In a linear mixed model the unconditional distribution of $\mathcal{B}$ and the conditional distribution, $(\mathcal{Y} | \mathcal{B}=\bf{b})$, are both multivariate Gaussian distributions,
```math
\begin{aligned}
  (\mathcal{Y} | \mathcal{B}=\bf{b}) &\sim\mathcal{N}(\bf{ X\beta + Z b},\sigma^2\bf{I})\\\\
  \mathcal{B}&\sim\mathcal{N}(\bf{0},\Sigma_\theta) .
\end{aligned}
```
The *conditional mean* of $\mathcal Y$, given $\mathcal B=\bf b$, is the *linear predictor*, $\bf X\bf\beta+\bf Z\bf b$, which depends on the $p$-dimensional *fixed-effects parameter*, $\bf \beta$, and on $\bf b$.
The *model matrices*, $\bf X$ and $\bf Z$, of dimension $n\times p$ and $n\times q$, respectively, are determined from the formula for the model and the values of covariates.
Although the matrix $\bf Z$ can be large (i.e. both $n$ and $q$ can be large), it is sparse (i.e. most of the elements in the matrix are zero).

The *relative covariance factor*, $\Lambda_\theta$, is a $q\times q$ lower-triangular matrix, depending on the *variance-component parameter*, $\bf\theta$, and generating the symmetric $q\times q$ variance-covariance matrix, $\Sigma_\theta$, as
```math
\Sigma_\theta=\sigma^2\Lambda_\theta\Lambda_\theta'
```

The *spherical random effects*, $\mathcal{U}\sim\mathcal{N}(\bf{0},\sigma^2\bf{I}_q)$, determine $\mathcal B$ according to
```math
\mathcal{B}=\Lambda_\theta\mathcal{U}.
```
The *penalized residual sum of squares* (PRSS),
```math
r^2(\theta,\beta,\bf{u})=\|\bf{y} - \bf{X}\beta -\bf{Z}\Lambda_\theta\bf{u}\|^2+\|\bf{u}\|^2,
```
is the sum of the residual sum of squares, measuring fidelity of the model to the data, and a penalty on the size of $\bf u$, measuring the complexity of the model.
Minimizing $r^2$ with respect to $\bf u$,
```math
r^2_{\beta,\theta} =\min_{\bf{u}}\left(\|\bf{y} -\bf{X}{\beta} -\bf{Z}\Lambda_\theta\bf{u}\|^2+\|\bf{u}\|^2\right)
```
is a direct (i.e. non-iterative) computation.
The particular method used to solve this generates a *blocked Choleksy factor*, $\bf{L}_\theta$, which is an lower triangular $q\times q$ matrix satisfying
```math
\bf{L}_\theta\bf{L}_\theta'=\Lambda_\theta'\bf{Z}'\bf{Z}\Lambda_\theta+\bf{I}_q .
```
where ${\bf I}_q$ is the $q\times q$ *identity matrix*.

Negative twice the log-likelihood of the parameters, given the data, $\bf y$, is
```math
d({\bf\theta},{\bf\beta},\sigma|{\bf y})
=n\log(2\pi\sigma^2)+\log(|{\bf L}_\theta|^2)+\frac{r^2_{\beta,\theta}}{\sigma^2}.
```
where $|{\bf L}_\theta|$ denotes the *determinant* of ${\bf L}_\theta$.
Because ${\bf L}_\theta$ is triangular, its determinant is the product of its diagonal elements.

Because the conditional mean, $\bf\mu_{\mathcal Y|\mathcal B=\bf b}=\bf
X\bf\beta+\bf Z\Lambda_\theta\bf u$, is a linear function of both $\bf\beta$ and $\bf u$, minimization of the PRSS with respect to both $\bf\beta$ and $\bf u$ to produce
```math
r^2_\theta =\min_{{\bf\beta},{\bf u}}\left(\|{\bf y} -{\bf X}{\bf\beta} -{\bf Z}\Lambda_\theta{\bf u}\|^2+\|{\bf u}\|^2\right)
```
is also a direct calculation.
The values of $\bf u$ and $\bf\beta$ that provide this minimum are called, respectively, the *conditional mode*, $\tilde{\bf u}_\theta$, of the spherical random effects and the conditional estimate, $\widehat{\bf\beta}_\theta$, of the fixed effects.
At the conditional estimate of the fixed effects the objective is
```math
d({\bf\theta},\widehat{\beta}_\theta,\sigma|{\bf y})
=n\log(2\pi\sigma^2)+\log(|{\bf L}_\theta|^2)+\frac{r^2_\theta}{\sigma^2}.
```
Minimizing this expression with respect to $\sigma^2$ produces the conditional estimate
```math
\widehat{\sigma^2}_\theta=\frac{r^2_\theta}{n}
```
which provides the *profiled log-likelihood* on the deviance scale as
```math
\tilde{d}(\theta|{\bf y})=d(\theta,\widehat{\beta}_\theta,\widehat{\sigma}_\theta|{\bf y})
=\log(|{\bf L}_\theta|^2)+n\left[1+\log\left(\frac{2\pi r^2_\theta}{n}\right)\right],
```
a function of $\bf\theta$ alone.

The MLE of $\bf\theta$, written $\widehat{\bf\theta}$, is the value that minimizes this profiled objective.
We determine this value by numerical optimization.
In the process of evaluating $\tilde{d}(\widehat{\theta}|{\bf y})$ we determine $\widehat{\beta}=\widehat{\beta}_{\widehat\theta}$, $\tilde{\bf u}_{\widehat{\theta}}$ and $r^2_{\widehat{\theta}}$, from which we can evaluate $\widehat{\sigma}=\sqrt{r^2_{\widehat{\theta}}/n}$.

The elements of the conditional mode of $\mathcal B$, evaluated at the parameter estimates,
```math
\tilde{\bf b}_{\widehat{\theta}}=\Lambda_{\widehat{\theta}}\tilde{\bf u}_{\widehat{\theta}}
```
are sometimes called the *best linear unbiased predictors* or BLUPs of the random effects.
Although BLUPs an appealing acronym, I don’t find the term particularly instructive (what is a “linear unbiased predictor” and in what sense are these the “best”?) and prefer the term “conditional modes”, because these are the values of $\bf b$ that maximize the density of the conditional distribution $\mathcal{B} | \mathcal{Y} = {\bf y}$.
For a linear mixed model, where all the conditional and unconditional distributions are Gaussian, these values are also the *conditional means*.

## Internal structure of $\Lambda_\theta$ and $\bf Z$

In the types of `LinearMixedModel` available through the `MixedModels` package, groups of random effects and the corresponding columns of the model matrix, $\bf Z$, are associated with *random-effects terms* in the model formula.

For the simple example
```@setup Main
using DisplayAs
```
```@example Main
using BenchmarkTools, DataFrames, MixedModels
```
```@example Main
dyestuff = MixedModels.dataset(:dyestuff)
fm1 = fit(MixedModel, @formula(yield ~ 1 + (1|batch)), dyestuff)
DisplayAs.Text(ans) # hide
```
the only random effects term in the formula is `(1|batch)`, a simple, scalar random-effects term.
```@example Main
t1 = only(fm1.reterms);
Int.(t1)  # convert to integers for more compact display
```

The matrix `t1` is a sparse matrix, meaning that most of the elements are zero, and its transpose is stored in a sparse form.
```@example Main
sparse(t1)'
```
provides a compact representation of the positions of the non-zeros in this matrix.

This `RandomEffectsTerm` contributes a block of columns to the model matrix $\bf Z$ and a diagonal block to $\Lambda_\theta$.
In this case the diagonal block of $\Lambda_\theta$ (which is also the only block) is a multiple of the $6\times6$
identity matrix where the multiple is
```@example Main
t1.λ
```

Because there is only one random-effects term in the model, the matrix $\bf Z$ is the indicators matrix shown as the result of `Int.(t1)`, but stored in a special sparse format.
Furthermore, there is only one block in $\Lambda_\theta$.


For a vector-valued random-effects term, as in
```@example Main
sleepstudy = MixedModels.dataset(:sleepstudy)
fm2 = fit(MixedModel, @formula(reaction ~ 1+days+(1+days|subj)), sleepstudy)
DisplayAs.Text(ans) # hide
```
the model matrix $\bf Z$ is of the form
```@example Main
t21 = only(fm2.reterms);
sparse(t21)'
```
and $\Lambda_\theta$ is a $36\times36$ block diagonal matrix with $18$ diagonal blocks, all of the form
```@example Main
t21.λ
```
The $\theta$ vector is
```@example Main
MixedModels.getθ(t21)
```

Random-effects terms in the model formula that have the same grouping factor are amalgamated into a single `ReMat` object.
```@example Main
fm3 = fit(MixedModel, @formula(reaction ~ 1+days+(1|subj) + (0+days|subj)), sleepstudy)
t31 = only(fm3.reterms);
sparse(t31)'
```

For this model the matrix $\bf Z$ is the same as that of model `fm2` but the diagonal blocks of $\Lambda_\theta$ are themselves diagonal.
```@example Main
t31.λ
```
```@example Main
MixedModels.getθ(t31)
```
Random-effects terms with distinct grouping factors generate distinct elements of the `reterms` field of the `LinearMixedModel` object.
Multiple `ReMat` objects are sorted by decreasing numbers of random effects.
```@example Main
penicillin = MixedModels.dataset(:penicillin)
fm4 = fit(MixedModel,
    @formula(diameter ~ 1 + (1|sample) + (1|plate)),
    penicillin)
sparse(first(fm4.reterms))'
```
```@example Main
sparse(last(fm4.reterms))'
```
Note that the first `ReMat` in `fm4.reterms` corresponds to grouping factor `plate` even though the term `(1|plate)` occurs in the formula after `(1|sample)`.

### Progress of the optimization

By default a progress display is shown when fitting a model that takes a second or more to fit. (The optional named argument, `progress=false`, can be used to suppress this display.) The number of iterations performed, the average time per iteration and the current value of the objective are shown in this display.

After the model has been fit, a summary of the optimization process is available as the `optsum` property of the `LinearMixedModel`.
```@example Main
fm2.optsum
DisplayAs.Text(ans) # hide
```

More detailed information about the intermediate steps of the nonlinear optimizer can be obtained the `fitlog` field.
By default, `fitlog` contains entries for only the initial and final steps, but additional information about every nth step can be obtained with the `thin` keyword-argument to `fit`, `fit!` and `refit!`:

```@example Main
refit!(fm2; thin=1)
fm2.optsum.fitlog[1:10]
DisplayAs.Text(ans) # hide
```

## A blocked Cholesky factor

A `LinearMixedModel` object contains two blocked matrices; a symmetric matrix `A` (only the lower triangle is stored) and a lower-triangular `L` which is the lower Cholesky factor of the updated and inflated `A`.
In versions 4.0.0 and later of `MixedModels` only the blocks in the lower triangle are stored in `A` and `L`, as a `Vector{AbstractMatrix{T}}`.

`BlockDescription` shows the structure of the blocks
```@example Main
BlockDescription(fm2)
DisplayAs.Text(ans) # hide
```

Another change in v4.0.0 and later is that the last row of blocks is constructed from `m.Xymat` which contains the full-rank model matrix `X` with the response `y` concatenated on the right.

The operation of installing a new value of the variance parameters, `θ`, and updating `L`
```@docs
setθ!
updateL!
```
is the central step in evaluating the objective (negative twice the log-likelihood).

Typically, the (1,1) block is the largest block in `A` and `L` and it has a special form, either `Diagonal` or
`UniformBlockDiagonal`
providing a compact representation and fast matrix multiplication or solutions of linear systems of equations.

### Modifying the optimization process

The `OptSummary` object contains both input and output fields for the optimizer.
To modify the optimization process the input fields can be changed after constructing the model but before fitting it.

Suppose, for example, that the user wishes to try a [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) optimization method instead of the default [`BOBYQA`](https://en.wikipedia.org/wiki/BOBYQA) (Bounded Optimization BY Quadratic Approximation) method.
```@example Main
fm2nm = LinearMixedModel(@formula(reaction ~ 1+days+(1+days|subj)), sleepstudy);
fm2nm.optsum.optimizer = :LN_NELDERMEAD;
fit!(fm2nm; thin=1)
fm2nm.optsum
DisplayAs.Text(ans) # hide
```

The parameter estimates are quite similar to those using `:LN_BOBYQA` but at the expense of 140 functions evaluations for `:LN_NELDERMEAD` versus 57 for `:LN_BOBYQA`.
When plotting the progress of the individual fits, it becomes obvious that `:LN_BOBYQA` has fully converged by the time `:LN_NELDERMEAD` begins to approach the optimum.

```@example Main
using Gadfly
nm = fm2nm.optsum.fitlog
bob = fm2.optsum.fitlog
convdf = DataFrame(algorithm=[repeat(["NelderMead"], length(nm));
                           repeat(["BOBYQA"], length(bob))],
                   objective=[last.(nm); last.(bob)],
                   step=[1:length(nm); 1:length(bob)])
plot(convdf, x=:step, y=:objective, color=:algorithm, Geom.line)
```

Run time can be constrained with  `maxfeval` and `maxtime`.

See the documentation for the [`NLopt`](https://github.com/JuliaOpt/NLopt.jl) package for details about the various settings.

### Convergence to singular covariance matrices

To ensure identifiability of $\Sigma_\theta=\sigma^2\Lambda_\theta \Lambda_\theta$, the elements of $\theta$ corresponding to diagonal elements of $\Lambda_\theta$ are constrained to be non-negative.
For example, in a trivial case of a single, simple, scalar, random-effects term as in `fm1`, the one-dimensional $\theta$ vector is the ratio of the standard deviation of the random effects to the standard deviation of the response.
It happens that $-\theta$ produces the same log-likelihood but, by convention, we define the standard deviation to be the positive square root of the variance.
Requiring the diagonal elements of $\Lambda_\theta$ to be non-negative is a generalization of using this positive square root.

If the optimization converges on the boundary of the feasible region, that is if one or more of the diagonal elements of $\Lambda_\theta$ is zero at convergence, the covariance matrix $\Sigma_\theta$ will be *singular*.
This means that there will be linear combinations of random effects that are constant.
Usually convergence to a singular covariance matrix is a sign of an over-specified model.

Singularity can be checked with the `issingular` predicate function.
```@docs
issingular
```
```@example Main
issingular(fm2)
```

## Generalized Linear Mixed-Effects Models

In a [*generalized linear model*](https://en.wikipedia.org/wiki/Generalized_linear_model) the responses are modelled as coming from a particular distribution, such as `Bernoulli` for binary responses or `Poisson` for responses that represent counts.
The scalar distributions of individual responses differ only in their means, which are determined by a *linear predictor* expression $\eta=\bf X\beta$, where, as before, $\bf X$ is a model matrix derived from the values of covariates and $\beta$ is a vector of coefficients.

The unconstrained components of $\eta$ are mapped to the, possibly constrained, components of the mean response, $\mu$, via a scalar function, $g^{-1}$, applied to each component of $\eta$.
For historical reasons, the inverse of this function, taking components of $\mu$ to the corresponding component of $\eta$ is called the *link function* and the more frequently used map from $\eta$ to $\mu$ is the *inverse link*.

A *generalized linear mixed-effects model* (GLMM) is defined, for the purposes of this package, by
```math
\begin{aligned}
  (\mathcal{Y} | \mathcal{B}=\bf{b}) &\sim\mathcal{D}(\bf{g^{-1}(X\beta + Z b)},\phi)\\\\
  \mathcal{B}&\sim\mathcal{N}(\bf{0},\Sigma_\theta) .
\end{aligned}
```
where $\mathcal{D}$ indicates the distribution family parameterized by the mean and, when needed, a common scale parameter, $\phi$.
(There is no scale parameter for `Bernoulli` or for `Poisson`.
Specifying the mean completely determines the distribution.)
```@docs
Bernoulli
Poisson
```

A `GeneralizedLinearMixedModel` object is generated from a formula, data frame and distribution family.

```@example Main
verbagg = MixedModels.dataset(:verbagg)
const vaform = @formula(r2 ~ 1 + anger + gender + btype + situ + (1|subj) + (1|item));
mdl = GeneralizedLinearMixedModel(vaform, verbagg, Bernoulli());
typeof(mdl)
```

A separate call to `fit!` can be used to fit the model.
This involves optimizing an objective function, the Laplace approximation to the deviance, with respect to the parameters, which are $\beta$, the fixed-effects coefficients, and $\theta$, the covariance parameters.
The starting estimate for $\beta$ is determined by fitting a GLM to the fixed-effects part of the formula

```@example Main
mdl.β
```

and the starting estimate for $\theta$, which is a vector of the two standard deviations of the random effects, is chosen to be

```@example Main
mdl.θ
```

The Laplace approximation to the deviance requires determining the conditional modes of the random effects.
These are the values that maximize the conditional density of the random effects, given the model parameters and the data.
This is done using Penalized Iteratively Reweighted Least Squares (PIRLS).
In most cases PIRLS is fast and stable.
It is simply a penalized version of the IRLS algorithm used in fitting GLMs.

The distinction between the "fast" and "slow" algorithms in the `MixedModels` package (`nAGQ=0` or `nAGQ=1` in `lme4`) is whether the fixed-effects parameters, $\beta$, are optimized in PIRLS or in the nonlinear optimizer.
In a call to the `pirls!` function the first argument is a `GeneralizedLinearMixedModel`, which is modified during the function call.
(By convention, the names of such *mutating functions* end in `!` as a warning to the user that they can modify an argument, usually the first argument.)
The second and third arguments are optional logical values indicating if $\beta$ is to be varied and if verbose output is to be printed.

```@example Main
pirls!(mdl, true, false)
DisplayAs.Text(ans) # hide
```

```@example Main
deviance(mdl)
```

```@example Main
mdl.β
```

```@example Main
mdl.θ # current values of the standard deviations of the random effects
```

If the optimization with respect to $\beta$ is performed within PIRLS then the nonlinear optimization of the Laplace approximation to the deviance requires optimization with respect to $\theta$ only.
This is the "fast" algorithm.
Given a value of $\theta$, PIRLS is used to determine the conditional estimate of $\beta$ and the conditional mode of the random effects, **b**.

```@example Main
mdl.b # conditional modes of b
```

```@example Main
fit!(mdl, fast=true);
DisplayAs.Text(ans) # hide
```

The optimization process is summarized by

```@example Main
mdl.LMM.optsum
DisplayAs.Text(ans) # hide
```

As one would hope, given the name of the option, this fit is comparatively fast.
```@example Main
@btime fit(MixedModel, vaform, verbagg, Bernoulli(), fast=true)
DisplayAs.Text(ans) # hide
```

The alternative algorithm is to use PIRLS to find the conditional mode of the random effects, given $\beta$ and $\theta$ and then use the general nonlinear optimizer to fit with respect to both $\beta$ and $\theta$.

```@example Main
mdl1 = @btime fit(MixedModel, vaform, verbagg, Bernoulli())
DisplayAs.Text(ans) # hide
```

This fit provided slightly better results (Laplace approximation to the deviance of 8151.400 versus 8151.583) but took 6 times as long.
That is not terribly important when the times involved are a few seconds but can be important when the fit requires many hours or days of computing time.
