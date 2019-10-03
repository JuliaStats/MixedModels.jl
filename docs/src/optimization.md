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
\begin{equation}
\begin{aligned}
  (\mathcal{Y} | \mathcal{B}=\bf{b}) &\sim\mathcal{N}(\bf{ X\beta + Z b},\sigma^2\bf{I})\\\\
  \mathcal{B}&\sim\mathcal{N}(\bf{0},\Sigma_\theta) .
\end{aligned}
\end{equation}
```
The *conditional mean* of $\mathcal Y$, given $\mathcal B=\bf b$, is the *linear predictor*, $\bf X\bf\beta+\bf Z\bf b$, which depends on the $p$-dimensional *fixed-effects parameter*, $\bf \beta$, and on $\bf b$.
The *model matrices*, $\bf X$ and $\bf Z$, of dimension $n\times p$ and $n\times q$, respectively, are determined from the formula for the model and the values of covariates.
Although the matrix $\bf Z$ can be large (i.e. both $n$ and $q$ can be large), it is sparse (i.e. most of the elements in the matrix are zero).

The *relative covariance factor*, $\Lambda_\theta$, is a $q\times q$ lower-triangular matrix, depending on the *variance-component parameter*, $\bf\theta$, and generating the symmetric $q\times q$ variance-covariance matrix, $\Sigma_\theta$, as
```math
\begin{equation}
\Sigma_\theta=\sigma^2\Lambda_\theta\Lambda_\theta'
\end{equation}
```

The *spherical random effects*, $\mathcal{U}\sim\mathcal{N}(\bf{0},\sigma^2\bf{I}_q)$, determine $\mathcal B$ according to
```math
\begin{equation}
\mathcal{B}=\Lambda_\theta\mathcal{U}.
\end{equation}
```
The *penalized residual sum of squares* (PRSS),
```math
\begin{equation}
r^2(\theta,\beta,\bf{u})=\|\bf{y} - \bf{X}\beta -\bf{Z}\Lambda_\theta\bf{u}\|^2+\|\bf{u}\|^2,
\end{equation}
```
is the sum of the residual sum of squares, measuring fidelity of the model to the data, and a penalty on the size of $\bf u$, measuring the complexity of the model.
Minimizing $r^2$ with respect to $\bf u$,
```math
\begin{equation}
r^2_{\beta,\theta} =\min_{\bf{u}}\left(\|\bf{y} -\bf{X}{\beta} -\bf{Z}\Lambda_\theta\bf{u}\|^2+\|\bf{u}\|^2\right)
\end{equation}
```
is a direct (i.e. non-iterative) computation.
The particular method used to solve this generates a *blocked Choleksy factor*, $\bf{L}_\theta$, which is an lower triangular $q\times q$ matrix satisfying
```math
\begin{equation}
\bf{L}_\theta\bf{L}_\theta'=\Lambda_\theta'\bf{Z}'\bf{Z}\Lambda_\theta+\bf{I}_q .
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

Because the conditional mean, $\bf\mu_{\mathcal Y|\mathcal B=\bf b}=\bf
X\bf\beta+\bf Z\Lambda_\theta\bf u$, is a linear function of both $\bf\beta$ and $\bf u$, minimization of the PRSS with respect to both $\bf\beta$ and $\bf u$ to produce
```math
\begin{equation}
r^2_\theta =\min_{{\bf\beta},{\bf u}}\left(\|{\bf y} -{\bf X}{\bf\beta} -{\bf Z}\Lambda_\theta{\bf u}\|^2+\|{\bf u}\|^2\right)
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
\tilde{\bf b}_{\widehat{\theta}}=\Lambda_{\widehat{\theta}}\tilde{\bf u}_{\widehat{\theta}}
\end{equation}
```
are sometimes called the *best linear unbiased predictors* or BLUPs of the random effects.
Although BLUPs an appealing acronym, I don’t find the term particularly instructive (what is a “linear unbiased predictor” and in what sense are these the “best”?) and prefer the term “conditional modes”, because these are the values of $\bf b$ that maximize the density of the conditional distribution $\mathcal{B} | \mathcal{Y} = {\bf y}$.
For a linear mixed model, where all the conditional and unconditional distributions are Gaussian, these values are also the *conditional means*.

## Internal structure of $\Lambda_\theta$ and $\bf Z$

In the types of `LinearMixedModel` available through the `MixedModels` package, groups of random effects and the corresponding columns of the model matrix, $\bf Z$, are associated with *random-effects terms* in the model formula.

For the simple example

````julia
julia> fm1 = fit!(LinearMixedModel(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff]))
Linear mixed model fit by maximum likelihood
 Y ~ 1 + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -163.66353  327.32706  333.32706  337.53065

Variance components:
            Column    Variance  Std.Dev. 
G        (Intercept)  1388.3333 37.260345
Residual              2451.2500 49.510100
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
──────────────────────────────────────────────────
             Estimate  Std.Error  z value  P(>|z|)
──────────────────────────────────────────────────
(Intercept)    1527.5    17.6946   86.326   <1e-99
──────────────────────────────────────────────────

````




the only random effects term in the formula is `(1|G)`, a simple, scalar random-effects term.
````julia
julia> t1 = first(fm1.reterms)
30×6 ReMat{Float64,1}:
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




```@docs
ReMat
```

This `RandomEffectsTerm` contributes a block of columns to the model matrix $\bf Z$ and a diagonal block to $\Lambda_\theta$.
In this case the diagonal block of $\Lambda_\theta$ (which is also the only block) is a multiple of the $6\times6$
identity matrix where the multiple is
````julia
julia> t1.λ
1×1 LinearAlgebra.LowerTriangular{Float64,Array{Float64,2}}:
 0.7525806757718846

````





Because there is only one random-effects term in the model, the matrix $\bf Z$ is the indicators matrix shown as the result of `Matrix(t1)`, but stored in a special sparse format.
Furthermore, there is only one block in $\Lambda_\theta$.


For a vector-valued random-effects term, as in
````julia
julia> fm2 = fit!(LinearMixedModel(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy]))
Linear mixed model fit by maximum likelihood
 Y ~ 1 + U + (1 + U | G)
   logLik   -2 logLik     AIC        BIC    
 -875.96967 1751.93934 1763.93934 1783.09709

Variance components:
            Column    Variance  Std.Dev.   Corr.
G        (Intercept)  565.51069 23.780469
         U             32.68212  5.716828  0.08
Residual              654.94145 25.591824
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
───────────────────────────────────────────────────
             Estimate  Std.Error   z value  P(>|z|)
───────────────────────────────────────────────────
(Intercept)  251.405     6.63226  37.9064    <1e-99
U             10.4673    1.50224   6.96781   <1e-11
───────────────────────────────────────────────────

````




the model matrix $\bf Z$ for is of the form
````julia
julia> t21 = first(fm2.reterms);

julia> convert(Array{Int}, Matrix(t21))  # convert to integers for more compact printing
180×36 Array{Int64,2}:
 1  0  0  0  0  0  0  0  0  0  0  0  0  …  0  0  0  0  0  0  0  0  0  0  0  0
 1  1  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 1  2  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 1  3  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 1  4  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 1  5  0  0  0  0  0  0  0  0  0  0  0  …  0  0  0  0  0  0  0  0  0  0  0  0
 1  6  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 1  7  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 1  8  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 1  9  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
 ⋮              ⋮              ⋮        ⋱     ⋮              ⋮              ⋮
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  1
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  2
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  3
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  4
 0  0  0  0  0  0  0  0  0  0  0  0  0  …  0  0  0  0  0  0  0  0  0  0  1  5
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  6
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  7
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  8
 0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  9

````




and $\Lambda_\theta$ is a $36\times36$ block diagonal matrix with $18$ diagonal blocks, all of the form
````julia
julia> t21.λ
2×2 LinearAlgebra.LowerTriangular{Float64,Array{Float64,2}}:
 0.929221    ⋅      
 0.0181684  0.222645

````




The $\theta$ vector is
````julia
julia> MixedModels.getθ(t21)
3-element Array{Float64,1}:
 0.9292213288149662  
 0.018168393450877257
 0.22264486671069741 

````





Random-effects terms in the model formula that have the same grouping factor are amagamated into a single `ReMat` object.
````julia
julia> fm3 = fit!(zerocorr!(LinearMixedModel(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy]), [:G]))
Linear mixed model fit by maximum likelihood
 Y ~ 1 + U + (1 + U | G)
   logLik   -2 logLik     AIC        BIC    
 -876.00163 1752.00326 1762.00326 1777.96804

Variance components:
            Column    Variance  Std.Dev.   Corr.
G        (Intercept)  584.258970 24.17145
         U             33.632805  5.79938  0.00
Residual              653.115782 25.55613
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
───────────────────────────────────────────────────
             Estimate  Std.Error   z value  P(>|z|)
───────────────────────────────────────────────────
(Intercept)  251.405     6.70771  37.48      <1e-99
U             10.4673    1.51931   6.88951   <1e-11
───────────────────────────────────────────────────

julia> t31 = first(fm3.reterms)
180×36 ReMat{Float64,2}:
 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  2.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  3.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  4.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  5.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  6.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  7.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  8.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  9.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 ⋮                        ⋮              ⋱       ⋮                        ⋮  
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  1.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  2.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  3.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  4.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  1.0  5.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  6.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  7.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  8.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  9.0

````




Note that we could also have achieved this by re-fitting (a copy of) `fm2`.
````julia
julia> fm3alt = zerocorr!(deepcopy(fm2), [:G])
Linear mixed model fit by maximum likelihood
 Y ~ 1 + U + (1 + U | G)
   logLik   -2 logLik     AIC        BIC    
 -875.96967 1751.93934 1761.93934 1777.90413

Variance components:
            Column    Variance  Std.Dev.   Corr.
G        (Intercept)  654.94145 25.591824
         U            654.94145 25.591824  0.00
Residual              654.94145 25.591824
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
───────────────────────────────────────────────────
             Estimate  Std.Error   z value  P(>|z|)
───────────────────────────────────────────────────
(Intercept)  251.405     6.63226  37.9064    <1e-99
U             10.4673    1.50224   6.96781   <1e-11
───────────────────────────────────────────────────

````




For this model the matrix $\bf Z$ is the same as that of model `fm2` but the diagonal blocks of $\Lambda_\theta$ are themselves diagonal.
````julia
julia> t31.λ
2×2 LinearAlgebra.LowerTriangular{Float64,Array{Float64,2}}:
 0.945818   ⋅      
 0.0       0.226927

julia> MixedModels.getθ(t31)
2-element Array{Float64,1}:
 0.9458180658294862 
 0.22692714882505358

````




Random-effects terms with distinct grouping factors generate distinct elements of the `trms` member of the `LinearMixedModel` object.
Multiple `ReMat` objects are sorted by decreasing numbers of random effects.
````julia
julia> fm4 = fit!(LinearMixedModel(@formula(Y ~ 1 + (1|H) + (1|G)), dat[:Penicillin]))
Linear mixed model fit by maximum likelihood
 Y ~ 1 + (1 | H) + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -166.09417  332.18835  340.18835  352.06760

Variance components:
            Column    Variance   Std.Dev. 
G        (Intercept)  0.71497949 0.8455646
H        (Intercept)  3.13519360 1.7706478
Residual              0.30242640 0.5499331
 Number of obs: 144; levels of grouping factors: 24, 6

  Fixed-effects parameters:
──────────────────────────────────────────────────
             Estimate  Std.Error  z value  P(>|z|)
──────────────────────────────────────────────────
(Intercept)   22.9722   0.744596  30.8519   <1e-99
──────────────────────────────────────────────────

julia> t41 = first(fm4.reterms)
144×24 ReMat{Float64,1}:
 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
 ⋮                        ⋮              ⋱                 ⋮                 
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  1.0

julia> t42 = last(fm4.reterms)
144×6 ReMat{Float64,1}:
 1.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0
 1.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0
 ⋮                        ⋮  
 0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0
 1.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0

````




Note that the first `ReMat` in `fm4.terms` corresponds to grouping factor `G` even though the term `(1|G)` occurs in the formula after `(1|H)`.

### Progress of the optimization

An optional named argument, `verbose=true`, in the call to `fit!` of a `LinearMixedModel` causes printing of the objective and the $\theta$ parameter at each evaluation during the optimization.
````julia
julia> fit!(LinearMixedModel(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff]), verbose=true);
f_1: 327.76702 [1.0]
f_2: 331.03619 [1.75]
f_3: 330.64583 [0.25]
f_4: 327.69511 [0.9761896354666064]
f_5: 327.56631 [0.9285689063998191]
f_6: 327.3826 [0.8333274482662446]
f_7: 327.35315 [0.8071883308443906]
f_8: 327.34663 [0.7996883308443905]
f_9: 327.341 [0.7921883308443906]
f_10: 327.33253 [0.7771883308443905]
f_11: 327.32733 [0.7471883308443905]
f_12: 327.32862 [0.7396883308443906]
f_13: 327.32706 [0.7527765100479509]
f_14: 327.32707 [0.7535265100479508]
f_15: 327.32706 [0.7525837539403791]
f_16: 327.32706 [0.7525087539403791]
f_17: 327.32706 [0.7525912539403792]
f_18: 327.32706 [0.7525806757718846]

julia> fit!(LinearMixedModel(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy]), verbose=true);
f_1: 1784.6423 [1.0, 0.0, 1.0]
f_2: 1790.12564 [1.75, 0.0, 1.0]
f_3: 1798.99962 [1.0, 1.0, 1.0]
f_4: 1803.8532 [1.0, 0.0, 1.75]
f_5: 1800.61398 [0.25, 0.0, 1.0]
f_6: 1798.60463 [1.0, -1.0, 1.0]
f_7: 1752.26074 [1.0, 0.0, 0.25]
f_8: 1797.58769 [1.1832612965367755, -0.008661887958064992, 0.0]
f_9: 1754.95411 [1.075, 0.0, 0.32499999999999996]
f_10: 1753.69568 [0.8166315695342949, 0.01116725445704404, 0.28823768689699913]
f_11: 1754.817 [1.0, -0.07071067811865475, 0.19696699141100893]
f_12: 1753.10673 [0.9436827046396457, 0.06383542916420701, 0.26269630296461177]
f_13: 1752.93938 [0.9801419885634466, -0.026656844944287855, 0.274742756095367]
f_14: 1752.25688 [0.9843428851817883, -0.013234749183441809, 0.24719098754484073]
f_15: 1752.05745 [0.9731403970871907, 0.002537849230181556, 0.23791031400587592]
f_16: 1752.02239 [0.9545259030381867, 0.003864210618257173, 0.23589201227534254]
f_17: 1752.02273 [0.9359285300962821, 0.0013317973239056733, 0.23444534669730788]
f_18: 1751.97169 [0.9549646039744493, 0.007906642484926975, 0.22904616789149207]
f_19: 1751.9526 [0.9533132639146138, 0.01662737064088488, 0.22576831302914813]
f_20: 1751.94852 [0.9469287318355107, 0.013076079975639282, 0.2228711267476747]
f_21: 1751.98718 [0.933417530315753, 0.006137673822980371, 0.2189509416721122]
f_22: 1751.98321 [0.9515444328101811, 0.005788999143094354, 0.22061819881026687]
f_23: 1751.95197 [0.9528093408803242, 0.01903319207431438, 0.22417760902815392]
f_24: 1751.94628 [0.9463215304085687, 0.01537385874331431, 0.22508817725341773]
f_25: 1751.9467 [0.9471235457915146, 0.014889405821099199, 0.22489234773246897]
f_26: 1751.94757 [0.9464970169172185, 0.015464270386082009, 0.22581419823185855]
f_27: 1751.94531 [0.9460858412855699, 0.015793369914856737, 0.22444946254845544]
f_28: 1751.94418 [0.945303692531435, 0.016690245680016, 0.22336052906852796]
f_29: 1751.94353 [0.9440720726903343, 0.01721060639973429, 0.22271587718885164]
f_30: 1751.94244 [0.9412710977046074, 0.016309946279096876, 0.22252263173656095]
f_31: 1751.94217 [0.9390004003055789, 0.015899017054786614, 0.2221319769489004]
f_32: 1751.94237 [0.9389790833997584, 0.016547964752330602, 0.2215617504914134]
f_33: 1751.94228 [0.938862818971217, 0.015246587950702079, 0.22268346178551512]
f_34: 1751.9422 [0.938268796248222, 0.015732967012906746, 0.22202359841537422]
f_35: 1751.94131 [0.9388391785671909, 0.016637330586219835, 0.2226114401240285]
f_36: 1751.94093 [0.9383965535141244, 0.017396535224413524, 0.2228172622338821]
f_37: 1751.94057 [0.9370059169178222, 0.018044488666758403, 0.2225344764557106]
f_38: 1751.94018 [0.9341094754153966, 0.018735420363947965, 0.2219495861627497]
f_39: 1751.94008 [0.9326416022319411, 0.018924172739422963, 0.22172575672064995]
f_40: 1751.94027 [0.9313571469210199, 0.01900817557091082, 0.22130945750747505]
f_41: 1751.9415 [0.9328207219136126, 0.02064542594163838, 0.22136730162142598]
f_42: 1751.93949 [0.9318674786371226, 0.01795736847676912, 0.22256364559268693]
f_43: 1751.93939 [0.9291674238866929, 0.0177824259865631, 0.22253384466220905]
f_44: 1751.9394 [0.929658748765743, 0.017772089175303034, 0.2225084400494112]
f_45: 1751.93943 [0.9291934372689493, 0.018780633556338307, 0.2225704230812979]
f_46: 1751.93935 [0.9289856088635435, 0.01823660205149773, 0.22248440134193276]
f_47: 1751.93949 [0.928697073161124, 0.018293692761595176, 0.22317535268936314]
f_48: 1751.93936 [0.9282426233173936, 0.018269520863927025, 0.22258371360750526]
f_49: 1751.93934 [0.9291127571124966, 0.018179125544642807, 0.22262389052542791]
f_50: 1751.93934 [0.9291906092835129, 0.018165755184528475, 0.22264320562154147]
f_51: 1751.93935 [0.9292543151589757, 0.01820927162948151, 0.22262081442060191]
f_52: 1751.93935 [0.9291892385976166, 0.01812979583813873, 0.2225732358654912]
f_53: 1751.93934 [0.9292535718810557, 0.018167626071831575, 0.22264990358023895]
f_54: 1751.93934 [0.9292145142782471, 0.018171737475469886, 0.22264674339201962]
f_55: 1751.93934 [0.9292084149211998, 0.018171474941281403, 0.22264619690675266]
f_56: 1751.93934 [0.9292093075508596, 0.018172964644914476, 0.22265206249870564]
f_57: 1751.93934 [0.9292213288149662, 0.018168393450877257, 0.22264486671069741]

````





A shorter summary of the optimization process is always available as an
```@docs
OptSummary
```
object, which is the `optsum` member of the `LinearMixedModel`.
````julia
julia> fm2.optsum
Initial parameter vector: [1.0, 0.0, 1.0]
Initial objective value:  1784.642296192471

Optimizer (from NLopt):   LN_BOBYQA
Lower bounds:             [0.0, -Inf, 0.0]
ftol_rel:                 1.0e-12
ftol_abs:                 1.0e-8
xtol_rel:                 0.0
xtol_abs:                 [1.0e-10, 1.0e-10, 1.0e-10]
initial_step:             [0.75, 1.0, 0.75]
maxfeval:                 -1

Function evaluations:     57
Final parameter vector:   [0.9292213288149662, 0.018168393450877257, 0.22264486671069741]
Final objective value:    1751.9393444647023
Return code:              FTOL_REACHED


````





### Modifying the optimization process

The `OptSummary` object contains both input and output fields for the optimizer.
To modify the optimization process the input fields can be changed after constructing the model but before fitting it.

Suppose, for example, that the user wishes to try a [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) optimization method instead of the default [`BOBYQA`](https://en.wikipedia.org/wiki/BOBYQA) (Bounded Optimization BY Quadratic Approximation) method.
````julia
julia> fm2 = LinearMixedModel(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy]);

julia> fm2.optsum.optimizer = :LN_NELDERMEAD;

julia> fit!(fm2)
Linear mixed model fit by maximum likelihood
 Y ~ 1 + U + (1 + U | G)
   logLik   -2 logLik     AIC        BIC    
 -875.96967 1751.93934 1763.93934 1783.09709

Variance components:
            Column    Variance   Std.Dev.   Corr.
G        (Intercept)  565.528831 23.780850
         U             32.681047  5.716734  0.08
Residual              654.941678 25.591828
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
──────────────────────────────────────────────────
             Estimate  Std.Error  z value  P(>|z|)
──────────────────────────────────────────────────
(Intercept)  251.405     6.63233  37.906    <1e-99
U             10.4673    1.50222   6.9679   <1e-11
──────────────────────────────────────────────────

julia> fm2.optsum
Initial parameter vector: [1.0, 0.0, 1.0]
Initial objective value:  1784.642296192471

Optimizer (from NLopt):   LN_NELDERMEAD
Lower bounds:             [0.0, -Inf, 0.0]
ftol_rel:                 1.0e-12
ftol_abs:                 1.0e-8
xtol_rel:                 0.0
xtol_abs:                 [1.0e-10, 1.0e-10, 1.0e-10]
initial_step:             [0.75, 1.0, 0.75]
maxfeval:                 -1

Function evaluations:     140
Final parameter vector:   [0.9292360739538559, 0.018168794976407835, 0.22264111430139058]
Final objective value:    1751.9393444750306
Return code:              FTOL_REACHED


````





The parameter estimates are quite similar to those using `:LN_BOBYQA` but at the expense of 140 functions evaluations for `:LN_NELDERMEAD` versus 57 for `:LN_BOBYQA`.

See the documentation for the [`NLopt`](https://github.com/JuliaOpt/NLopt.jl) package for details about the various settings.

### Convergence to singular covariance matrices

To ensure identifiability of $\Sigma_\theta=\sigma^2\Lambda_\theta \Lambda_\theta$, the elements of $\theta$ corresponding to diagonal elements of $\Lambda_\theta$ are constrained to be non-negative.
For example, in a trivial case of a single, simple, scalar, random-effects term as in `fm1`, the one-dimensional $\theta$ vector is the ratio of the standard deviation of the random effects to the standard deviation of the response.
It happens that $-\theta$ produces the same log-likelihood but, by convention, we define the standard deviation to be the positive square root of the variance.
Requiring the diagonal elements of $\Lambda_\theta$ to be non-negative is a generalization of using this positive square root.

If the optimization converges on the boundary of the feasible region, that is if one or more of the diagonal elements of $\Lambda_\theta$ is zero at convergence, the covariance matrix $\Sigma_\theta$ will be *singular*.
This means that there will be linear combinations of random effects that are constant.
Usually convergence to a singular covariance matrix is a sign of an over-specified model.

## Generalized Linear Mixed-Effects Models

In a [*generalized linear model*](https://en.wikipedia.org/wiki/Generalized_linear_model) the responses are modelled as coming from a particular distribution, such as `Bernoulli` for binary responses or `Poisson` for responses that represent counts.
The scalar distributions of individual responses differ only in their means, which are determined by a *linear predictor* expression $\eta=\bf X\beta$, where, as before, $\bf X$ is a model matrix derived from the values of covariates and $\beta$ is a vector of coefficients.

The unconstrained components of $\eta$ are mapped to the, possiby constrained, components of the mean response, $\mu$, via a scalar function, $g^{-1}$, applied to each component of $\eta$.
For historical reasons, the inverse of this function, taking components of $\mu$ to the corresponding component of $\eta$ is called the *link function* and more frequently used map from $\eta$ to $\mu$ is the *inverse link*.

A *generalized linear mixed-effects model* (GLMM) is defined, for the purposes of this package, by
```math
\begin{equation}
\begin{aligned}
  (\mathcal{Y} | \mathcal{B}=\bf{b}) &\sim\mathcal{D}(\bf{g^{-1}(X\beta + Z b)},\phi)\\\\
  \mathcal{B}&\sim\mathcal{N}(\bf{0},\Sigma_\theta) .
\end{aligned}
\end{equation}
```
where $\mathcal{D}$ indicates the distribution family parameterized by the mean and, when needed, a common scale parameter, $\phi$.
(There is no scale parameter for `Bernoulli` or for `Poisson`.
Specifying the mean completely determines the distribution.)
```@docs
Bernoulli
Poisson
```

A `GeneralizedLinearMixedModel` object is generated from a formula, data frame and distribution family.

````julia
julia> mdl = GeneralizedLinearMixedModel(@formula(r2 ~ 1 + a + g + b + s + (1|id) + (1|item)),
           dat[:VerbAgg], Bernoulli());

julia> typeof(mdl)
GeneralizedLinearMixedModel{Float64}

````





A separate call to `fit!` is required to fit the model.
This involves optimizing an objective function, the Laplace approximation to the deviance, with respect to the parameters, which are $\beta$, the fixed-effects coefficients, and $\theta$, the covariance parameters.
The starting estimate for $\beta$ is determined by fitting a GLM to the fixed-effects part of the formula

````julia
julia> mdl.β
6-element Array{Float64,1}:
  0.20605302210322762
  0.03994037605114987
  0.23131667674984463
 -0.7941857249205363 
 -1.539188208545692  
 -0.7766556048305915 

````





and the starting estimate for $\theta$, which is a vector of the two standard deviations of the random effects, is chosen to be

````julia
julia> mdl.θ
2-element Array{Float64,1}:
 1.0
 1.0

````





The Laplace approximation to the deviance requires determining the conditional modes of the random effects.
These are the values that maximize the conditional density of the random effects, given the model parameters and the data.
This is done using Penalized Iteratively Reweighted Least Squares (PIRLS).
In most cases PIRLS is fast and stable.
It is simply a penalized version of the IRLS algorithm used in fitting GLMs.

The distinction between the "fast" and "slow" algorithms in the `MixedModels` package (`nAGQ=0` or `nAGQ=1` in `lme4`) is whether the fixed-effects parameters, $\beta$, are optimized in PIRLS or in the nonlinear optimizer.
In a call to the `pirls!` function the first argument is a `GeneralizedLinearMixedModel`, which is modified during the function call.
(By convention, the names of such *mutating functions* end in `!` as a warning to the user that they can modify an argument, usually the first argument.)
The second and third arguments are optional logical values indicating if $\beta$ is to be varied and if verbose output is to be printed.

````julia
julia> pirls!(mdl, true, true)
varyβ = true, obj₀ = 10210.853438905406, β = [0.20605302210322762, 0.03994037605114987, 0.23131667674984463, -0.7941857249205363, -1.539188208545692, -0.7766556048305915]
   1: 8301.483049027265
   2: 8205.604285133919
   3: 8201.89659746689
   4: 8201.848598910705
   5: 8201.848559060705
   6: 8201.848559060621
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)
  Distribution: Bernoulli{Float64}
  Link: LogitLink()

  Deviance: 8201.8486

Variance components:
        Column    Variance  Std.Dev. 
id   (Intercept)  0.8920336 0.9444753
item (Intercept)  0.8920336 0.9444753

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
─────────────────────────────────────────────────────
               Estimate  Std.Error   z value  P(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.218535    0.464651   0.47032   0.6381
a             0.0514385   0.012319   4.17556   <1e-4 
g: M          0.290225    0.140555   2.06485   0.0389
b: scold     -0.979124    0.476395  -2.05528   0.0399
b: shout     -1.95402     0.477182  -4.09491   <1e-4 
s: self      -0.979493    0.389283  -2.51615   0.0119
─────────────────────────────────────────────────────

````



````julia
julia> deviance(mdl)
8201.848559060621

````



````julia
julia> mdl.β
6-element Array{Float64,1}:
  0.21853493716529312 
  0.051438542580811646
  0.29022454166301215 
 -0.979123706190123   
 -1.9540167628141032  
 -0.979492571803745   

````



````julia
julia> mdl.θ # current values of the standard deviations of the random effects
2-element Array{Float64,1}:
 1.0
 1.0

````





If the optimization with respect to $\beta$ is performed within PIRLS then the nonlinear optimization of the Laplace approximation to the deviance requires optimization with respect to $\theta$ only.
This is the "fast" algorithm.
Given a value of $\theta$, PIRLS is used to determine the conditional estimate of $\beta$ and the conditional mode of the random effects, **b**.

````julia
julia> mdl.b # conditional modes of b
2-element Array{Array{Float64,2},1}:
 [-0.6007716038488827 -1.9322680866219626 … -0.1445537397533593 -0.5752238433557015] 
 [-0.18636418747913763 0.18055184071668703 … 0.2820923275093833 -0.22197445972406996]

````



````julia
julia> fit!(mdl, fast=true, verbose=true);
varyβ = true, obj₀ = 10251.003116042972, β = [0.21853493716529312, 0.051438542580811646, 0.29022454166301215, -0.979123706190123, -1.9540167628141032, -0.979492571803745]
   1: 8292.390783437771
   2: 8204.692089323946
   3: 8201.87681054392
   4: 8201.848569551963
   5: 8201.848559060629
   6: 8201.848559060621
f_1: 8201.848559060621 [1.0, 1.0]
varyβ = true, obj₀ = 10565.66000371902, β = [0.21853493716503283, 0.05143854258081699, 0.2902245416630157, -0.9791237061900286, -1.9540167628140865, -0.9794925718036074]
   1: 8356.488185490085
   2: 8200.93270879382
   3: 8190.472561852921
   4: 8190.11991835326
   5: 8190.117817016412
   6: 8190.117816802472
f_2: 8190.117816802472 [1.75, 1.0]
varyβ = true, obj₀ = 10317.246304689328, β = [0.19342058735129392, 0.05738776368387251, 0.3179960395260618, -1.0485464263732682, -2.0960319812113872, -1.0511288016837288]
   1: 8322.599130522533
   2: 8227.32605948889
   3: 8224.479329416501
   4: 8224.45098905339
   5: 8224.450978980776
   6: 8224.45097898077
f_3: 8224.45097898077 [1.0, 1.75]
varyβ = true, obj₀ = 9776.196238005228, β = [0.21876438895170147, 0.051488841002308744, 0.29046070478859926, -0.9797643406720656, -1.9572962794104538, -0.9812079030633857]
   1: 9035.7252403443
   2: 9026.060558389585
   3: 9026.003998095694
   4: 9026.003906404212
   5: 9026.003906403308
f_4: 9026.003906403308 [0.25, 1.0]
varyβ = true, obj₀ = 10149.608772216947, β = [0.19893695106098588, 0.041806707259910605, 0.2418986481305363, -0.815395934043452, -1.633471838666976, -0.8201772064234893]
   1: 8286.346044727727
   2: 8208.252412394584
   3: 8205.812828017322
   4: 8205.793778239711
   5: 8205.793775487933
f_5: 8205.793775487933 [1.0, 0.25]
varyβ = true, obj₀ = 10406.59045641377, β = [0.21728094120284439, 0.05073109226840724, 0.2867962059041711, -0.9699881548970234, -1.9127335969466108, -0.9591823894790034]
   1: 8290.678720764578
   2: 8163.613865769223
   3: 8157.167433289951
   4: 8157.041222813376
   5: 8157.041032394056
   6: 8157.041032392805
f_6: 8157.041032392805 [1.385829382367975, 0.7364567143287959]
varyβ = true, obj₀ = 10334.285609061479, β = [0.20721523412591733, 0.05490998576682396, 0.30652969414240133, -1.0227806700669044, -2.040115543154628, -1.022735054186773]
   1: 8461.553656168611
   2: 8371.201859251127
   3: 8367.771392346047
   4: 8367.724263488171
   5: 8367.724223100673
   6: 8367.724223100575
f_7: 8367.724223100575 [1.3371524465545725, 0.0]
varyβ = true, obj₀ = 10441.992888908886, β = [0.2131077542758985, 0.05262369623724615, 0.2959485161926296, -0.9987562912090396, -1.9339374639156974, -0.9735353380195938]
   1: 8308.813446103562
   2: 8177.318105403589
   3: 8170.433438731547
   4: 8170.289084465014
   5: 8170.288828193685
   6: 8170.288828191519
f_8: 8170.288828191519 [1.4136494266702277, 1.1104233507216976]
varyβ = true, obj₀ = 10394.123754834105, β = [0.20658453714115443, 0.0552133731832785, 0.3079133180419676, -1.0262965361212701, -2.050456621649885, -1.0280362485644565]
   1: 8283.540345224605
   2: 8163.918696581622
   3: 8158.905795827853
   4: 8158.829375849114
   5: 8158.829317592088
   6: 8158.829317591995
f_9: 8158.829317591995 [1.2722464441311776, 0.7628110428959163]
varyβ = true, obj₀ = 10449.122512319105, β = [0.211222824893935, 0.054005484057214115, 0.3023095984920876, -1.012299568744933, -2.019065583937926, -1.0121221456081302]
   1: 8299.982309744813
   2: 8168.539378419906
   3: 8162.061112545691
   4: 8161.933600781367
   5: 8161.933408558586
   6: 8161.933408557298
f_10: 8161.933408557298 [1.4093623667537367, 0.8680844411698712]
varyβ = true, obj₀ = 10414.10793339631, β = [0.20653729580026609, 0.05513368643868066, 0.3075576744964961, -1.0253459434062084, -2.0468884488248453, -1.0261895534473846]
   1: 8286.138316663308
   2: 8161.902045658136
   3: 8156.392925946188
   4: 8156.301068837267
   5: 8156.300980018685
   6: 8156.300980018442
f_11: 8156.300980018442 [1.3269393938015104, 0.7210153433276878]
varyβ = true, obj₀ = 10407.145905747244, β = [0.2092936301084049, 0.054440786160244045, 0.3043450395272363, -1.017410038992177, -2.028913544405111, -1.0170762549016636]
   1: 8284.87780744784
   2: 8161.70870009135
   3: 8156.20852356967
   4: 8156.116764659553
   5: 8156.116675394207
   6: 8156.1166753939615
f_12: 8156.1166753939615 [1.32364939166362, 0.7142754704852168]
varyβ = true, obj₀ = 10404.356312407202, β = [0.20939665067022264, 0.05441123183582883, 0.3042078506972696, -1.0170635346399668, -2.0281059261519285, -1.0166669610427554]
   1: 8284.155879798209
   2: 8161.539014566784
   3: 8156.092168783243
   4: 8156.00215545072
   5: 8156.002070048329
   6: 8156.002070048108
f_13: 8156.002070048108 [1.318465324298537, 0.7088555585344303]
varyβ = true, obj₀ = 10404.532964559618, β = [0.2095672895079697, 0.054366773553632244, 0.30400097123017966, -1.016543388757478, -2.026962379921656, -1.0160886153663389]
   1: 8284.027704597473
   2: 8161.308933576595
   3: 8155.844273740122
   4: 8155.7536767715355
   5: 8155.753590029702
   6: 8155.7535900294715
f_14: 8155.7535900294715 [1.3207169388571065, 0.7017015224791928]
varyβ = true, obj₀ = 10406.080590126852, β = [0.20947609033473094, 0.05438170870904225, 0.3040713812584796, -1.0167138677795677, -2.027198583515515, -1.0162057468818038]
   1: 8283.987555395453
   2: 8160.874305438911
   3: 8155.367283709278
   4: 8155.275310500365
   5: 8155.275220625106
   6: 8155.275220624852
f_15: 8155.275220624852 [1.3263558167617628, 0.6878017722664596]
varyβ = true, obj₀ = 10409.58800683889, β = [0.20925331974761546, 0.05442052407950494, 0.3042539578337662, -1.0171578113767068, -2.027878717803174, -1.016544913381576]
   1: 8284.070875041096
   2: 8160.10415511768
   3: 8154.505123130813
   4: 8154.410093357726
   5: 8154.409996159076
   6: 8154.409996158768
f_16: 8154.409996158768 [1.338585716556117, 0.6604078030210205]
varyβ = true, obj₀ = 10422.1762244059, β = [0.208767985448389, 0.05450401062923641, 0.3046466447026768, -1.0181072531046944, -2.02933490275995, -1.017271875479356]
   1: 8286.148930569396
   2: 8159.408927883495
   3: 8153.501004776194
   4: 8153.3950829175765
   5: 8153.394956151767
   6: 8153.394956151186
f_17: 8153.394956151186 [1.375819293003887, 0.6133582463290808]
varyβ = true, obj₀ = 10427.671854015944, β = [0.20732305195168838, 0.05476609730561899, 0.3058757219213498, -1.0210628480445056, -2.034408391104947, -1.0198199412184]
   1: 8286.691066751973
   2: 8158.869556551054
   3: 8152.851268971594
   4: 8152.741083892395
   5: 8152.740942646417
   6: 8152.740942645647
f_18: 8152.740942645647 [1.3951508993607586, 0.5630963988309517]
varyβ = true, obj₀ = 10412.134716835031, β = [0.2064753179533161, 0.05487438320976242, 0.3063896760690838, -1.0222230837236252, -2.035530371922692, -1.0203755723552719]
   1: 8282.017684913157
   2: 8157.522016815847
   3: 8151.862297991553
   4: 8151.764833400491
   5: 8151.764726543835
   6: 8151.7647265434
f_19: 8151.7647265434 [1.3676323395311074, 0.5091235621209732]
varyβ = true, obj₀ = 10364.111639274586, β = [0.20730625718464057, 0.05461186270260612, 0.30517481958639114, -1.019180421035945, -2.027680778966582, -1.0164075726138482]
   1: 8271.970907129908
   2: 8157.555557753274
   3: 8152.875645706799
   4: 8152.808979106631
   5: 8152.8089359813775
   6: 8152.808935981318
f_20: 8152.808935981318 [1.2677640577720342, 0.4751231463192672]
varyβ = true, obj₀ = 10420.049185026432, β = [0.2106775805665516, 0.05376804338834675, 0.3012295551223652, -1.0092843628508157, -2.0063961814965667, -1.0056787436383596]
   1: 8285.752253360863
   2: 8159.0739780961285
   3: 8152.980811897887
   4: 8152.866577306928
   5: 8152.866415001937
   6: 8152.866415000804
f_21: 8152.866415000804 [1.4147967067149354, 0.4710990870322999]
varyβ = true, obj₀ = 10395.489389328312, β = [0.20541659536895954, 0.054920824377173515, 0.3066277972982445, -1.0225711061466487, -2.0331407851428946, -1.0191682418227233]
   1: 8278.256525896222
   2: 8157.154672639001
   3: 8151.854900401971
   4: 8151.769667985628
   5: 8151.769591419717
   6: 8151.769591419514
f_22: 8151.769591419514 [1.3258860654160562, 0.5275226916683007]
varyβ = true, obj₀ = 10406.847022583019, β = [0.2088636721792782, 0.05430317854243116, 0.3037280282635222, -1.0156611090060117, -2.021140390458354, -1.013109090312085]
   1: 8281.195202179439
   2: 8157.492470480393
   3: 8151.835428349371
   4: 8151.737863139826
   5: 8151.737755041608
   6: 8151.737755041154
f_23: 8151.737755041154 [1.3668062704788007, 0.4986062131354728]
varyβ = true, obj₀ = 10396.623183199756, β = [0.20729825419589779, 0.05459363004474104, 0.3050918349451794, -1.018955366191625, -2.0268543124624085, -1.015990702931356]
   1: 8278.460002110456
   2: 8157.053323937651
   3: 8151.673309975076
   4: 8151.5852461190025
   5: 8151.585161942599
   6: 8151.585161942339
f_24: 8151.585161942339 [1.3397371137798133, 0.4934921982196282]
varyβ = true, obj₀ = 10393.609316251064, β = [0.2082549659762824, 0.05437660298254685, 0.3040778501929497, -1.0164625987923002, -2.0216118170020616, -1.0133460788422102]
   1: 8277.918179767854
   2: 8157.043669989388
   3: 8151.689440593432
   4: 8151.6021477649765
   5: 8151.602064961415
   6: 8151.60206496116
f_25: 8151.60206496116 [1.3375752795401954, 0.4863105220419929]
varyβ = true, obj₀ = 10397.764486560543, β = [0.20830542908591665, 0.054350965531116, 0.3039593106174819, -1.0161551809706721, -2.0207219752077274, -1.0128978657452172]
   1: 8278.93523656
   2: 8157.145230819091
   3: 8151.691194952198
   4: 8151.600591190692
   5: 8151.600500691274
   6: 8151.600500690967
f_26: 8151.600500690967 [1.3469240905706887, 0.4913480516277605]
varyβ = true, obj₀ = 10395.774178197908, β = [0.2079898825146905, 0.05443059796928582, 0.3043309127063365, -1.017080341971736, -2.022785425822423, -1.0139382184731314]
   1: 8278.406326249258
   2: 8157.064425497862
   3: 8151.6719513436765
   4: 8151.583465843251
   5: 8151.583380717896
   6: 8151.583380717631
f_27: 8151.583380717631 [1.3395817610361234, 0.4973372768680226]
varyβ = true, obj₀ = 10395.44394318946, β = [0.20827453095419746, 0.05437984247156581, 0.304092246454986, -1.01650604386414, -2.021839296729242, -1.0134606162646231]
   1: 8278.362655736912
   2: 8157.064544983574
   3: 8151.672160260854
   4: 8151.583675017631
   5: 8151.583589892588
   6: 8151.5835898923215
f_28: 8151.5835898923215 [1.3392718767161806, 0.49802026420923295]
varyβ = true, obj₀ = 10395.535382999502, β = [0.20828805461964026, 0.054378180211730166, 0.30408433495433757, -1.0164879468260317, -2.0218271157127967, -1.0134544452465786]
   1: 8278.380710340909
   2: 8157.0666510451265
   3: 8151.671999799076
   4: 8151.583432800489
   5: 8151.583347442414
   6: 8151.583347442147
f_29: 8151.583347442147 [1.339721081832098, 0.49695538534324185]
varyβ = true, obj₀ = 10395.855932031016, β = [0.2082681763393742, 0.05438050242434848, 0.3040954094451179, -1.016513052233775, -2.0218397675904893, -1.0134608700295284]
   1: 8278.456305937305
   2: 8157.074106468336
   3: 8151.672400464071
   4: 8151.583600201321
   5: 8151.5835143054655
   6: 8151.5835143051945
f_30: 8151.5835143051945 [1.3404085881855257, 0.4972551102404334]
varyβ = true, obj₀ = 10395.450813218806, β = [0.20824473327990112, 0.05438627252764272, 0.30412234667908083, -1.0165800287330231, -2.0219863550427606, -1.0135347946043904]
   1: 8278.353361601025
   2: 8157.0635500366825
   3: 8151.671954135131
   4: 8151.583485621319
   5: 8151.583400470982
   6: 8151.583400470716
f_31: 8151.583400470716 [1.3395728772314808, 0.4962201742195801]
varyβ = true, obj₀ = 10395.726653541109, β = [0.20827079143733107, 0.054378483644543785, 0.30408610909567624, -1.0164886208435944, -2.021763716970704, -1.0134225446121257]
   1: 8278.42617724777
   2: 8157.071791450588
   3: 8151.6723002428325
   4: 8151.583568754748
   5: 8151.583482990036
   6: 8151.583482989767
f_32: 8151.583482989767 [1.340312897109935, 0.4964946742874612]
varyβ = true, obj₀ = 10395.480743444834, β = [0.20824538649023855, 0.05438464054171437, 0.3041148615287681, -1.0165600218527935, -2.021918394381767, -1.013500549394553]
   1: 8278.364600272938
   2: 8157.064659600618
   3: 8151.671928507908
   4: 8151.583425349218
   5: 8151.583340139288
   6: 8151.583340139021
f_33: 8151.583340139021 [1.3395583304714713, 0.4968332751128286]
varyβ = true, obj₀ = 10395.442267530685, β = [0.20827353831169187, 0.054379076960376395, 0.304088764133916, -1.0164964269722128, -2.021801661225427, -1.0134416550289336]
   1: 8278.359702896314
   2: 8157.064704875615
   3: 8151.671931416345
   4: 8151.583426430177
   5: 8151.58334121545
   6: 8151.583341215184
f_34: 8151.583341215184 [1.3395286260143233, 0.4969021419787769]
varyβ = true, obj₀ = 10395.47977474893, β = [0.20827484776301908, 0.05437892187903585, 0.3040880249695567, -1.016494747325385, -2.0218007362745203, -1.01344118546257]
   1: 8278.368369189124
   2: 8157.065584741338
   3: 8151.671959910018
   4: 8151.583426365368
   5: 8151.583341082107
   6: 8151.5833410818395
f_35: 8151.5833410818395 [1.3396255162934776, 0.49686660775930835]
varyβ = true, obj₀ = 10395.478844625672, β = [0.20827126293061596, 0.05437964575020409, 0.3040914187934823, -1.0165030367587211, -2.0218162635245838, -1.0134490187641039]
   1: 8278.367429147484
   2: 8157.065445657713
   3: 8151.671955750737
   4: 8151.583426429477
   5: 8151.583341154114
   6: 8151.583341153846
f_36: 8151.583341153846 [1.3396267569264424, 0.49680256973447084]
varyβ = true, obj₀ = 10395.453884471744, β = [0.20827098617850712, 0.05437958168698981, 0.3040911316853156, -1.0165021974730213, -2.02181225482042, -1.0134469996599562]
   1: 8278.361837062677
   2: 8157.064895678783
   3: 8151.671936855646
   4: 8151.583425364515
   5: 8151.583340132135
   6: 8151.583340131868
f_37: 8151.583340131868 [1.339563899977355, 0.4968327838741215]
varyβ = true, obj₀ = 10395.451247319943, β = [0.20827333787893662, 0.054379120359464864, 0.3040889672849741, -1.0164969265394836, -2.0218026564144926, -1.013442157010629]
   1: 8278.361538691712
   2: 8157.064912191569
   3: 8151.671937454679
   4: 8151.583425366178
   5: 8151.583340132135
   6: 8151.583340131868

````





The optimization process is summarized by

````julia
julia> mdl.LMM.optsum
Initial parameter vector: [1.0, 1.0]
Initial objective value:  8201.848559060621

Optimizer (from NLopt):   LN_BOBYQA
Lower bounds:             [0.0, 0.0]
ftol_rel:                 1.0e-12
ftol_abs:                 1.0e-8
xtol_rel:                 0.0
xtol_abs:                 [1.0e-10, 1.0e-10]
initial_step:             [0.75, 0.75]
maxfeval:                 -1

Function evaluations:     37
Final parameter vector:   [1.339563899977355, 0.4968327838741215]
Final objective value:    8151.583340131868
Return code:              FTOL_REACHED


````





As one would hope, given the name of the option, this fit is comparatively fast.
````julia
julia> @time(fit!(GeneralizedLinearMixedModel(@formula(r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)),
        dat[:VerbAgg], Bernoulli()), fast=true))
  1.584015 seconds (3.08 M allocations: 75.805 MiB, 3.20% gc time)
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)
  Distribution: Bernoulli{Float64}
  Link: LogitLink()

  Deviance: 8151.5833

Variance components:
        Column    Variance   Std.Dev.  
id   (Intercept)  1.63965872 1.28049159
item (Intercept)  0.22555221 0.47492337

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
──────────────────────────────────────────────────────
               Estimate  Std.Error    z value  P(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.208273   0.387547    0.537414   0.5910
a             0.0543791  0.0160145   3.39561    0.0007
g: M          0.304089   0.182791    1.66359    0.0962
b: scold     -1.0165     0.246175   -4.12917    <1e-4 
b: shout     -2.0218     0.247803   -8.15891    <1e-15
s: self      -1.01344    0.201588   -5.02728    <1e-6 
──────────────────────────────────────────────────────

````





The alternative algorithm is to use PIRLS to find the conditional mode of the random effects, given $\beta$ and $\theta$ and then use the general nonlinear optimizer to fit with respect to both $\beta$ and $\theta$.
Because it is slower to incorporate the $\beta$ parameters in the general nonlinear optimization, the fast fit is performed first and used to determine starting estimates for the more general optimization.

````julia
julia> @time mdl1 = fit!(GeneralizedLinearMixedModel(@formula(r2 ~ 1+a+g+b+s+(1|id)+(1|item)),
        dat[:VerbAgg], Bernoulli()))
  4.707377 seconds (12.54 M allocations: 150.400 MiB, 1.68% gc time)
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)
  Distribution: Bernoulli{Float64}
  Link: LogitLink()

  Deviance: 8151.3997

Variance components:
        Column    Variance  Std.Dev. 
id   (Intercept)  1.6436439 1.2820468
item (Intercept)  0.2246580 0.4739810

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
──────────────────────────────────────────────────────
               Estimate  Std.Error    z value  P(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.199067    0.387738   0.513404   0.6077
a             0.0574286   0.016036   3.58122    0.0003
g: M          0.320724    0.183026   1.75233    0.0797
b: scold     -1.0588      0.24575   -4.30846    <1e-4 
b: shout     -2.10541     0.247399  -8.51017    <1e-16
s: self      -1.05545     0.201249  -5.24451    <1e-6 
──────────────────────────────────────────────────────

````





This fit provided slightly better results (Laplace approximation to the deviance of 8151.400 versus 8151.583) but took 6 times as long.
That is not terribly important when the times involved are a few seconds but can be important when the fit requires many hours or days of computing time.

The comparison of the slow and fast fit is available in the optimization summary after the slow fit.

````julia
julia> mdl1.LMM.optsum
Initial parameter vector: [0.20827333787865795, 0.05437912035956125, 0.3040889672854007, -1.0164969265405766, -2.021802656417862, -1.0134421570122811, 1.339563899984453, 0.4968327839056769]
Initial objective value:  8151.583340131868

Optimizer (from NLopt):   LN_BOBYQA
Lower bounds:             [-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, 0.0, 0.0]
ftol_rel:                 1.0e-12
ftol_abs:                 1.0e-8
xtol_rel:                 0.0
xtol_abs:                 [1.0e-10, 1.0e-10]
initial_step:             [0.12918231705368088, 0.005338175496712345, 0.060930228044501794, 0.08205826500272394, 0.08260097852815702, 0.06719613448159588, 0.05, 0.05]
maxfeval:                 -1

Function evaluations:     175
Final parameter vector:   [0.19906651279679713, 0.05742863004160202, 0.3207236109550407, -1.058803035484438, -2.105406894038309, -1.0554511236109871, 1.339708246131341, 0.4952988280474784]
Final objective value:    8151.399719759675
Return code:              FTOL_REACHED


````


