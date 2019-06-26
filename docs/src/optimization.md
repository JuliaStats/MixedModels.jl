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




the only random effects term in the formula is `(1|G)`, a simple, scalar random-effects term.
````julia
julia> t1 = fm1.trms[1]
MixedModels.ScalarFactorReTerm{Float64,UInt8}(UInt8[0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02  …  0x05, 0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x06, 0x06, 0x06], ["A", "B", "C", "D", "E", "F"], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], :G, ["(Intercept)"], 0.7525806757718846)

````




```@docs
ScalarFactorReTerm
```

This `ScalarFactorReTerm` contributes a block of columns to the model matrix $\bf Z$ and a diagonal block to $\Lambda_\theta$.
````julia
julia> getθ(t1)
1-element Array{Float64,1}:
 0.7525806757718846

julia> getΛ(t1)
0.7525806757718846

julia> Matrix(t1)
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





Because there is only one random-effects term in the model, the matrix $\bf Z$ is the indicators matrix shown as the result of `Matrix(t1)`, but stored in a special sparse format.
Furthermore, there is only one block in $\Lambda_\theta$.
For a `ScalarFactorReTerm` this block is a multiple of the identity, in this case $0.75258\cdot{\bf I}_6$.


For a vector-valued random-effects term, as in
````julia
julia> fm2 = fit(LinearMixedModel, @formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy])
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


julia> t21 = fm2.trms[1]
MixedModels.VectorFactorReTerm{Float64,UInt8,2}(UInt8[0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01  …  0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12], ["308", "309", "310", "330", "331", "332", "333", "334", "335", "337", "349", "350", "351", "352", "369", "370", "371", "372"], [1.0 1.0 … 1.0 1.0; 0.0 1.0 … 8.0 9.0], [1.0 1.0 … 1.0 1.0; 0.0 1.0 … 8.0 9.0], StaticArrays.SArray{Tuple{2},Float64,1,2}[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0]  …  [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0]], :G, ["(Intercept)", "U"], [2], [0.929221 0.0; 0.0181684 0.222645], [1, 2, 4])

````




the random-effects term `(1+U|G)` generates a
```@docs
VectorFactorReTerm
```
The model matrix $\bf Z$ for this model is
````julia
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
julia> getΛ(t21)
2×2 LinearAlgebra.LowerTriangular{Float64,Array{Float64,2}}:
 0.929221    ⋅      
 0.0181684  0.222645

````




The $\theta$ vector is
````julia
julia> getθ(t21)
3-element Array{Float64,1}:
 0.9292213140364632  
 0.018168394459481596
 0.22264486740050538 

````





Random-effects terms in the model formula that have the same grouping factor are amagamated into a single `VectorFactorReTerm` object.
````julia
julia> fm3 = fit(LinearMixedModel, @formula(Y ~ 1 + U + (1|G) + (0+U|G)), dat[:sleepstudy])
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + U + (1 | G) + ((0 + U) | G)
   logLik   -2 logLik     AIC        BIC    
 -876.00163 1752.00326 1762.00326 1777.96804

Variance components:
              Column    Variance  Std.Dev.   Corr.
 G        (Intercept)  584.258977 24.17145
          U             33.632805  5.79938  0.00
 Residual              653.115782 25.55613
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   251.405   6.70771   37.48  <1e-99
U             10.4673   1.51931 6.88951  <1e-11


julia> t31 = fm3.trms[1]
MixedModels.VectorFactorReTerm{Float64,UInt8,2}(UInt8[0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01  …  0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12], ["308", "309", "310", "330", "331", "332", "333", "334", "335", "337", "349", "350", "351", "352", "369", "370", "371", "372"], [1.0 1.0 … 1.0 1.0; 0.0 1.0 … 8.0 9.0], [1.0 1.0 … 1.0 1.0; 0.0 1.0 … 8.0 9.0], StaticArrays.SArray{Tuple{2},Float64,1,2}[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0]  …  [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0]], :G, ["(Intercept)", "U"], [1, 1], [0.945818 0.0; 0.0 0.226927], [1, 4])

````




For this model the matrix $\bf Z$ is the same as that of model `fm2` but the diagonal blocks of $\Lambda_\theta$ are themselves diagonal.
````julia
julia> getΛ(t31)
2×2 LinearAlgebra.LowerTriangular{Float64,Array{Float64,2}}:
 0.945818   ⋅      
 0.0       0.226927

julia> getθ(t31)
2-element Array{Float64,1}:
 0.9458180716002255
 0.2269271485512249

````




Random-effects terms with distinct grouping factors generate distinct elements of the `trms` member of the `LinearMixedModel` object.
Multiple `AbstractFactorReTerm` (i.e. either a `ScalarFactorReTerm` or a `VectorFactorReTerm`) objects are sorted by decreasing numbers of random effects.
````julia
julia> fm4 = fit!(LinearMixedModel(@formula(Y ~ 1 + (1|H) + (1|G)), dat[:Penicillin]))
Linear mixed model fit by maximum likelihood
 Formula: Y ~ 1 + (1 | H) + (1 | G)
   logLik   -2 logLik     AIC        BIC    
 -166.09417  332.18835  340.18835  352.06760

Variance components:
              Column    Variance   Std.Dev. 
 G        (Intercept)  0.71497949 0.8455646
 H        (Intercept)  3.13519360 1.7706478
 Residual              0.30242640 0.5499331
 Number of obs: 144; levels of grouping factors: 24, 6

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   22.9722  0.744596 30.8519  <1e-99


julia> t41 = fm4.trms[1]
MixedModels.ScalarFactorReTerm{Float64,UInt8}(UInt8[0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02  …  0x17, 0x17, 0x17, 0x17, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18], ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"  …  "o", "p", "q", "r", "s", "t", "u", "v", "w", "x"], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], :G, ["(Intercept)"], 1.5375772433917159)

julia> t42 = fm4.trms[2]
MixedModels.ScalarFactorReTerm{Float64,UInt8}(UInt8[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x01, 0x02, 0x03, 0x04  …  0x03, 0x04, 0x05, 0x06, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06], ["A", "B", "C", "D", "E", "F"], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], :H, ["(Intercept)"], 3.219751343843134)

````




Note that the first `ScalarFactorReTerm` in `fm4.trms` corresponds to grouping factor `G` even though the term `(1|G)` occurs in the formula after `(1|H)`.

### Progress of the optimization

An optional named argument, `verbose=true`, in the call to `fit!` of a `LinearMixedModel` causes printing of the objective and the $\theta$ parameter at each evaluation during the optimization.
````julia
julia> fit!(LinearMixedModel(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff]), verbose=true);
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

julia> fit!(LinearMixedModel(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy]), verbose=true);
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

````





A shorter summary of the optimization process is always available as an
```@docs
OptSummary
```
object, which is the `optsum` member of the `LinearMixedModel`.
````julia
julia> fm2.optsum
Initial parameter vector: [1.0, 0.0, 1.0]
Initial objective value:  1784.6422961924707

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
Final objective value:    1751.93934446471
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
 Formula: Y ~ 1 + U + ((1 + U) | G)
   logLik   -2 logLik     AIC        BIC    
 -875.96967 1751.93934 1763.93934 1783.09709

Variance components:
              Column    Variance   Std.Dev.   Corr.
 G        (Intercept)  565.528831 23.780850
          U             32.681047  5.716734  0.08
 Residual              654.941678 25.591828
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
             Estimate Std.Error z value P(>|z|)
(Intercept)   251.405   6.63233  37.906  <1e-99
U             10.4673   1.50222  6.9679  <1e-11


julia> fm2.optsum
Initial parameter vector: [1.0, 0.0, 1.0]
Initial objective value:  1784.6422961924707

Optimizer (from NLopt):   LN_NELDERMEAD
Lower bounds:             [0.0, -Inf, 0.0]
ftol_rel:                 1.0e-12
ftol_abs:                 1.0e-8
xtol_rel:                 0.0
xtol_abs:                 [1.0e-10, 1.0e-10, 1.0e-10]
initial_step:             [0.75, 1.0, 0.75]
maxfeval:                 -1

Function evaluations:     140
Final parameter vector:   [0.929236, 0.0181688, 0.222641]
Final objective value:    1751.939344475031
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
MixedModels.GeneralizedLinearMixedModel{Float64}

````





A separate call to `fit!` is required to fit the model.
This involves optimizing an objective function, the Laplace approximation to the deviance, with respect to the parameters, which are $\beta$, the fixed-effects coefficients, and $\theta$, the covariance parameters.
The starting estimate for $\beta$ is determined by fitting a GLM to the fixed-effects part of the formula

````julia
julia> mdl.β
6-element Array{Float64,1}:
  0.039940376051149876
 -0.7766556048305918  
 -0.7941857249205364  
  0.23131667674984455 
 -1.5391882085456923  
  0.2060530221032278  

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
varyβ = true
obj₀ = 10210.853438905404
β = [0.0399404, -0.776656, -0.794186, 0.231317, -1.53919, 0.206053]
iter = 1
obj = 8301.483049027265
iter = 2
obj = 8205.604285133919
iter = 3
obj = 8201.89659746689
iter = 4
obj = 8201.848598910705
iter = 5
obj = 8201.848559060705
iter = 6
obj = 8201.848559060621
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  Formula: r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 8201.8486

Variance components:
          Column   Variance Std.Dev. 
 id   (Intercept)         1        1
 item (Intercept)         1        1

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
              Estimate Std.Error  z value P(>|z|)
(Intercept)   0.218535  0.491968 0.444206  0.6569
a            0.0514385 0.0130432  3.94371   <1e-4
g: M          0.290225  0.148818   1.9502  0.0512
b: scold     -0.979124  0.504402 -1.94116  0.0522
b: shout      -1.95402  0.505235 -3.86754  0.0001
s: self      -0.979493  0.412168 -2.37644  0.0175


````



````julia
julia> deviance(mdl)
8201.848559060621

````



````julia
julia> mdl.β
6-element Array{Float64,1}:
  0.05143854258081106
 -0.979492571803745  
 -0.9791237061900818 
  0.29022454166301054
 -1.954016762814084  
  0.21853493716528646

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
 [-0.600772 -1.93227 … -0.144554 -0.575224]
 [-0.186364 0.180552 … 0.282092 -0.221974] 

````



````julia
julia> fit!(mdl, fast=true, verbose=true);
varyβ = true
obj₀ = 10251.003116042968
β = [0.0514385, -0.979493, -0.979124, 0.290225, -1.95402, 0.218535]
iter = 1
obj = 8292.390783437773
iter = 2
obj = 8204.692089323944
iter = 3
obj = 8201.87681054392
iter = 4
obj = 8201.848569551963
iter = 5
obj = 8201.848559060627
iter = 6
obj = 8201.848559060621
varyβ = true
obj₀ = 10251.003116042964
β = [0.0514385, -0.979493, -0.979124, 0.290225, -1.95402, 0.218535]
iter = 1
obj = 8292.390783437771
iter = 2
obj = 8204.692089323944
iter = 3
obj = 8201.87681054392
iter = 4
obj = 8201.848569551963
iter = 5
obj = 8201.848559060627
iter = 6
obj = 8201.848559060621

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

Function evaluations:     1
Final parameter vector:   [1.0, 1.0]
Final objective value:    0.0
Return code:              FORCED_STOP


````





As one would hope, given the name of the option, this fit is comparatively fast.
````julia
julia> @time(fit!(GeneralizedLinearMixedModel(@formula(r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)), 
        dat[:VerbAgg], Bernoulli()), fast=true))
  0.505523 seconds (2.12 M allocations: 24.048 MiB, 1.29% gc time)
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  Formula: r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 8151.5833

Variance components:
          Column    Variance   Std.Dev. 
 id   (Intercept)  1.79443144 1.3395639
 item (Intercept)  0.24684282 0.4968328

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
              Estimate Std.Error  z value P(>|z|)
(Intercept)   0.208273  0.405425 0.513715  0.6075
a            0.0543791 0.0167533  3.24587  0.0012
g: M          0.304089  0.191223  1.59023  0.1118
b: scold       -1.0165  0.257531 -3.94708   <1e-4
b: shout       -2.0218  0.259235 -7.79912  <1e-14
s: self       -1.01344  0.210888 -4.80559   <1e-5


````





The alternative algorithm is to use PIRLS to find the conditional mode of the random effects, given $\beta$ and $\theta$ and then use the general nonlinear optimizer to fit with respect to both $\beta$ and $\theta$.
Because it is slower to incorporate the $\beta$ parameters in the general nonlinear optimization, the fast fit is performed first and used to determine starting estimates for the more general optimization.

````julia
julia> @time mdl1 = fit!(GeneralizedLinearMixedModel(@formula(r2 ~ 1+a+g+b+s+(1|id)+(1|item)), 
        dat[:VerbAgg], Bernoulli()))
 12.563066 seconds (56.25 M allocations: 491.640 MiB, 0.62% gc time)
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  Formula: r2 ~ 1 + a + g + b + s + (1 | id) + (1 | item)
  Distribution: Distributions.Bernoulli{Float64}
  Link: GLM.LogitLink()

  Deviance: 8151.3998

Variance components:
          Column    Variance   Std.Dev.  
 id   (Intercept)  1.79475272 1.33968381
 item (Intercept)  0.24539704 0.49537565

 Number of obs: 7584; levels of grouping factors: 316, 24

Fixed-effects parameters:
              Estimate Std.Error  z value P(>|z|)
(Intercept)   0.197687  0.405188  0.48789  0.6256
a            0.0574436  0.016757  3.42804  0.0006
g: M          0.320874  0.191255  1.67773  0.0934
b: scold      -1.05879  0.256839 -4.12238   <1e-4
b: shout      -2.10547  0.258562   -8.143  <1e-15
s: self        -1.0532   0.21033 -5.00736   <1e-6


````





This fit provided slightly better results (Laplace approximation to the deviance of 8151.400 versus 8151.583) but took 6 times as long.
That is not terribly important when the times involved are a few seconds but can be important when the fit requires many hours or days of computing time.

The comparison of the slow and fast fit is available in the optimization summary after the slow fit.

````julia
julia> mdl1.LMM.optsum
Initial parameter vector: [0.0543791, -1.01344, -1.0165, 0.304089, -2.0218, 0.208273, 1.33956, 0.496833]
Initial objective value:  8151.583340131867

Optimizer (from NLopt):   LN_BOBYQA
Lower bounds:             [-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, 0.0, 0.0]
ftol_rel:                 1.0e-12
ftol_abs:                 1.0e-8
xtol_rel:                 0.0
xtol_abs:                 [1.0e-10, 1.0e-10]
initial_step:             [0.135142, 0.00558444, 0.0637411, 0.0858438, 0.0864116, 0.0702961, 0.05, 0.05]
maxfeval:                 -1

Function evaluations:     976
Final parameter vector:   [0.0574436, -1.0532, -1.05879, 0.320874, -2.10547, 0.197687, 1.33968, 0.495376]
Final objective value:    8151.399838282872
Return code:              FTOL_REACHED


````


