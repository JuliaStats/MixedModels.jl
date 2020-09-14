# Rank deficiency in mixed-effects models

```@setup Main
using MixedModels
```

The *(column) rank* of a matrix refers to the number of independent columns in the matrix.
Clearly, the rank can never be more than the number of columns; however, the rank can be less than the number of columns.
In a regression context, this corresponds to (linear) independency of the predictors.
The simplest case of rank deficiency is a duplicated predictor or a predictor that is exactly a multiple of another predictor.
However, rank deficiency can also arise from missing cells in the experimental design, in which case all values of the corresponding contrast/predictor are constant.
Rank deficiency may also arise as an extreme case of multicollinearity.
In all cases, it is important to remember that rank deficiency may arise numerically, even where it does not occur exactly.

Rank deficiency can occur in two ways in mixed-effects models: in the fixed effects and in the random effects.
The implications of rank deficiency and thus the handling of of it differ between these.

## Fixed effects

Rank deficiency in the fixed effects works much the same way as it does in classical ordinary least squares regression.
If one or more predictors can be expressed as a linear combination of the other columns, then this column is redudant and matrix is rank deficient.
Note however, that the redudant column is not defined uniquely.
For example, in the case that of two columns `a` and `b` where `b = 2a`, then the rank deficiency can be solved by eliminating `a` or `b`.
While we defined `b` here in terms of `a`, it may be that `b` is actually the more 'fundamental' predictor it is `a` that is defined in terms of `b`: `a = 0.5b`.
The user may of course possess this information, but this is not apparent to modelling software.
As such, the handling of rank deficiency in `MixedModels.jl` should not be taken as a replacement for thinking about the nature of the predictors in a given model.

There is a widely accepted convention for how to make the coefficient estimates for these redudant columns well-defined: we set their value to zero and their standard errors to `NaN` (and thus also their $z$ and $p$-values).
In practice this is done via 'pivoting': the effective rank of the model matrix determined and the extra columnns are moved to the end (right side) of the matrix.
In subsequent calculations, these columns are effectively ignored (as their estimates are zero and thus won't contribute to any other computations).
For display purposes, this pivoting is unwound at the end and the zeroed estimates are displayed in the output.

Both the pivoted and unpivoted coefficients are available in MixedModels:

```@docs
fixef
```

### Pivoting is platform dependent
In MixedModels.jl, we use standard numerical techniques to detect rank eficiency.
We currently offer no guarantees as to which exactly of the standard techniques (e.g. pivoted QR decomposition, pivoted Cholesky decomposition).
This should instead be viewed as an implementation detail.
Similarly, we offer no guarentees as to which of columns will be treated as redudant.
This may vary between releases and even between platforms (both in broad strokes "Linux" vs. "Windows" and at the level of which BLAS options are loaded on a given processor architecture) for the same release.
In other words, *you should not rely on the pivoted columns being consistent!*
If consistency in the pivoted columns is important to you, then you should instead determine your rank ahead of time and remove extraneous columns / predictors from your model specification.

This lack of consistency guarantees arises from a more fundamental issue: numeric linear algebra is challenging and sensitive to the underlying floating point operations.
Due to rounding error, floating point arithmetic is not associative:

```@example
0.1 + 0.1 + 0.1 - 0.3 == 0.1 + 0.1 + (0.1 - 0.3)
```

This means that "nearly" / numerically rank deficient matrices may or may not be detected as rank deficient, depending on details of the platform.

Currently, a coarse heuristic is applied to reduce the chance that the intercept column will be pivoted, but even this behavior is not guaranteed.

## Random effects

Rank deficiency presents less of a problem in the random effects than in the fixed effects.
The same shrinkage that moves the conditional modes (group-level predictons) towards the grand mean is more generally a form of *regularization*.
With regularization, we are able to find unique estimates for overparameterized models.
(For more reading on this general idea, see also this [blog post](https://jakevdp.github.io/blog/2015/07/06/model-complexity-myth/) on the model complexity myth.)
In fact, this rank deficiency occurs in the case of a "singular" or "boundary" fit, where one or more of the variance components is estimated to be zero or a correlation is estimated to be exactly ±1 (i.e. a parameter is estimated to be at the boundary.)
(Formally, a singular matrix has no multiplicative inverse, but a matrix is singular if and only if it is rank deficient and this corresponds to the value of one of the random-effects parameters being on the boundary of the parameter space.)

In addition to handling naturally occuring rank deficiency in the random effects, we can also use regularization to fit explicitly overparameterized random effects.
For example, we can use `fulldummy` to fit both an intercept term and $n$ indicator variables in the random effects for a categorical variable with $n$ levels instead of the usual $n-1$ contrasts.

```@example Main
kb07 = MixedModels.dataset(:kb07)
contrasts = Dict(var => HelmertCoding() for var in (:spkr, :prec, :load))
fit(MixedModel, @formula(rt_raw ~ spkr * prec * load + (1|subj) + (1+prec|item)), kb07; contrasts=contrasts)
```

```@example Main
fit(MixedModel, @formula(rt_raw ~ spkr * prec * load + (1|subj) + (1+fulldummy(prec)|item)), kb07; contrasts=contrasts)
```

This may be useful when rePCA suggests a random effects structure larger than merely main effects but smaller than all interaction terms.
This is also simiar to the functionality provided by `dummy` in `lme4`, but as in the difference between `zerocorr` in Julia and `||` in R, there are subtle differences in how this explansion interacts with other terms in the random effects.