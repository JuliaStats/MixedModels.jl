# Normalized Gauss-Hermite Quadrature

[*Gaussian Quadrature rules*](https://en.wikipedia.org/wiki/Gaussian_quadrature) provide sets of `x` values, called *abscissae*, and corresponding weights, `w`, to approximate an integral with respect to a *weight function*, $g(x)$.
For a `k`th order rule the approximation is
```math
\int f(x)g(x)\,dx \approx \sum_{i=1}^k w_i f(x_i)
```

For the *Gauss-Hermite* rule the weight function is
```math
g(x) = e^{-x^2}
```

and the domain of integration is $(-\infty, \infty)$.
A slight variation of this is the *normalized Gauss-Hermite* rule for which the weight function is the standard normal density
```math
g(z) = \phi(z) = \frac{e^{-z^2/2}}{\sqrt{2\pi}}
```

Thus, the expected value of $f(z)$, where $\mathcal{Z}\sim\mathscr{N}(0,1)$, is approximated as
```math
\mathbb{E}[f]=\int_{-\infty}^{\infty} f(z) \phi(z)\,dz\approx\sum_{i=1}^k w_i\,f(z_i) .
```

Naturally, there is a caveat. For the approximation to be accurate the function $f(z)$ must behave like a low-order polynomial over the range of interest.
More formally, a `k`th order rule is exact when `f` is a polynomial of order `2k-1` or less. [^1]

## Evaluating the weights and abscissae

In the [*Golub-Welsch algorithm*](https://en.wikipedia.org/wiki/Gaussian_quadrature#The_Golub-Welsch_algorithm) the abscissae for a particular Gaussian quadrature rule are determined as the eigenvalues of a symmetric tri-diagonal matrix and the weights are derived from the squares of the first row of the matrix of eigenvectors.
For a `k`th order normalized Gauss-Hermite rule the tridiagonal matrix has zeros on the diagonal and the square roots of `1:k-1` on the super- and sub-diagonal, e.g.
```@setup Main
using DisplayAs
```
```@example Main
using DataFrames, LinearAlgebra, Gadfly
sym3 = SymTridiagonal(zeros(3), sqrt.(1:2))
ev = eigen(sym3);
ev.values
```
```@example Main
abs2.(ev.vectors[1,:])
```
As a function of `k` this can be written as
```@example Main
function gausshermitenorm(k)
    ev = eigen(SymTridiagonal(zeros(k), sqrt.(1:k-1)))
    ev.values, abs2.(ev.vectors[1,:])
end;
```
providing
```@example Main
gausshermitenorm(3)
```

The weights and positions are often shown as a *lollipop plot*.
For the 9th order rule these are
```@example Main
gh9=gausshermitenorm(9)
plot(x=gh9[1], y=gh9[2], Geom.hair, Geom.point, Guide.ylabel("Weight"), Guide.xlabel(""))
```
Notice that the magnitudes of the weights drop quite dramatically away from zero, even on a logarithmic scale
```@example Main
plot(
    x=gh9[1], y=gh9[2], Geom.hair, Geom.point,
    Scale.y_log2, Guide.ylabel("Weight (log scale)"),
    Guide.xlabel(""),
)
```

The definition of `MixedModels.GHnorm` is similar to the `gausshermitenorm` function with some extra provisions for ensuring symmetry of the abscissae and the weights and for caching values once they have been calculated.
```@docs
GHnorm
```
```@example Main
using MixedModels
GHnorm(3)
```

By the properties of the normal distribution, when $\mathcal{X}\sim\mathscr{N}(\mu, \sigma^2)$
```math
\mathbb{E}[g(x)] \approx \sum_{i=1}^k g(\mu + \sigma z_i)\,w_i
```

For example, $\mathbb{E}[\mathcal{X}^2]$ where $\mathcal{X}\sim\mathcal{N}(2, 3^2)$ is

```@example Main
μ = 2; σ = 3; ghn3 = GHnorm(3);
sum(@. ghn3.w * abs2(μ + σ * ghn3.z))  # should be μ² + σ² = 13
```

(In general a dot, '`.`', after the function name in a function call, as in `abs2.(...)`, or before an operator creates a [*fused vectorized*](https://docs.julialang.org/en/stable/manual/performance-tips/#More-dots:-Fuse-vectorized-operations-1) evaluation in Julia.
The macro `@.` has the effect of vectorizing all operations in the subsequent expression.)

## Application to a model for contraception use

A *binary response* is a "Yes"/"No" type of answer.
For example, in a 1989 fertility survey of women in Bangladesh (reported in [Huq, N. M. and Cleland, J., 1990](https://www.popline.org/node/371841)) one response of interest was whether the woman used artificial contraception.
Several covariates were recorded including the woman's age (centered at the mean), the number of live children the woman has had (in 4 categories: 0, 1, 2, and 3 or more), whether she lived in an urban setting, and the district in which she lived.
The version of the data used here is that used in review of multilevel modeling software conducted by the Center for Multilevel Modelling, currently at University of Bristol (http://www.bristol.ac.uk/cmm/learning/mmsoftware/data-rev.html).
These data are available as the `:contra` dataset.
```@example Main
contra = DataFrame(MixedModels.dataset(:contra))
describe(contra)
```

A smoothed scatterplot of contraception use versus age
```@example Main
plot(contra, x=:age, y=:use, Geom.smooth, Guide.xlabel("Centered age (yr)"),
    Guide.ylabel("Contraception use"))
```
shows that the proportion of women using artificial contraception is approximately quadratic in age.

A model with fixed-effects for age, age squared, number of live children and urban location and with random effects for district, is fit as
```@example Main
const form1 = @formula use ~ 1 + age + abs2(age) + livch + urban + (1|dist);
m1 = fit(MixedModel, form1, contra, Bernoulli(), fast=true)
DisplayAs.Text(ans) # hide
```

For a model such as `m1`, which has a single, scalar random-effects term, the unscaled conditional density of the spherical random effects variable, $\mathcal{U}$,
given the observed data, $\mathcal{Y}=\mathbf{y}_0$, can be expressed as a product of scalar density functions, $f_i(u_i),\; i=1,\dots,q$.
In the PIRLS algorithm, which determines the conditional mode vector, $\tilde{\mathbf{u}}$, the optimization is performed on the *deviance scale*,
```math
D(\mathbf{u})=-2\sum_{i=1}^q \log(f_i(u_i))
```
The objective, $D$, consists of two parts: the sum of the (squared) *deviance residuals*, measuring fidelity to the data, and the squared length of $\mathbf{u}$, which is the penalty.
In the PIRLS algorithm, only the sum of these components is needed.
To use Gauss-Hermite quadrature the contributions of each of the $u_i,\;i=1,\dots,q$ should be separately evaluated.
```@example Main
const devc0 = map!(abs2, m1.devc0, m1.u[1]);  # start with uᵢ²
const devresid = m1.resp.devresid;   # n-dimensional vector of deviance residuals
const refs = only(m1.LMM.reterms).refs;  # n-dimensional vector of indices in 1:q
for (dr, i) in zip(devresid, refs)
    devc0[i] += dr
end
show(devc0)
```

One thing to notice is that, even on the deviance scale, the contributions of different districts can be of different magnitudes.
This is primarily due to different sample sizes in the different districts.
```@example Main
using FreqTables
freqtable(contra, :dist)'
```

Because the first district has one of the largest sample sizes and the third district has the smallest sample size, these two will be used for illustration.
For a range of $u$ values, evaluate the individual components of the deviance and store them in a matrix.
```@example Main
const devc = m1.devc;
const xvals = -5.0:2.0^(-4):5.0;
const uv = vec(m1.u[1]);
const u₀ = vec(m1.u₀[1]);
results = zeros(length(devc0), length(xvals))
for (j, u) in enumerate(xvals)
    fill!(devc, abs2(u))
    fill!(uv, u)
    MixedModels.updateη!(m1)
    for (dr, i) in zip(devresid, refs)
        devc[i] += dr
    end
    copyto!(view(results, :, j), devc)
end
```
A plot of the deviance contribution versus $u_1$
```@example Main
plot(x=xvals, y=view(results, 1, :), Geom.line, Guide.xlabel("u₁"),
    Guide.ylabel("Deviance contribution"))
```
shows that the deviance contribution is very close to a quadratic.
This is also true for $u_3$
```@example Main
plot(x=xvals, y=view(results, 3, :), Geom.line, Guide.xlabel("u₃"),
    Guide.ylabel("Deviance contribution"))
```

The PIRLS algorithm provides the locations of the minima of these scalar functions, stored as
```@example Main
m1.u₀[1]
```
the minima themselves, evaluated as `devc0` above, and a horizontal scale, which is the inverse of diagonal of the Cholesky factor.
As shown below, this is an estimate of the conditional standard deviations of the components of $\mathcal{U}$.
```@example Main
using MixedModels: block
const s = inv.(m1.LMM.L[block(1,1)].diag);
s'
```

The curves can be put on a common scale, corresponding to the standard normal, as
```@example Main
for (j, z) in enumerate(xvals)
    @. uv = u₀ + z * s
    MixedModels.updateη!(m1)
    @. devc = abs2(uv) - devc0
    for (dr, i) in zip(devresid, refs)
        devc[i] += dr
    end
    copyto!(view(results, :, j), devc)
end
```
```@example Main
plot(x=xvals, y=view(results, 1, :), Geom.line,
    Guide.xlabel("Scaled and shifted u₁"),
    Guide.ylabel("Shifted deviance contribution"))
```
```@example Main
plot(x=xvals, y=view(results, 3, :), Geom.line,
    Guide.xlabel("Scaled and shifted u₃"),
    Guide.ylabel("Shifted deviance contribution"))
```

On the original density scale these become
```@example Main
for (j, z) in enumerate(xvals)
    @. uv = u₀ + z * s
    MixedModels.updateη!(m1)
    @. devc = abs2(uv) - devc0
    for (dr, i) in zip(devresid, refs)
        devc[i] += dr
    end
    copyto!(view(results, :, j), @. exp(-devc/2))
end
```
```@example Main
plot(x=xvals, y=view(results, 1, :), Geom.line,
    Guide.xlabel("Scaled and shifted u₁"),
    Guide.ylabel("Conditional density"))
```
```@example Main
plot(x=xvals, y=view(results, 3, :), Geom.line,
    Guide.xlabel("Scaled and shifted u₃"),
    Guide.ylabel("Conditional density"))
```
and the function to be integrated with the normalized Gauss-Hermite rule is
```@example Main
for (j, z) in enumerate(xvals)
    @. uv = u₀ + z * s
    MixedModels.updateη!(m1)
    @. devc = abs2(uv) - devc0
    for (dr, i) in zip(devresid, refs)
        devc[i] += dr
    end
    copyto!(view(results, :, j), @. exp((abs2(z) - devc)/2))
end
```
```@example Main
plot(x=xvals, y=view(results, 1, :), Geom.line,
    Guide.xlabel("Scaled and shifted u₁"), Guide.ylabel("Kernel ratio"))
```
```@example Main
plot(x=xvals, y=view(results, 3, :), Geom.line,
    Guide.xlabel("Scaled and shifted u₃"), Guide.ylabel("Kernel ratio"))
```

[^1]: https://en.wikipedia.org/wiki/Gaussian_quadrature