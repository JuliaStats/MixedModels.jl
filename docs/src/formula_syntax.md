
# Formula syntax

MixedModels.jl uses the variant of the Wilkinson-Rogers (1973) notation for models of (co)variance implemented by [StatsModels.jl](https://juliastats.org/StatsModels.jl/stable/formula/#Modeling-tabular-data).
Additionally, MixedModels.jl extends this syntax to use the pipe `|` as the grouping operator.
Further extensions are provided by [RegressionFormulae.jl](https://github.com/kleinschmidt/RegressionFormulae.jl?tab=readme-ov-file), in particular the use of the slash `/` as the nesting operator and the use of the caret `^` to indicate main effects and interactions up to a specified order.
Currently, MixedModels.jl loads RegressionFormulae.jl by default, though this may change in a future release.
If you require specific functionality from RegressionFormulae.jl, it is best to load it directly so that you can control the version used.

## General rules

- "Addition" (`+`) indicates additive, i.e., main effects: `a + b` indicates main effects of `a` and `b`.
- "Multiplication" (`*`) indicates crossing: main effects and interactions between two terms: `a * b` indicates main effects of `a` and `b` as well as their interaction.
- Usual algebraic rules apply (associativity and distributivity):
  - `(a + b) * c` is equivalent to `a * c + b * c`
  - `a * b * c` corresponds to main effects of `a`, `b`, and `c`, as well as all three two-way interactions and the three-way interaction.
- Categorical terms are expanded into the associated indicators/contrast variables. See the [StatsModels.jl documentation on contrasts](https://juliastats.org/StatsModels.jl/stable/contrasts/) for more information.
- Interactions are expressed with the ampersand (`&`). (This is contrast to R, which uses the colon `:` for this operation.). `a&b` is the interaction of `a` and `b`. For categorical terms, appropriate combinations of indicators/contrast variables are generated.
- Tilde (`~`) is used to separate response from predictors.
- The intercept is indicated by `1`.
- `y ~ 1 + (a + b) * c` is read as:
  - The response variable is `y`.
  - The model contains an intercept.
  - The model contains main effects of `a`, `b`, and `c`.
  - The model contains interactions between `a` and `c` and between `b` and `c` but not `a` and `b`.
- An intercept is included by default, i.e. there is an implicit `1 + ` in every formula. The intercept may be suppressed by including a `0 + ` in the formula. (In contrast to R, the use of `-1` is **not** supported.)

### MixedModels.jl provided extensions

- The pipe operator (`|`) indicates grouping or blocking.
- `(1 + a | subject)` indicates "by-subject random effects for the intercept and main effect `a`".
- This is in line with the usual statistical reading of `|` as "conditional on".

### RegressionFormulae.jl provided extensions

- "Exponentiation" (`^`) works like repeated multiplication and generates all multiplicative and additive terms up to the given order.
  - `(a + b + c)^2` generates `a + b + c + a&b + a&c + b&c`, but not `a&b&c`.
  - The presence of interaction terms within the base will result in redundant terms and is currently unsupported.
- `fulldummy(a)` assigns "contrasts" to `a` that include all indicator columns (dummy variables) and an intercept column. The resulting overparameterization is generally useful in the fixed effects only as part of nesting.
- The slash operator (`/`) indicates nesting:
  - `a / b` is read as "`b` is nested within `a`".
  - `a / b` expands to `a + fulldummy(a) & b`.
- It is generally not necessary to specify nesting in the blocking variables, when the inner levels are unique across outer levels. In other words, in a study with children (`C1`, `C2`, etc. ) nested within schools (`S1`, `S2`, etc.),
  - it is not necessary to specify the nesting when `C1` identifies a unique child across schools. In other words, intercept-only random effects terms can be written as `(1|C) + `(1|S)`.
  - it is necessary to specify the nesting when chid identifiers are re-used across schools, e.g. `C1` refers to a child in `S1` and a different child in `S2`. In this case, the nested syntax `(1|S/C)` expands to `(1|S) + (1|S&C)`. The interaction term in the second blocking variable generates unique labels for each child across schools.



## Mixed models in Wilkinson-Rogers and mathematical notation

Models fit with MixedModels.jl are generally linear mixed-effects models with unconstrained random effects covariance matrices and homoskedastic, normally distributed residuals.
Under these assumptions, the model specification

`response ~ 1 + (age + sex) * education * n_children  + (1 | subject)`

corresponds to the statistical model

\begin{align*}
\left(Y |\mathcal{B}=b\right) &\sim N\left(X\beta + Zb, \sigma^2 I \right) \\
\mathcal{B} &\sim N\left(0, G\right)
\end{align*}

for which we wish to obtain the maximum-likelihood estimates for $G$ and thus the fixed-effects $\beta$.

- The model contains no restrictions on $G$, except that it is positive semidefinite.
- The response Y is the value of a given response.
- The fixed-effects design matrix X consists of columns for
  - the intercept, age, sex, education, and number of children (contrast coded as appropriate)
  - the interaction of all lower order terms, excluding interactions between age and sex
- The random-effects design matrix Z includes a column for
  - the intercept for each subject
