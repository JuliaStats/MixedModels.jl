# Multimembership models

In an ordinary mixed model, each observation is associated with exactly one
level of each grouping factor: a trial belongs to one subject, a test score to
one student. In a *multimembership* model, an observation may instead be
associated with several levels of a grouping factor, possibly with different
weights: a manuscript has several authors, a student is taught by several
instructors, a patient is treated by several nurses. The random effect for
such an observation is the (weighted) sum of the effects of all the levels it
belongs to.

MixedModels.jl supports multimembership random effects via the `memberships`
keyword argument to the model constructors and to `fit`. The membership
structure is specified as a [`MembershipMatrix`](@ref), a sparse weights
matrix with one row per level of the grouping factor and one column per
observation, along with the names of the levels. The grouping "variable"
named in the formula does *not* need to be a column in the data table — it
serves only to mark where the multimembership term enters the model and to
match it with the corresponding entry in `memberships`.

## Constructing membership matrices

The [`membershipmatrix`](@ref) helper constructs a `MembershipMatrix` from
common representations of memberships. From a vector of delimited strings:

```@example Main
using MixedModels
mm = membershipmatrix(["a,b", "b,c", "a", "c"])
```

or from several columns, each holding at most one membership per observation
(with `missing` for unused slots):

```@example Main
mm2 = membershipmatrix(["a", "b", "a", "c"], ["b", "c", missing, missing])
```

Weights default to the number of times a level is mentioned for an
observation; pass `normalize=true` to rescale each observation's weights to
sum to one. Arbitrary (e.g. continuous) weights can be supplied by
constructing the `MembershipMatrix` directly from a matrix:

```@example Main
using SparseArrays
W = sparse([0.7 0.0 1.0 0.0
            0.3 0.5 0.0 0.0
            0.0 0.5 0.0 1.0])
mm3 = MembershipMatrix(W; levels=["a", "b", "c"])
```

## Fitting a model

We simulate a small example: 500 observations, each belonging to a random
subset of 20 groups.

```@example Main
using DataFrames, Random, StatsModels
rng = MersenneTwister(42)
nobs, ngrps = 500, 20
W = Float64.(rand(rng, ngrps, nobs) .< 0.2)
b = 2 .* randn(rng, ngrps)
x = rand(rng, nobs)
df = DataFrame(; x)
df.y = 1 .+ 2 .* x .+ W'b .+ randn(rng, nobs)
m = fit(MixedModel, @formula(y ~ 1 + x + (1 | members)), df;
        memberships=Dict(:members => W), progress=false)
```

Here `members` is not a column of `df`; the membership matrix supplies both
the structure and the level names. All the usual accessors work:

```@example Main
VarCorr(m)
```

```@example Main
first(raneftables(m).members, 3)
```

Random slopes are supported with the usual formula syntax, e.g.
`(1 + x | members)`, as are generalized linear mixed models:

```@example Main
df.success = rand(rng, nobs) .< 1 ./ (1 .+ exp.(-(x .- 0.5 .+ W' * randn(rng, ngrps))))
mglm = fit(MixedModel, @formula(success ~ 1 + x + (1 | members)), df, Bernoulli();
           memberships=Dict(:members => W), progress=false)
```

A multimembership term with a one-hot membership matrix (each observation
belonging to exactly one level with weight one) is equivalent to an ordinary
random-effects term, which can be a useful sanity check.

## Prediction and simulation

Because the membership structure lives outside the data table, `predict` and
`simulate` with new data also require a `memberships` keyword argument
describing the memberships of the new observations:

```@example Main
newdf = df[1:10, :]
Wnew = W[:, 1:10]
predict(m, newdf; memberships=Dict(:members => Wnew), new_re_levels=:error)
```

`parametricbootstrap`, `refit!`, and `simulate` *without* new data work
unchanged.

## Interactions and nesting

Interactions between a multimembership factor and other grouping factors
cannot be written in the formula. Instead, construct the interaction weights
explicitly with [`interactionweights`](@ref) and use the result as a new
multimembership factor:

```@example Main
grp = repeat(["u", "v"], nobs ÷ 2)
mmint = interactionweights(MembershipMatrix(W), membershipmatrix(grp))
```

## Technical details and limitations

For a multimembership term the matrix ``Z'Z`` no longer has the block-diagonal
structure that MixedModels.jl usually exploits; the corresponding diagonal
block of the Cholesky factor is stored as a *dense* matrix. Fitting therefore
requires ``O(q^2)`` memory and ``O(q^3)`` time per objective evaluation, where
``q`` is the number of levels times the number of random effects per level.
This is fine for hundreds and even a few thousands of levels, but very large
multimembership factors will be slow. Multimembership terms are always placed
*last* in the internal block structure so that other random-effects terms
retain their sparse representation.

Current limitations:

- adaptive Gauss-Hermite quadrature (`nAGQ > 1`) is not available for models
  with multimembership terms; the Laplace approximation (`nAGQ = 1`) is used.
- [`leverage`](@ref) and `cooksdistance` are not yet supported.
- a multimembership grouping factor cannot share its name with another
  grouping factor or with a column of the data table.
