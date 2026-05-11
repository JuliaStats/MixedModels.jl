# Issue 833 and unconstrained optimization

Issue [#833](https://github.com/JuliaStats/MixedModels.jl/issues/833) was one of the motivations for considering unconstrained optimization of the objective, followed by *rectifying* the converged parameter vector to ensure non-negative diagonal elements of $\Lambda$.

The objective is well-defined for negative values on the diagonal of $\Lambda$.
The reason for rectifying the final parameter estimate, which involves changing the signs of any negative diagonal values, plus any other parameters that occur in the same columns of $\Lambda$, is to have a single choice of converged parameter values.

Along the way to MixedModels v5.0.0 we changed to always keeping a `fitlog`, which is why the `fitlog=true` argument in the call to `fit!(_lmm; REML=true)` was removed.
