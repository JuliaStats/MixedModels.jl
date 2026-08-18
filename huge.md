┌ Info: block structure
└   model = "elp_ldt"
┌ Info: evaluation cost
└   model = "elp_ldt"
┌ Info: fit
│   model = "elp_ldt"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "elp_ldt"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "elp_ldt"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"

# Optimizer benchmark for `LinearMixedModel`

* julia 1.12.6 on tigerlake (8 threads)
* BLAS: libopenblas64_.so with 4 threads; julia threads: 1
* MixedModels at commit cb56ff6b
* `RFPthreshold` = 1000, reps = 1
* 1 models: elp_ldt

## The model suite

Block types are those of `L`, lower triangle by rows, random-effects terms
only; the fixed-effects row is dense in every model.  `A/L` denotes a block
whose type differs between `A` and `L`.

| model | tier | dataset | n | p | nθ | #RE | RE block sizes | block types of `L` | `L` (MiB) | shape |
|---|---|---|--:|--:|--:|--:|--:|---|--:|---|
| elp_ldt | huge | elp_ldt_trial | 2745952 | 1 | 2 | 2 | 80962,814 | `Diagonal ; Sparse,Diag/Dense` | 38.7 | n = 2745952, 80962 items crossed with 814 subjects; a structurally small `L` whose cost is entirely in the sweep over the observations |

## The cost of one evaluation at the optimum

`objective` is `updateL!` plus the profiled objective, exactly what a
derivative-free optimizer evaluates; both gradient columns include that same
work.  `rel. diff` is the largest relative discrepancy between the analytic
and the ForwardDiff gradient, and `ws` is the size of the reusable workspace
each gradient source allocates once per optimization.  On the smallest models
all three kernels are dominated by fixed per-call overhead rather than by
arithmetic, which is why ForwardDiff can come out ahead there.

| model | nθ | objective (ms) | analytic ∇ (ms) | ∇/obj | ForwardDiff ∇ (ms) | FD/analytic | obj alloc (KiB) | ∇ alloc (KiB) | FD ∇ alloc (KiB) | analytic ws (MiB) | FD ws (MiB) | rel. diff |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| elp_ldt | 2 | 121.206 | 2376.948 | 19.6× | — | — | 0.4 | 1.2 | — | 512.4 | — | — |

## Complete fits

`Δobjective` is measured against the smallest objective reached for that
model, so a positive value means that configuration stopped short of the
best optimum found.  `max|∇|` is the analytic gradient at the returned
optimum on the deviance scale, a scale-free measure of how tightly each
optimizer converged.  ms/eval includes the gradient for the `LD_*` rows.

| model | nθ | configuration | algorithm | feval | time (s) | ms/eval | alloc (MiB) | # allocs | peak RSS (MiB) | objective | Δobjective | max\|∇\| | singular | return |
|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| elp_ldt | 2 | LN_NEWUOA | LN_NEWUOA | 61 | 8.210 | 134.593 | 0.9 | 18752 | 1430.1 | 39994070.9658 | 3.73e-08 | 8.4e-03 |  | FTOL_REACHED |
| elp_ldt | 2 | LN_BOBYQA | LN_BOBYQA | 76 | 9.682 | 127.398 | 0.9 | 19067 | 1483.4 | 39994070.9659 | 1.39e-05 | 6.2e+00 |  | FTOL_REACHED |
| elp_ldt | 2 | LD_LBFGS + analytic | LD_LBFGS | 27 | 58.124 | 2152.750 | 513.3 | 19202 | 2045.7 | 39994070.9658 | 0.00e+00 | 3.9e-04 |  | SUCCESS |

## Summary: time to a complete fit

Each cell is the wall-clock time in seconds and, in parentheses, the speed-up
relative to the fastest derivative-free configuration for that model.  A
value below 1× means the gradient did not pay for itself.

| model | nθ | n | LN_NEWUOA | LN_BOBYQA | LD_LBFGS + analytic |
|---|--:|--:|--:|--:|--:|
| elp_ldt | 2 | 2745952 | 8.210 (1.0×) | 9.682 (0.8×) | 58.124 (0.1×) |

