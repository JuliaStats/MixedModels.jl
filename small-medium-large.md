✖ julia --startup-file=no --project=gradients gradients/optimizer_benchmark.jl --tiers=small,medium,large
┌ Info: block structure
└   model = "sleepstudy1"
┌ Info: block structure
└   model = "sleepstudy_zc"
┌ Info: block structure
└   model = "sleepstudy"
┌ Info: block structure
└   model = "pastes"
┌ Info: block structure
└   model = "penicillin"
┌ Info: block structure
└   model = "oxide"
┌ Info: block structure
└   model = "kb07"
┌ Info: block structure
└   model = "kwdyz11"
┌ Info: block structure
└   model = "kkl15"
┌ Info: block structure
└   model = "mrk17"
┌ Info: block structure
└   model = "insteval"
┌ Info: block structure
└   model = "insteval_fe"
┌ Info: block structure
└   model = "insteval_vec"
┌ Info: block structure
└   model = "d3"
┌ Info: block structure
└   model = "ml1m"
┌ Info: evaluation cost
└   model = "sleepstudy1"
┌ Info: evaluation cost
└   model = "sleepstudy_zc"
┌ Info: evaluation cost
└   model = "sleepstudy"
┌ Info: evaluation cost
└   model = "pastes"
┌ Info: evaluation cost
└   model = "penicillin"
┌ Info: evaluation cost
└   model = "oxide"
┌ Info: evaluation cost
└   model = "kb07"
┌ Info: evaluation cost
└   model = "kwdyz11"
┌ Info: evaluation cost
└   model = "kkl15"
┌ Info: evaluation cost
└   model = "mrk17"
┌ Info: evaluation cost
└   model = "insteval"
┌ Info: evaluation cost
└   model = "insteval_fe"
┌ Info: evaluation cost
└   model = "insteval_vec"
┌ Info: evaluation cost
└   model = "d3"
┌ Info: evaluation cost
└   model = "ml1m"
┌ Info: fit
│   model = "sleepstudy1"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy1"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy1"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy1"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "sleepstudy_zc"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy_zc"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy_zc"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy_zc"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "sleepstudy"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "sleepstudy"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "pastes"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "pastes"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "pastes"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "pastes"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "penicillin"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "penicillin"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "penicillin"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "penicillin"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "oxide"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "oxide"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "oxide"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "oxide"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "kb07"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "kb07"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "kb07"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "kb07"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "kwdyz11"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "kwdyz11"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "kwdyz11"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "kwdyz11"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "kkl15"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "kkl15"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "kkl15"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "kkl15"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "mrk17"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "mrk17"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "mrk17"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "mrk17"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "insteval"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "insteval_fe"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval_fe"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval_fe"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval_fe"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "insteval_vec"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval_vec"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval_vec"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "insteval_vec"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "d3"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "d3"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "d3"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "d3"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"
┌ Info: fit
│   model = "ml1m"
│   optimizer = "LN_NEWUOA"
└   gradient = "analytic"
┌ Info: fit
│   model = "ml1m"
│   optimizer = "LN_BOBYQA"
└   gradient = "analytic"
┌ Info: fit
│   model = "ml1m"
│   optimizer = "LD_LBFGS"
└   gradient = "analytic"
┌ Info: fit
│   model = "ml1m"
│   optimizer = "LD_LBFGS"
└   gradient = "forwarddiff"

# Optimizer benchmark for `LinearMixedModel`

* julia 1.12.6 on tigerlake (8 threads)
* BLAS: libopenblas64_.so with 4 threads; julia threads: 1
* MixedModels at commit cb56ff6b
* `RFPthreshold` = 1000, reps = 1
* 15 models: sleepstudy1, sleepstudy_zc, sleepstudy, pastes, penicillin, oxide, kb07, kwdyz11, kkl15, mrk17, insteval, insteval_fe, insteval_vec, d3, ml1m

## The model suite

Block types are those of `L`, lower triangle by rows, random-effects terms
only; the fixed-effects row is dense in every model.  `A/L` denotes a block
whose type differs between `A` and `L`.

| model | tier | dataset | n | p | nθ | #RE | RE block sizes | block types of `L` | `L` (MiB) | shape |
|---|---|---|--:|--:|--:|--:|--:|---|--:|---|
| sleepstudy1 | small | sleepstudy | 180 | 2 | 1 | 1 | 18 | `Diagonal` | 0.0 | one scalar term, a single parameter (NEWUOA falls back to BOBYQA) |
| sleepstudy_zc | small | sleepstudy | 180 | 2 | 2 | 1 | 36 | `BlkDiag` | 0.0 | one vector-valued term with diagonal Λ |
| sleepstudy | small | sleepstudy | 180 | 2 | 3 | 1 | 36 | `BlkDiag` | 0.0 | one vector-valued term, 2×2 faces |
| pastes | small | pastes | 60 | 1 | 2 | 2 | 30,10 | `Diagonal ; Sparse,Diagonal` | 0.0 | nested scalar terms, sparse off-diagonal block |
| penicillin | small | penicillin | 144 | 1 | 2 | 2 | 24,6 | `Diagonal ; Dense,Diag/Dense` | 0.0 | small fully crossed scalar terms |
| oxide | small | oxide | 72 | 2 | 6 | 2 | 48,16 | `BlkDiag ; Dense,BlkDiag` | 0.0 | nested vector-valued terms, UniformBlockDiagonal diagonal blocks |
| kb07 | small | kb07 | 1789 | 8 | 20 | 2 | 224,128 | `BlkDiag ; Dense,BlkDiag/Dense` | 0.4 | maximal crossed vector-valued model: 20θ on only 1789 rows |
| kwdyz11 | medium | kwdyz11 | 28710 | 4 | 20 | 2 | 1920,244 | `BlkDiag ; Dense,BlkDiag/Dense` | 4.2 | crossed 4-column vector-valued terms, n = 28710 |
| kkl15 | medium | kkl15 | 53765 | 4 | 10 | 1 | 344 | `BlkDiag` | 0.0 | a single 4-column vector-valued term, n = 53765 |
| mrk17 | medium | mrk17_exp1 | 16409 | 32 | 36 | 2 | 1200,438 | `BlkDiag ; Dense,BlkDiag/Dense` | 5.9 | the most parameters in the suite: 36θ and p = 32 |
| insteval | medium | insteval | 73421 | 2 | 3 | 3 | 2972,1128,14 | `Diagonal ; Sparse,Diag/TrRFP ; Dense,Sparse/Dense,Diag/Dense` | 6.3 | three crossed scalar terms, n = 73421, sparse cross blocks |
| insteval_fe | medium | insteval | 73421 | 28 | 2 | 2 | 2972,1128 | `Diagonal ; Sparse,Diag/TrRFP` | 6.6 | two crossed scalar terms but p = 28 fixed-effects columns |
| insteval_vec | medium | insteval | 73421 | 2 | 5 | 3 | 2972,1128,28 | `Diagonal ; Sparse,Diag/TrRFP ; Dense,Sparse/Dense,BlkDiag/Dense` | 6.7 | scalar and vector-valued terms mixed in one model |
| d3 | medium | d3 | 130418 | 2 | 9 | 3 | 9452,344,68 | `BlkDiag ; Sparse,BlkDiag ; Sparse,Dense,BlkDiag/Dense` | 11.3 | three crossed vector-valued terms, n = 130418 |
| ml1m | large | ml1m | 1000209 | 1 | 2 | 2 | 6040,3706 | `Diagonal ; Sparse,Diag/TrRFP` | 64.1 | n = 10^6 with only 2θ; the fill-in lands in a 3706×3706 diagonal block, which is stored in RFP format at the default threshold |

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
| sleepstudy1 | 1 | 0.002 | 0.005 | 2.8× | 0.002 | 0.5× | 0.2 | 0.5 | 0.4 | 0.0 | 0.0 | 8.7e-13 |
| sleepstudy_zc | 2 | 0.005 | 0.018 | 3.8× | 0.004 | 0.3× | 0.2 | 0.5 | 0.4 | 0.0 | 0.1 | 3.1e-12 |
| sleepstudy | 3 | 0.008 | 0.022 | 2.8× | 0.005 | 0.2× | 0.2 | 0.5 | 0.6 | 0.0 | 0.1 | 3.1e-12 |
| pastes | 2 | 0.004 | 0.010 | 2.8× | 0.003 | 0.3× | 0.7 | 1.3 | 0.9 | 0.0 | 0.0 | 7.1e-12 |
| penicillin | 2 | 0.004 | 0.013 | 3.5× | 0.005 | 0.4× | 0.4 | 1.1 | 0.6 | 0.0 | 0.1 | 2.5e-11 |
| oxide | 6 | 0.022 | 0.057 | 2.6× | 0.088 | 1.5× | 0.5 | 1.2 | 1.0 | 0.0 | 0.2 | 4.9e-09 |
| kb07 | 20 | 0.319 | 0.765 | 2.4× | 66.935 | 87.5× | 0.7 | 1.4 | 2.7 | 1.1 | 11.9 | 3.2e-11 |
| kwdyz11 | 20 | 3.552 | 12.606 | 3.5× | 2735.755 | 217.0× | 0.7 | 1.4 | 2.7 | 15.0 | 160.0 | 6.5e-09 |
| kkl15 | 10 | 0.055 | 0.130 | 2.4× | 0.350 | 2.7× | 0.4 | 0.6 | 1.0 | 0.0 | 62.4 | 8.1e-11 |
| mrk17 | 36 | 5.585 | 20.499 | 3.7× | 10939.237 | 533.6× | 1.0 | 2.3 | 4.6 | 18.5 | 209.8 | 5.0e-09 |
| insteval | 3 | 11.737 | 92.213 | 7.9× | 663.750 | 7.2× | 0.8 | 2.3 | 1.1 | 33.2 | 85.3 | 1.0e-08 |
| insteval_fe | 2 | 11.596 | 88.074 | 7.6× | 555.580 | 6.3× | 0.5 | 1.4 | 0.6 | 33.4 | 78.9 | 2.1e-08 |
| insteval_vec | 5 | 12.511 | 96.167 | 7.7× | 1066.227 | 11.1× | 0.9 | 2.2 | 1.4 | 34.7 | 136.7 | 2.8e-09 |
| d3 | 9 | 4.373 | 13.055 | 3.0× | 23.575 | 1.8× | 38.1 | 39.1 | 39.0 | 12.5 | 248.7 | 3.1e-09 |
| ml1m | 2 | 767.846 | 2379.203 | 3.1× | 21192.071 | 8.9× | 0.5 | 1.3 | 0.6 | 227.2 | 667.1 | 2.7e-08 |

## Complete fits

`Δobjective` is measured against the smallest objective reached for that
model, so a positive value means that configuration stopped short of the
best optimum found.  `max|∇|` is the analytic gradient at the returned
optimum on the deviance scale, a scale-free measure of how tightly each
optimizer converged.  ms/eval includes the gradient for the `LD_*` rows.

| model | nθ | configuration | algorithm | feval | time (s) | ms/eval | alloc (MiB) | # allocs | peak RSS (MiB) | objective | Δobjective | max\|∇\| | singular | return |
|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| sleepstudy1 | 1 | LN_NEWUOA | LN_BOBYQA | 13 | 0.012 | 0.886 | 0.9 | 17622 | 849.7 | 1794.0786 | 1.14e-11 | 1.4e-07 |  | FTOL_REACHED |
| sleepstudy1 | 1 | LN_BOBYQA | LN_BOBYQA | 13 | 0.012 | 0.950 | 0.9 | 17622 | 865.9 | 1794.0786 | 1.14e-11 | 1.4e-07 |  | FTOL_REACHED |
| sleepstudy1 | 1 | LD_LBFGS + analytic | LD_LBFGS | 7 | 0.015 | 2.170 | 0.9 | 17683 | 864.4 | 1794.0786 | 4.77e-12 | 4.3e-08 |  | SUCCESS |
| sleepstudy1 | 1 | LD_LBFGS + forwarddiff | LD_LBFGS | 7 | 0.015 | 2.144 | 0.9 | 17669 | 867.3 | 1794.0786 | 0.00e+00 | 4.3e-08 |  | SUCCESS |
| sleepstudy_zc | 2 | LN_NEWUOA | LN_NEWUOA | 45 | 0.014 | 0.317 | 0.9 | 18173 | 853.3 | 1752.0033 | 7.44e-11 | 3.3e-04 |  | FTOL_REACHED |
| sleepstudy_zc | 2 | LN_BOBYQA | LN_BOBYQA | 47 | 0.015 | 0.310 | 0.9 | 18203 | 859.7 | 1752.0033 | 1.09e-09 | 6.8e-04 |  | FTOL_REACHED |
| sleepstudy_zc | 2 | LD_LBFGS + analytic | LD_LBFGS | 13 | 0.018 | 1.359 | 0.9 | 17857 | 865.0 | 1752.0033 | 1.14e-11 | 2.6e-08 |  | SUCCESS |
| sleepstudy_zc | 2 | LD_LBFGS + forwarddiff | LD_LBFGS | 13 | 0.017 | 1.309 | 0.9 | 17829 | 882.0 | 1752.0033 | 0.00e+00 | 2.6e-08 |  | SUCCESS |
| sleepstudy | 3 | LN_NEWUOA | LN_NEWUOA | 82 | 0.015 | 0.185 | 0.9 | 18820 | 851.7 | 1751.9393 | 7.48e-11 | 2.2e-04 |  | FTOL_REACHED |
| sleepstudy | 3 | LN_BOBYQA | LN_BOBYQA | 73 | 0.015 | 0.204 | 0.9 | 18676 | 859.6 | 1751.9393 | 1.80e-09 | 1.7e-03 |  | FTOL_REACHED |
| sleepstudy | 3 | LD_LBFGS + analytic | LD_LBFGS | 17 | 0.018 | 1.035 | 0.9 | 17981 | 856.2 | 1751.9393 | 0.00e+00 | 4.0e-05 |  | FTOL_REACHED |
| sleepstudy | 3 | LD_LBFGS + forwarddiff | LD_LBFGS | 17 | 0.019 | 1.120 | 0.9 | 17915 | 866.5 | 1751.9393 | 2.14e-11 | 4.0e-05 |  | FTOL_REACHED |
| pastes | 2 | LN_NEWUOA | LN_NEWUOA | 33 | 0.012 | 0.357 | 0.9 | 18334 | 860.7 | 247.9945 | 0.00e+00 | 1.1e-05 |  | FTOL_REACHED |
| pastes | 2 | LN_BOBYQA | LN_BOBYQA | 46 | 0.012 | 0.256 | 0.9 | 18672 | 858.5 | 247.9945 | 1.39e-10 | 4.9e-06 |  | FTOL_REACHED |
| pastes | 2 | LD_LBFGS + analytic | LD_LBFGS | 14 | 0.015 | 1.075 | 0.9 | 18309 | 870.9 | 247.9945 | 1.25e-10 | 2.7e-06 |  | FTOL_REACHED |
| pastes | 2 | LD_LBFGS + forwarddiff | LD_LBFGS | 12 | 0.015 | 1.216 | 0.9 | 17976 | 892.1 | 247.9945 | 4.50e-11 | 4.9e-08 |  | SUCCESS |
| penicillin | 2 | LN_NEWUOA | LN_NEWUOA | 37 | 0.012 | 0.315 | 0.9 | 18248 | 860.2 | 332.1883 | 5.13e-10 | 1.1e-04 |  | FTOL_REACHED |
| penicillin | 2 | LN_BOBYQA | LN_BOBYQA | 61 | 0.012 | 0.194 | 0.9 | 18752 | 862.5 | 332.1883 | 9.58e-09 | 2.0e-04 |  | FTOL_REACHED |
| penicillin | 2 | LD_LBFGS + analytic | LD_LBFGS | 25 | 0.015 | 0.585 | 0.9 | 18823 | 872.2 | 332.1883 | 0.00e+00 | 1.4e-05 |  | FTOL_REACHED |
| penicillin | 2 | LD_LBFGS + forwarddiff | LD_LBFGS | 13 | 0.014 | 1.108 | 0.9 | 17926 | 896.2 | 332.1883 | 1.35e-10 | 1.5e-07 |  | SUCCESS |
| oxide | 6 | LN_NEWUOA | LN_NEWUOA | 110 | 0.018 | 0.163 | 1.0 | 20822 | 867.5 | 453.2275 | 5.80e-08 | 6.6e-04 |  | FTOL_REACHED |
| oxide | 6 | LN_BOBYQA | LN_BOBYQA | 157 | 0.018 | 0.115 | 1.0 | 22232 | 871.1 | 453.2275 | 2.96e-09 | 3.1e-04 |  | FTOL_REACHED |
| oxide | 6 | LD_LBFGS + analytic | LD_LBFGS | 15 | 0.020 | 1.366 | 0.9 | 18441 | 915.7 | 453.2275 | 9.58e-09 | 4.5e-07 |  | SUCCESS |
| oxide | 6 | LD_LBFGS + forwarddiff | LD_LBFGS | 19 | 0.021 | 1.107 | 1.1 | 18217 | 920.7 | 453.2275 | 0.00e+00 | 2.5e-05 |  | FTOL_REACHED |
| kb07 | 20 | LN_NEWUOA | LN_NEWUOA | 798 | 0.301 | 0.378 | 1.6 | 51120 | 890.3 | 28637.1232 | 4.91e-04 | 5.9e-02 | yes | FTOL_REACHED |
| kb07 | 20 | LN_BOBYQA | LN_BOBYQA | 964 | 0.359 | 0.373 | 1.8 | 58092 | 888.9 | 28637.1228 | 1.22e-04 | 6.0e-02 | yes | FTOL_REACHED |
| kb07 | 20 | LD_LBFGS + analytic | LD_LBFGS | 49 | 0.058 | 1.187 | 2.0 | 20853 | 889.8 | 28637.1227 | 6.18e-11 | 8.4e-03 | yes | FTOL_REACHED |
| kb07 | 20 | LD_LBFGS + forwarddiff | LD_LBFGS | 49 | 3.728 | 76.088 | 11.7 | 20399 | 897.5 | 28637.1227 | 0.00e+00 | 8.4e-03 | yes | FTOL_REACHED |
| kwdyz11 | 20 | LN_NEWUOA | LN_NEWUOA | 1138 | 4.865 | 4.275 | 2.0 | 65358 | 893.6 | 325100.0562 | 1.59e-03 | 4.0e-01 |  | FTOL_REACHED |
| kwdyz11 | 20 | LN_BOBYQA | LN_BOBYQA | 1264 | 5.579 | 4.414 | 2.1 | 70650 | 893.8 | 325100.0547 | 0.00e+00 | 1.2e-01 |  | FTOL_REACHED |
| kwdyz11 | 20 | LD_LBFGS + analytic | LD_LBFGS | 39 | 0.562 | 14.416 | 16.0 | 20179 | 893.7 | 325100.0548 | 1.54e-04 | 4.2e-01 |  | FTOL_REACHED |
| kwdyz11 | 20 | LD_LBFGS + forwarddiff | LD_LBFGS | 39 | 130.107 | 3336.075 | 146.9 | 19827 | 1066.9 | 325100.0548 | 1.54e-04 | 4.2e-01 |  | FTOL_REACHED |
| kkl15 | 10 | LN_NEWUOA | LN_NEWUOA | 361 | 0.046 | 0.128 | 1.1 | 25825 | 866.7 | 602662.2396 | 2.25e-04 | 4.4e-01 |  | FTOL_REACHED |
| kkl15 | 10 | LN_BOBYQA | LN_BOBYQA | 256 | 0.035 | 0.136 | 1.0 | 23410 | 871.0 | 602664.3635 | 2.12e+00 | 2.2e-01 |  | FTOL_REACHED |
| kkl15 | 10 | LD_LBFGS + analytic | LD_LBFGS | 28 | 0.026 | 0.919 | 0.9 | 18477 | 880.1 | 602662.2394 | 0.00e+00 | 2.0e-02 |  | FTOL_REACHED |
| kkl15 | 10 | LD_LBFGS + forwarddiff | LD_LBFGS | 28 | 0.048 | 1.723 | 56.6 | 18202 | 941.2 | 602662.2394 | 9.31e-10 | 2.0e-02 |  | FTOL_REACHED |
| mrk17 | 36 | LN_NEWUOA | LN_NEWUOA | 3241 | 28.861 | 8.905 | 5.2 | 205680 | 996.6 | 7147.5215 | 2.56e-03 | 2.3e-01 | yes | FTOL_REACHED |
| mrk17 | 36 | LN_BOBYQA | LN_BOBYQA | 3592 | 30.295 | 8.434 | 5.7 | 226038 | 957.8 | 7147.5190 | 0.00e+00 | 3.4e-02 | yes | FTOL_REACHED |
| mrk17 | 36 | LD_LBFGS + analytic | LD_LBFGS | 91 | 3.142 | 34.525 | 19.6 | 25480 | 998.4 | 7147.5201 | 1.12e-03 | 8.7e-01 | yes | FTOL_REACHED |
| mrk17 | 36 | LD_LBFGS + forwarddiff | LD_LBFGS | 91 | 1120.648 | 12314.810 | 191.7 | 25309 | 1177.4 | 7147.5201 | 1.11e-03 | 8.2e-01 | yes | FTOL_REACHED |
| insteval | 3 | LN_NEWUOA | LN_NEWUOA | 81 | 1.232 | 15.216 | 0.9 | 20487 | 913.3 | 237721.7688 | 1.26e-07 | 1.1e-01 |  | FTOL_REACHED |
| insteval | 3 | LN_BOBYQA | LN_BOBYQA | 110 | 1.521 | 13.827 | 1.0 | 21560 | 900.5 | 237721.7688 | 7.48e-08 | 2.4e-02 |  | FTOL_REACHED |
| insteval | 3 | LD_LBFGS + analytic | LD_LBFGS | 31 | 3.751 | 120.991 | 34.1 | 21097 | 931.1 | 237721.7688 | 0.00e+00 | 3.5e-05 |  | SUCCESS |
| insteval | 3 | LD_LBFGS + forwarddiff | LD_LBFGS | 31 | 21.814 | 703.677 | 80.5 | 18921 | 1024.3 | 237721.7688 | 9.90e-10 | 2.8e-04 |  | SUCCESS |
| insteval_fe | 2 | LN_NEWUOA | LN_NEWUOA | 49 | 0.682 | 13.912 | 0.9 | 18650 | 933.4 | 237585.5534 | 0.00e+00 | 1.0e-04 |  | FTOL_REACHED |
| insteval_fe | 2 | LN_BOBYQA | LN_BOBYQA | 46 | 0.657 | 14.290 | 0.9 | 18578 | 940.2 | 237585.5534 | 1.60e-09 | 6.1e-03 |  | FTOL_REACHED |
| insteval_fe | 2 | LD_LBFGS + analytic | LD_LBFGS | 16 | 1.581 | 98.815 | 34.3 | 18602 | 967.6 | 237585.5534 | 1.75e-10 | 2.5e-03 |  | FTOL_REACHED |
| insteval_fe | 2 | LD_LBFGS + forwarddiff | LD_LBFGS | 16 | 9.956 | 622.247 | 60.6 | 18071 | 1001.6 | 237585.5534 | 2.50e-09 | 2.5e-03 |  | FTOL_REACHED |
| insteval_vec | 5 | LN_NEWUOA | LN_NEWUOA | 204 | 3.174 | 15.561 | 1.1 | 25896 | 906.8 | 237647.0584 | 0.00e+00 | 6.4e-03 |  | FTOL_REACHED |
| insteval_vec | 5 | LN_BOBYQA | LN_BOBYQA | 323 | 5.194 | 16.081 | 1.2 | 30775 | 908.5 | 237647.0584 | 2.01e-08 | 2.9e-02 |  | FTOL_REACHED |
| insteval_vec | 5 | LD_LBFGS + analytic | LD_LBFGS | 41 | 4.349 | 106.071 | 35.6 | 21736 | 946.2 | 237647.0584 | 9.98e-08 | 2.7e-02 |  | FTOL_REACHED |
| insteval_vec | 5 | LD_LBFGS + forwarddiff | LD_LBFGS | 41 | 44.648 | 1088.987 | 129.4 | 19437 | 1014.5 | 237647.0584 | 9.73e-08 | 2.7e-02 |  | FTOL_REACHED |
| d3 | 9 | LN_NEWUOA | LN_NEWUOA | 820 | 4.162 | 5.076 | 31.5 | 59370 | 885.7 | 884957.5540 | 2.83e-05 | 3.6e+00 |  | FTOL_REACHED |
| d3 | 9 | LN_BOBYQA | LN_BOBYQA | 926 | 4.956 | 5.352 | 35.5 | 64776 | 892.4 | 884957.5540 | 0.00e+00 | 4.2e+00 |  | FTOL_REACHED |
| d3 | 9 | LD_LBFGS + analytic | LD_LBFGS | 100 | 1.901 | 19.005 | 11.6 | 26510 | 904.8 | 884957.5541 | 1.65e-04 | 8.9e-01 |  | FTOL_REACHED |
| d3 | 9 | LD_LBFGS + forwarddiff | LD_LBFGS | 96 | 2.450 | 25.525 | 209.5 | 22253 | 1275.9 | 884957.5542 | 2.72e-04 | 1.4e+00 |  | FTOL_REACHED |
| ml1m | 2 | LN_NEWUOA | LN_NEWUOA | 52 | 46.511 | 894.449 | 0.9 | 18722 | 1137.1 | 2663972.0116 | 7.73e-08 | 1.2e-03 |  | FTOL_REACHED |
| ml1m | 2 | LN_BOBYQA | LN_BOBYQA | 49 | 40.911 | 834.913 | 0.9 | 18650 | 1154.9 | 2663972.0116 | 8.24e-08 | 1.7e-02 |  | FTOL_REACHED |
| ml1m | 2 | LD_LBFGS + analytic | LD_LBFGS | 13 | 36.044 | 2772.650 | 228.1 | 18397 | 1448.5 | 2663972.0116 | 7.92e-08 | 4.3e-02 |  | FTOL_REACHED |
| ml1m | 2 | LD_LBFGS + forwarddiff | LD_LBFGS | 13 | 278.013 | 21385.578 | 627.6 | 17991 | 1670.8 | 2663972.0116 | 0.00e+00 | 3.9e-02 |  | FTOL_REACHED |

## Summary: time to a complete fit

Each cell is the wall-clock time in seconds and, in parentheses, the speed-up
relative to the fastest derivative-free configuration for that model.  A
value below 1× means the gradient did not pay for itself.

| model | nθ | n | LN_NEWUOA | LN_BOBYQA | LD_LBFGS + analytic | LD_LBFGS + forwarddiff |
|---|--:|--:|--:|--:|--:|--:|
| sleepstudy1 | 1 | 180 | 0.012 (1.0×) | 0.012 (0.9×) | 0.015 (0.8×) | 0.015 (0.8×) |
| sleepstudy_zc | 2 | 180 | 0.014 (1.0×) | 0.015 (1.0×) | 0.018 (0.8×) | 0.017 (0.8×) |
| sleepstudy | 3 | 180 | 0.015 (1.0×) | 0.015 (1.0×) | 0.018 (0.8×) | 0.019 (0.8×) |
| pastes | 2 | 60 | 0.012 (1.0×) | 0.012 (1.0×) | 0.015 (0.8×) | 0.015 (0.8×) |
| penicillin | 2 | 144 | 0.012 (1.0×) | 0.012 (1.0×) | 0.015 (0.8×) | 0.014 (0.8×) |
| oxide | 6 | 72 | 0.018 (1.0×) | 0.018 (1.0×) | 0.020 (0.9×) | 0.021 (0.9×) |
| kb07 | 20 | 1789 | 0.301 (1.0×) | 0.359 (0.8×) | 0.058 (5.2×) | 3.728 (0.1×) |
| kwdyz11 | 20 | 28710 | 4.865 (1.0×) | 5.579 (0.9×) | 0.562 (8.7×) | 130.107 (0.0×) |
| kkl15 | 10 | 53765 | 0.046 (0.8×) | 0.035 (1.0×) | 0.026 (1.4×) | 0.048 (0.7×) |
| mrk17 | 36 | 16409 | 28.861 (1.0×) | 30.295 (1.0×) | 3.142 (9.2×) | 1120.648 (0.0×) |
| insteval | 3 | 73421 | 1.232 (1.0×) | 1.521 (0.8×) | 3.751 (0.3×) | 21.814 (0.1×) |
| insteval_fe | 2 | 73421 | 0.682 (1.0×) | 0.657 (1.0×) | 1.581 (0.4×) | 9.956 (0.1×) |
| insteval_vec | 5 | 73421 | 3.174 (1.0×) | 5.194 (0.6×) | 4.349 (0.7×) | 44.648 (0.1×) |
| d3 | 9 | 130418 | 4.162 (1.0×) | 4.956 (0.8×) | 1.901 (2.2×) | 2.450 (1.7×) |
| ml1m | 2 | 1000209 | 46.511 (0.9×) | 40.911 (1.0×) | 36.044 (1.1×) | 278.013 (0.1×) |

