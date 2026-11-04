# Analytic gradient of the LMM objective: rewrite vs. `db/pa/gradient`

This note documents how the analytic gradient on `pa/gradient-fable`
(`src/gradient.jl`, `objective_gradient!` + `GradientWorkspace`) differs from the
prototype on `db/pa/gradient` (`eval_grad_p!` + `initialize_blocks!`), and why. Both
branches start from the same identity; the rewrite changes *how the trace is evaluated*,
*which objectives are supported*, and *how the block algebra is organized for speed*.

Throughout, $\Omega = L L^\top$ is the blocked lower-Cholesky factorization of the
penalized augmented cross-product that MixedModels factors, $\theta$ is the covariance
parameter vector, and $\Lambda_b(\theta)$ is the relative covariance factor of
random-effects term $b$. Block $k+1$ (written `nb`) is the combined $[X\ y]$
fixed-effects/response block.

---

## 1. The objective is affine in $\log L_{jj}$

**Claim.** For all three fitting criteria the objective has the form

$$
\text{obj} \;=\; 2\sum_j w_j \log L_{jj} \;+\; \text{const},
$$

with weights $w_j$ that differ only by criterion. The gradient machinery is identical
across criteria; only the weight vector $w$ changes.

**Derivation.** Let $r^2 = \text{pwrss}$ be the penalized residual sum of squares. In the
augmented factorization $r^2 = \ell_{yy}^2$, where $\ell_{yy} = L_{nb,nb}[\text{last}]$
is the trailing diagonal entry of $L$. The log-determinant of the random-effects portion
is $2\sum_{j\in\text{RE}} \log L_{jj}$, and of the fixed-effects portion (the $R_X$ block)
$2\sum_{j\in\text{FE}} \log L_{jj}$.

*ML*, profiled over $\sigma$ with $\hat\sigma^2 = r^2/n$:

$$
\text{obj}_{\text{ML}}
= 2\!\!\sum_{j\in\text{RE}}\!\! \log L_{jj} \;+\; n\bigl(1 + \log(2\pi\hat\sigma^2)\bigr)
= 2\!\!\sum_{j\in\text{RE}}\!\! \log L_{jj} \;+\; 2n\log \ell_{yy} \;+\; \text{const},
$$

using $n\log\hat\sigma^2 = n\log(r^2/n) = 2n\log\ell_{yy} + \text{const}$. So
$w_j = 1$ on RE rows, $w_j = 0$ on FE rows, $w = n$ on the $\ell_{yy}$ row.

*REML*, with $s^2 = r^2/(n-p)$:

$$
\text{obj}_{\text{REML}}
= 2\!\!\sum_{j\in\text{RE}}\!\! \log L_{jj}
+ 2\!\!\sum_{j\in\text{FE}}\!\! \log L_{jj}
+ 2(n-p)\log \ell_{yy} + \text{const}.
$$

So $w_j = 1$ on RE **and** FE rows, $w = n-p$ on the $\ell_{yy}$ row.

*Fixed $\sigma$* (not profiled): the residual term is $r^2/\sigma^2$ rather than a log,
but it is still expressible through $\log\ell_{yy}$:

$$
\frac{\partial}{\partial\theta_p}\!\left(\frac{r^2}{\sigma^2}\right)
= \frac{1}{\sigma^2}\,\frac{\partial \ell_{yy}^2}{\partial\theta_p}
= \frac{2\ell_{yy}}{\sigma^2}\,\frac{\partial \ell_{yy}}{\partial\theta_p}
= \frac{2\,r^2}{\sigma^2}\,\frac{\partial \log\ell_{yy}}{\partial\theta_p}.
$$

So the $\ell_{yy}$ weight is $w = r^2/\sigma^2 = \text{pwrss}/\sigma^2$, and FE rows follow
the REML/ML rule for whichever criterion is combined with the fixed $\sigma$.

In code this is exactly `_yweight` (`ssqdenom` $= n$ or $n-p$, else $\text{pwrss}/\sigma^2$)
and `wx` ($=1$ for REML, $0$ for ML) on the fixed-effects rows.

---

## 2. From $\log L_{jj}$ to a trace (Murray 2016)

Differentiate $\Omega = LL^\top$ w.r.t. a scalar $\theta_p$:

$$
\dot\Omega = \dot L L^\top + L \dot L^\top
\;\;\Longrightarrow\;\;
L^{-1}\dot\Omega L^{-\top} = L^{-1}\dot L + (L^{-1}\dot L)^\top = M + M^\top,
\quad M := L^{-1}\dot L .
$$

$L$ and $\dot L$ are lower triangular, so $M$ is lower triangular and
$M_{jj} = \dot L_{jj}/L_{jj} = \partial_p \log L_{jj}$. Hence the diagonal of the
symmetric matrix $M + M^\top$ is

$$
\bigl[L^{-1}\dot\Omega_p L^{-\top}\bigr]_{jj} = 2\,\partial_p \log L_{jj}.
$$

Combining with §1,

$$
\boxed{\;\frac{\partial\,\text{obj}}{\partial\theta_p}
= \sum_j w_j \bigl[L^{-1}\dot\Omega_p L^{-\top}\bigr]_{jj}
= \operatorname{tr}\!\bigl(W\,L^{-1}\dot\Omega_p L^{-\top}\bigr),\quad W=\operatorname{diag}(w).\;}
\tag{$\star$}
$$

Both branches agree up to here.

---

## 3. The reformulation: compute $S$ once (the core change)

**Prototype (`db/pa/gradient`).** Evaluates $(\star)$ literally, **once per parameter**.
For each $p$, `initialize_blocks!` materializes the full $(k{+}1)\times(k{+}1)$ blocked
$\dot\Omega_p$ (both triangles), then `Lldiv!` + `rdiv!` perform a two-sided blocked
triangular solve $L^{-1}\dot\Omega_p L^{-\top}$, and `diag_sum` reads off the trace. Cost:
$P$ independent two-sided blocked solves over the full block matrix, most of which is
zero.

**Rewrite (`pa/gradient-fable`).** Move the weight matrix through the trace by cyclic
invariance:

$$
\operatorname{tr}\!\bigl(W L^{-1}\dot\Omega_p L^{-\top}\bigr)
= \operatorname{tr}\!\bigl(\underbrace{L^{-\top} W L^{-1}}_{=:S}\,\dot\Omega_p\bigr)
= \langle S, \dot\Omega_p\rangle,
\qquad S = X^\top W X,\ \ X := L^{-1},
$$

with $\langle A,B\rangle = \sum_{ij}A_{ij}B_{ij}$ (both operands symmetric). **$S$ does not
depend on $p$.** So $X = L^{-1}$ is formed once, the Gram matrix $S = X^\top W X$ is
formed once, and each of the $P$ gradient components is a cheap contraction of the sparse
$\dot\Omega_p$ against $S$.

Blockwise, since $X$ is lower triangular,

$$
S[r,b] = \sum_{s \ge \max(r,b)} X[s,r]^\top W_s\, X[s,b].
$$

This is the whole reason the file is reorganized around a **workspace holding the lower
blocks of $X$ and the blocks of $S$**, and it is the source of the ~60× speedup: the
prototype's per-parameter two-sided solve is replaced by one blocked inverse plus $O(P)$
small contractions.

---

## 4. $\dot\Omega_p$ is sparse; the contraction is a small matrix product

Let `parmap[p] = (b,i,j)`, i.e. $\theta_p$ is entry $(i,j)$ of $\Lambda_b$, so
$\partial\Lambda_b/\partial\theta_p = E_{ij}$ (the single-entry indicator) and every other
$\Lambda$ is constant. The system blocks are $\Omega[r,c] = \Lambda_r^\top A[r,c]\Lambda_c$
on RE blocks (with $\Lambda_{nb}=I$ on the $[X\,y]$ block, and constant $+I$ augmentation
whose derivative vanishes). Therefore $\dot\Omega_p$ is supported only on **block row/column
$b$**:

$$
\dot\Omega_p[b,b] = E_{ij}^\top A[b,b]\Lambda_b + \Lambda_b^\top A[b,b] E_{ij},
\qquad
\dot\Omega_p[r,b] = \Lambda_r^\top A[r,b]\,E_{ij}\ \ (r>b),
$$

and the mirror images on block column $b$.

**Reducing $\langle S[r,b],\dot\Omega_p[r,b]\rangle$ to one entry.** With $C := \Lambda_r^\top A[r,b]$,
the matrix $C E_{ij}$ has its $j$-th column equal to the $i$-th column of $C$ and is zero
elsewhere, so

$$
\langle S[r,b],\, C E_{ij}\rangle
= \sum_a S[r,b]_{a j}\,C_{a i}
= \bigl(C^\top S[r,b]\bigr)_{ij}
= \bigl((\Lambda_r^\top A[r,b])^\top S[r,b]\bigr)_{ij}.
$$

So the contribution of pair $(r,b)$ to term $b$ is the $(i,j)$ entry of a single
$k_b\times k_b$ matrix — **for all parameters of the term at once**. Collect every
contribution touching term $b$ into an accumulator $G_b$:

$$
G_b \;=\; (\Lambda_b^\top A[b,b])^\top S[b,b]\big|_{\text{diag part}}
\;+\; \sum_{r\ne b}(\Lambda_r^\top A[r,b])^\top S[r,b]
\;+\; A[nb,b]^\top S[nb,b].
$$

**The factor of 2.** For the diagonal block, using symmetry of $S$,

$$
\langle S[b,b],\dot\Omega_p[b,b]\rangle
= \langle S, E_{ij}^\top A\Lambda_b\rangle + \langle S,\Lambda_b^\top A E_{ij}\rangle
= 2\bigl((\Lambda_b^\top A[b,b])^\top S[b,b]\bigr)_{ij}.
$$

For each off-diagonal block, both $(r,b)$ and $(b,r)$ appear in the symmetric trace and are
equal, again contributing a factor 2. Hence

$$
\frac{\partial\,\text{obj}}{\partial\theta_p} = 2\,G_b[i,j],
$$

which is precisely `g[p] = 2 * w.G[b][i,j]`. In the implementation each pair updates *both*
term accumulators (`_densepair!` builds $C_1 = \Lambda_r^\top A[r,b]$ for $G_b$ and
$C_2 = A[r,b]\Lambda_b$ for $G_r$), and the sum over grouping-factor levels is the
face-by-face loop (`_facecontract!` / `_facecontract_rows!`). $\dot\Omega_p$ is **never
materialized** — the prototype's `copyskip!`/`initialize_blocks!` machinery is gone.

---

## 5. Corrections

| Correction | Prototype | Rewrite | Why it matters |
|---|---|---|---|
| **REML** | `gradient!` documented as the *ML* objective; the trace loop stops at block $k$ (FE block dropped) and hardcodes $n\cdot\ell_{yy}$. | `wx = REML ? 1 : 0` includes the FE block; `_yweight` uses $n-p$. | Prototype returns a **wrong vector** for REML fits. §1 gives the exact FE and $\ell_{yy}$ weights. |
| **Fixed $\sigma$** | not handled. | `_yweight = pwrss/σ²`. | Derived in §1; matches ForwardDiff/FiniteDiff on fixed-$\sigma$ fits. |
| **Objective value** | `gradient!` returns only `g`. | `objective_gradient!` returns `objective(m)` too. | One traversal of $L$ yields value + gradient for the optimizer. |
| **Vector diagonal path** | `Omega_dot_diag_block!(::Matrix)` throws `"Code not yet written for k > 1"`. | uniform handling via `lmulΛ!`/`rmulΛ!` + face contraction. | no size ceiling on term width. |
| **Dead/debug code** | `@info`, commented reference blocks, `blks2dense` hardcoded to ≤4 blocks, duplicate `Lldiv!`. | removed. | maintainability. |

---

## 6. Optimizations (all motivated by §3)

1. **Compute $X = L^{-1}$ once, lower blocks only** (`_invL!`). $S = X^\top W X$ needs only
   the lower triangle; the prototype solved the full square block system, with transpose
   copies, once per parameter.
2. **Never materialize $\dot\Omega_p$** (§4). Its structure folds into face contractions
   against the compact $A$ blocks.
3. **Preallocated, concretely-typed `GradientWorkspace`.** Buffers ($X, S, C_1, C_2, G,
   P_\text{panel}$) are allocated once and reused across every gradient call in the
   optimizer loop. The abstract-eltype block storage is confined behind **function
   barriers** (`_sparseacc`, `_gram*`), so the hot loops run type-stable — removing the
   pervasive instability of the prototype's `AbstractMatrix`-typed arithmetic.
4. **Sparse selected-entry path** (`_sparsepair!`). Between two scalar terms with sparse
   $A[r,b]$, only the entries of $S[r,b]$ on $A$'s sparsity pattern are evaluated — no dense
   $S$ block is formed. Decisive for crossed designs where $S[r,b]$ is huge but mostly
   irrelevant.
5. **BLAS-3 cross-term kernel** (`_crossacc_blas3!`, gated by `_use_blas3_cross`). When the
   fill block $L[r,r]$ is dense (crossed subject×item), the $s=r$ term of the selected-entry
   sum is bandwidth-bound BLAS-1 column dot products; a panelled ($P_\text{panel}=128$)
   BLAS-3 product replaces them, materializing only a $q_r\times128$ slice at a time. Gated
   on both dense fill *and* $A[r,b]$ density $>3\%$ so the extra flops pay off. No analogue
   in the prototype.
6. **`_mulsub!` sparse×`Diagonal` fast path** — writes $C \mathrel{-}= nz\cdot d$ directly
   over nonzeros; the cleaned-up successor of the prototype's `mm_mul!` special case.

---

## 7. Validation

Cross-checked against `ForwardDiff.gradient` (which honors `optsum.REML` and fixed
$\sigma$ in its objective), on the crossed kb07 subject×item design:

| Case | Path exercised | rel. error |
|---|---|---|
| ML @ perturbed θ | vector-term dense | 1.2e-13 |
| REML @ perturbed θ | vector-term dense | 1.2e-13 |
| ML scalar-crossed | sparse / BLAS-3 | ~1e-8* |
| REML scalar-crossed | sparse / BLAS-3 | ~1e-7* |

\*evaluated near the optimum, where $\|g\|\to0$ inflates the ratio; absolute residuals are
near machine precision.

Regression tests added to `test/grad.jl` parametrize over
`{scalar, vector}-crossed × {ML, REML}` on kb07, asserting both the returned objective
value and agreement with ForwardDiff (`rtol=1e-6, atol=1e-6`). The REML crossed cases are
the capability the prototype could not produce correctly.

---

## 8. Summary

The prototype and the rewrite share the identity $(\star)$. The prototype evaluates it by
forming and two-sided-solving the full $\dot\Omega_p$ per parameter, and is correct only
for ML with free $\sigma$. The rewrite pushes $W$ through the trace to get the
$p$-independent Gram matrix $S = L^{-\top}WL^{-1}$, builds it once, and contracts the sparse
$\dot\Omega_p$ against it — yielding a correct ML/REML/fixed-$\sigma$ gradient that is
allocation-free after warmup, with sparse and BLAS-3 fast paths for crossed designs.
