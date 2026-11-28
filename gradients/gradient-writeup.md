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

**Claim.** For all three fitting criteria the gradient has the form

$$
\frac{\partial\,\text{obj}}{\partial\theta_p} \;=\; 2\sum_j w_j\, \frac{\partial \log L_{jj}}{\partial\theta_p},
$$

with weights $w_j$ that differ only by criterion. For ML and REML the objective itself is
affine in $\log L_{jj}$, $\text{obj} = 2\sum_j w_j \log L_{jj} + \text{const}$, with
constant weights; for fixed $\sigma$ only the gradient has this form, with an $\ell_{yy}$
weight that depends on $\theta$. The gradient machinery is identical across criteria;
only the weight vector $w$ changes.

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
depend on $p$.** So $X = L^{-1}$ is formed once, the blocks of $S = X^\top W X$ that the
contraction touches are formed once, and the whole gradient follows from contracting the
sparse $\dot\Omega_p$ against $S$.

Blockwise, since $X$ is lower triangular,

$$
S[r,b] = \sum_{s \ge \max(r,b)} X[s,r]^\top W_s\, X[s,b].
$$

This is why `src/gradient.jl` is organized around a **workspace holding the lower blocks
of $X$ and the needed blocks of $S$**. The prototype's $P$ per-parameter two-sided solves
become one blocked inverse plus one contraction per nonzero block pair of $A$,
independent of $P$ (§4).

---

## 4. $\dot\Omega_p$ is sparse; the contraction is a small matrix product

Let `parmap[p] = (b,i,j)`, i.e. $\theta_p$ is entry $(i,j)$ of $\Lambda_b$, so
$\partial\Lambda_b/\partial\theta_p = E_{ij}$ (the single-entry indicator) and every other
$\Lambda$ is constant. The system blocks are $\Omega[r,c] = \Lambda_r^\top A[r,c]\Lambda_c$
(with $\Lambda_{nb}=I$ for the $[X\,y]$ block, and a constant $+I$ on the RE diagonal
blocks whose derivative vanishes). Therefore $\dot\Omega_p$ is supported on block row and
column $b$; in the lower triangle,

$$
\begin{aligned}
\dot\Omega_p[b,b] &= E_{ij}^\top A[b,b]\Lambda_b + \Lambda_b^\top A[b,b] E_{ij}, \\
\dot\Omega_p[r,b] &= \Lambda_r^\top A[r,b]\,E_{ij} && (r>b,\ \text{including } r = nb), \\
\dot\Omega_p[b,c] &= E_{ij}^\top A[b,c]\,\Lambda_c && (c<b).
\end{aligned}
$$

**Each block contributes one entry of a $k_b\times k_b$ matrix.** With $C_1 := \Lambda_r^\top A[r,b]$,
the matrix $C_1 E_{ij}$ has its $j$-th column equal to the $i$-th column of $C_1$ and is
zero elsewhere, so

$$
\langle S[r,b],\, C_1 E_{ij}\rangle = \sum_a S[r,b]_{a j}\,(C_1)_{a i} = \bigl(C_1^\top S[r,b]\bigr)_{ij}.
$$

Likewise, with $C_2 := A[b,c]\Lambda_c$, the matrix $E_{ij}^\top C_2$ has its $j$-th row
equal to the $i$-th row of $C_2$, so
$\langle S[b,c],\, E_{ij}^\top C_2\rangle = \bigl(C_2\, S[b,c]^\top\bigr)_{ij}$.

The same $k_b\times k_b$ matrix serves **all parameters of term $b$ at once**. Collect
every contribution touching term $b$ into an accumulator

$$
G_b \;=\; (\Lambda_b^\top A[b,b])^\top S[b,b]
\;+\; \sum_{r>b}(\Lambda_r^\top A[r,b])^\top S[r,b]
\;+\; \sum_{c<b} A[b,c]\Lambda_c\, S[b,c]^\top .
$$

**The factor of 2.** For the diagonal block, using symmetry of $S$ and $A[b,b]$,

$$
\langle S[b,b],\dot\Omega_p[b,b]\rangle
= \langle S, E_{ij}^\top A\Lambda_b\rangle + \langle S,\Lambda_b^\top A E_{ij}\rangle
= 2\bigl((\Lambda_b^\top A[b,b])^\top S[b,b]\bigr)_{ij}.
$$

Each off-diagonal block appears twice in the symmetric trace, at $(r,b)$ and $(b,r)$,
again contributing a factor 2. Hence

$$
\frac{\partial\,\text{obj}}{\partial\theta_p} = 2\,G_b[i,j],
$$

which is `g[p] = 2 * w.G[b][i,j]`. Each off-diagonal pair $(r,b)$, $r>b$, updates both
accumulators: $G_b$ from $C_1 = \Lambda_r^\top A[r,b]$ and $G_r$ from
$C_2 = A[r,b]\Lambda_b$ (the $c<b$ case with $r$ in the role of $b$). The sum over the
levels of a grouping factor is the face-by-face loop (`_facecontract!` /
`_facecontract_rows!`). $\dot\Omega_p$ is **never materialized**; the prototype's
`copyskip!`/`initialize_blocks!` machinery is gone.

---

## 5. Corrections

| Correction | Prototype | Rewrite | Why it matters |
|---|---|---|---|
| **REML** | `gradient!` documented as the *ML* objective; the trace sums only the RE diagonal blocks (FE block dropped) and hardcodes $n\cdot\ell_{yy}$. | `wx = REML ? 1 : 0` includes the FE block; `_yweight` uses $n-p$. | Prototype returns a **wrong vector** for REML fits. §1 gives the exact FE and $\ell_{yy}$ weights. |
| **Fixed $\sigma$** | not handled. | `_yweight = pwrss/σ²`. | Derived in §1; matches ForwardDiff/FiniteDiff on fixed-$\sigma$ fits. |
| **Later terms of multi-term models** | for $\theta_p$ in term $b$, the blocks $\dot\Omega_p[b,c]$, $c<b$, are built as $\Lambda_b^\top A[b,c]E_{ij}$ (`initialize_blocks!`/`copyskip!`) instead of $E_{ij}^\top A[b,c]\Lambda_c$. | contracted against $C_2 = A[b,c]\Lambda_c$ (§4). | Prototype is wrong for every term after the first that shares a nonzero $A[b,c]$ with an earlier term: penicillin ML at a perturbed $\theta$, `sample` component −9.318 vs ForwardDiff −2.196. Single-term models (scalar or vector) agree. |
| **Objective value** | `gradient!` returns only `g`. | `objective_gradient!` returns `objective(m)` too. | value and gradient from one call, as the optimizer needs. |
| **Dense diagonal blocks of vector terms** | `Omega_dot_diag_block!(::Matrix)` throws `"Code not yet written for k > 1"` (the `UniformBlockDiagonal` method covers the usual case). | the same face contraction for every diagonal block type. | no block-type restriction. |

---

## 6. Optimizations

Items 1–3 follow directly from §3; the rest exploit the block structure of $L$ and $A$.

1. **$X = L^{-1}$ once, lower blocks only** (`_invL!`). The diagonal blocks are inverted
   first, by `trtri` (or `tftri` for RFP-packed blocks): a third of the flops of a
   triangular solve against the identity. The prototype solved the full square block
   system, with transpose copies, once per parameter.
2. **$\dot\Omega_p$ is never materialized** (§4).
3. **Preallocated `GradientWorkspace`.** The blocks of $X$ and $S$, $C_1$/$C_2$, $G$, the
   panel scratch, the sparse right-hand-side buffers (`R`, `Xt`) and the per-pair route
   (`path`) are allocated once and reused across the optimizer loop. Function barriers
   (`_sparseacc`, `_selacc!`, `_gram*`) keep the hot loops type-stable despite the
   abstractly typed block storage; an evaluation allocates less than 2 KB.
4. **Structure-mirroring $X$.** The blocks of $X$ keep the types of the blocks of $L$:
   `Diagonal`/`UniformBlockDiagonal` diagonal blocks stay block diagonal, RFP diagonal
   blocks are inverted in packed storage, and for nested grouping factors the off-diagonal
   blocks are `SparseMatrixCSC` sharing the pattern of $L$. The workspace is then of the
   order of $L$ itself; a dense $X[1,1]$ for fggk21 ($q_1 = 541475$) would need 2.3 PB.
5. **Selected entries of $S$.** For sparse $A[r,b]$ only the entries of $S[r,b]$ on the
   pattern of $A$ are computed: accumulated directly between scalar terms (path `:scalar`,
   `_scalarpair!`), otherwise into a sparse buffer mirroring $A$ (path `:selected`,
   `_selectedpair!`). Dense $S$/$C_1$/$C_2$ buffers exist only for dense $A$ blocks (path
   `:dense`).
6. **BLAS-3 cross term** (path `:scalar3`, `_crossacc_blas3!`). Between scalar terms with a
   dense fill block $L[r,r]$ and $A[r,b]$ density above 3%, the dominant $s=r$ term of the
   selected-entry sum (bandwidth-bound BLAS-1 column dot products) is replaced by panelled
   BLAS-3 products, `GRAD_PANEL` = 128 columns at a time.
7. **Sparse right-hand side for $X[c+1,c]$** (`_trimulsp!`, flag `sparserhs`). When
   $-L[c+1,c]X[c,c]$ is sparse (density at most 3%, e.g. the second of two crossed scalar
   factors), $X[c+1,c] = X[c+1,c+1]\,(-L[c+1,c]X[c,c])$ is formed as a dense-triangular ×
   sparse product, at a cost proportional to the nonzeros. This replaces a triangular solve
   against the densified right-hand side; on insteval it cut the gradient from 169 ms to
   70 ms (single-threaded BLAS).
8. **Statically sized kernels.** The face and block products are $k\times k$ with
   $k \le 4$ in practice, far too small for a BLAS call to pay off; `SMatrix` register
   kernels replace the per-face `mul!` calls up to `GRAD_STATIC_K` (and, for
   `_facecontract!`, `GRAD_STATIC_ROWS`).

---

## 7. Validation

Cross-checked against `ForwardDiff.gradient` (which honors `optsum.REML` and fixed
$\sigma$ in its objective) on kb07. The design is fully crossed, so $A[2,1]$ is dense and
both models take the `:dense` pair path.

| Model | Criterion | rel. error, perturbed θ | rel. error, optimum | max. abs. error |
|---|---|---|---|---|
| scalar-crossed | ML | 6.7e-15 | 3.3e-8 | 3.1e-12 |
| scalar-crossed | REML | 6.9e-15 | 6.0e-8 | 1.2e-12 |
| vector-crossed | ML | 1.6e-14 | 5.5e-11 | 6.9e-12 |
| vector-crossed | REML | 1.6e-14 | 4.8e-11 | 3.6e-12 |

The relative error is $\|g - g_\text{FD}\|/\|g_\text{FD}\|$. At the optimum
$\|g\| \approx 10^{-5}$–$10^{-1}$ inflates it, while the absolute error stays near
$10^{-12}$.

`test/grad.jl` compares against ForwardDiff (`rtol` 1e-6 or tighter) for

- every cached dyestuff, pastes, penicillin, sleepstudy and kb07 model, at the initial,
  a perturbed and the fitted $\theta$;
- REML, and fixed $\sigma$ (also against FiniteDiff);
- the kb07 scalar- and vector-crossed models × {ML, REML};
- nested grouping factors (sparse $X$ blocks), including three-level scalar nesting and
  the fggk21 structure;
- synthetic designs that take the `:scalar3` and sparse right-hand-side paths (dense and
  RFP storage), each also against the path it replaces.

The individual kernels are unit-tested on both sides of the static-size cutoffs.

---

## 8. Summary

Both implementations start from $(\star)$. The prototype forms and two-sided-solves the
full $\dot\Omega_p$ per parameter and is correct only for ML with free $\sigma$ and a
single random-effects term. The rewrite builds the $p$-independent $S = L^{-\top}WL^{-1}$
once and contracts it against the structure of $A$, giving a correct ML, REML and
fixed-$\sigma$ gradient for any number of terms.
