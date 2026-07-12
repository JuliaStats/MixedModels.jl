# GPU Support for MixedModels.jl — Feasibility Analysis

## Context

The question is whether GPU acceleration of LMM fitting is worth adding as a
package extension, and whether the heavy use of sparse matrices plus host↔device
transfer cost would erase any gains. This document is a **written feasibility
analysis** (no code), targeting a **portable backend** (KernelAbstractions.jl /
GPUArrays.jl) rather than CUDA-only. The goal is to let you decide whether to
invest before any implementation.

**Top-line verdict:** Not worthwhile as a general drop-in — most models people
fit lose on the GPU. There is a genuine but *narrow* win regime: large **crossed**
random-effects designs and/or large fixed-effects matrices, where the densified
blocks make `updateL!` BLAS3-bound. The commonly-cited objection (data movement)
is *not* the real blocker; block size and kernel-launch latency are.

---

## How fitting actually works (the hot path)

Per optimizer iteration: [linearmixedmodel.jl:885](src/linearmixedmodel.jl#L885)

```julia
objective!(m, θ) = objective(updateL!(setθ!(m, θ)))
```

- `setθ!` writes θ into the small `λ` matrices on each `ReMat` ([remat.jl:20](src/remat.jl#L20)).
- `updateL!` ([linearmixedmodel.jl:1391](src/linearmixedmodel.jl#L1391)) is the cost
  center: it copies the constant cross-product blocks `A` into `L`, scales them by
  `Λ` (`rmulΛ!`/`lmulΛ!`/`copyscaleinflate!`), then performs a **blocked Cholesky**:
  `rankUpdate!` (syrk), `cholUnblocked!` (potrf), `mul!` (gemm), `rdiv!` (trsm) over
  the block grid.
- `objective` ([linearmixedmodel.jl:873](src/linearmixedmodel.jl#L873)) needs only
  `logdet(m)` (sum of logs of `L`'s diagonal) and `pwrss` (one element of the last block).

### The decisive structural fact

`A` is **constant** across all optimizer iterations — only the tiny `λ` change.
So a device-resident design uploads `A` **once**; per iteration the host sends θ
(O(dim θ), typically <20 numbers) and receives back `logdet` + `pwrss` (O(1)).
**Per-iteration PCIe traffic is negligible.** The "cost of moving data to/from the
GPU" objection essentially does not apply to the inner loop — only to one-time setup.

---

## Where GPU wins and loses

Block types are chosen dynamically ([arraytypes.jl](src/arraytypes.jl),
[blockdescription.jl](src/blockdescription.jl)): `Diagonal`,
`UniformBlockDiagonal` (an `Array{T,3}` — a batch of equal-sized blocks),
`SparseMatrixCSC`, `BlockedSparse`, and dense `Matrix`. Densification kicks in at
>10% fill ([arraytypes.jl:73](src/arraytypes.jl#L73)), >25% for cross-products
([remat.jl:479](src/remat.jl#L479)), and **unconditionally for non-nested
(crossed) off-diagonal blocks** ([linearmixedmodel.jl:419](src/linearmixedmodel.jl#L419)).

**GPU-hostile (the common case):** single scalar RE, or nested RE. Blocks stay
`Diagonal`/`UniformBlockDiagonal` with tiny faces (S = 1–10) and the factorization
is O(q) over many small, sequentially-dependent blocks. This is memory-bandwidth /
launch-latency bound; the GPU loses to CPU BLAS and the `Vector{AbstractMatrix}`
dynamic dispatch makes per-block launches worse.

**GPU-favorable (the niche):**
- **Large crossed designs** (e.g. subject × item with many levels each): the
  forced-dense off-diagonal block becomes a large dense `Matrix`, and its
  `rankUpdate!`/`rdiv!` are real BLAS3.
- **Large fixed-effects `p`:** `X'X` and `X'Z` blocks are large and dense.
- **`UniformBlockDiagonal` with many levels:** already `Array{T,3}`, which maps
  directly onto *batched* GEMM/syrk/potrf — the one place batched GPU kernels shine.

So the win is real only when the dense/batched blocks are large enough that
per-iteration arithmetic dwarfs the (dozens to hundreds of) kernel launches.

---

## What it would take (architecture sketch)

The existing extension pattern is clean and well-suited: `[weakdeps]` + bare
function stubs filled in by `ext/` modules, with `__init__` registration
(see `MixedModelsPRIMAExt`, `MixedModelsForwardDiffExt`; stubs in
[derivatives.jl](src/derivatives.jl) and [optsummary.jl](src/optsummary.jl)).

But GPU support is **not** purely additive the way PRIMA/ForwardDiff are, because
`updateL!` dispatches on the *types of the blocks themselves*. To run on device the
blocks must already be GPU arrays. Required pieces:

1. **Device-conversion path** (new stub, e.g. `MixedModels.gpu(m)`/`cpu(m)`,
   implemented in the ext): returns a model whose `A`/`L` blocks are device arrays
   — dense `Matrix`→`AbstractGPUArray`; `UniformBlockDiagonal` backed by a device
   `Array{T,3}`; sparse blocks via the GPU sparse type or densified. `createAL`
   ([linearmixedmodel.jl:405](src/linearmixedmodel.jl#L405)) currently builds CPU
   types, so this is a conversion layer, not just kernel stubs.

2. **Reimplement the ~6 custom kernels for device arrays** — the real work:
   `copyscaleinflate!`, `rmulΛ!`/`lmulΛ!` ([remat.jl:212](src/remat.jl#L212)),
   `rankUpdate!` for `UniformBlockDiagonal`/`BlockedSparse`
   ([linalg/rankUpdate.jl](src/linalg/rankUpdate.jl)), `cholUnblocked!` for
   `UniformBlockDiagonal` ([linalg/cholUnblocked.jl](src/linalg/cholUnblocked.jl)),
   and the `logdet` diagonal extraction ([linalg/logdet.jl](src/linalg/logdet.jl)).
   These become KernelAbstractions.jl kernels (portable across CUDA/ROCm/Metal/oneAPI).

3. **Dense LAPACK-level ops** (`potrf`/`trsm`/`syrk` on single large blocks) rely on
   the vendor library via the `LinearAlgebra`/GPUArrays dispatch. **Portability
   caveat:** this is solid on CUDA (cuSOLVER/cuBLAS, including *batched* potrf/trsm),
   but batched dense Cholesky/triangular-solve are **not** uniformly exposed across
   AMD/Metal/oneAPI through a common interface. So even with KA kernels, CUDA is
   realistically the only fully-working target at first; "portable" buys you the
   element-wise/batched custom kernels for free and a migration path, but not
   turnkey multi-vendor LAPACK.

4. **Sparse handling decision:** `BlockedSparse`/`SparseMatrixCSC` blocks are small
   and irregular — poor GPU citizens. Recommended policy: densify where the fill
   warrants it and keep genuinely-sparse models on CPU (i.e. `gpu(m)` refuses or
   warns when the block structure won't benefit).

5. **Block-loop type stability:** `m.L::Vector{AbstractMatrix}` already costs dynamic
   dispatch per block on CPU; on GPU this compounds launch overhead. A worthwhile
   model may need the block iteration restructured (e.g. grouping homogeneous blocks)
   to amortize launches.

---

## Variant A: restrict the GPU to the fixed-effects block

A low-risk partition, and a degenerate special case of the 3-block reformulation
below (the FE block is a sub-block of that form's dense tail).

β is profiled out, so "solving for the fixed effects" is not a separate solve during
fitting — it is the last block-row `i = k+1` of `updateL!`
([linearmixedmodel.jl:1407](src/linearmixedmodel.jl#L1407)), recomputed every
iteration:

1. `copyto!` + `rmulΛ!` on the `X'Z` cross-blocks (`block(k+1, j)`), each `p × q_j`
   with `q_j = S_j · nlevels_j`;
2. the `X'X` downdate via `rankUpdate!`/`mul!` of those cross-blocks (`syrk`/`gemm`);
3. `cholUnblocked!` (potrf) on the `p × p` block and `rdiv!` (trsm) of the
   cross-blocks against the random-effects factors.

`logdet` touches this block only under **REML**
([logdet.jl:29](src/linalg/logdet.jl#L29)); either way the result is a scalar.

**Why it is attractive:**
- Leaves every GPU-hostile thing on the CPU — the sparse / `UniformBlockDiagonal`
  random-effects Cholesky stays put. None of the risky custom batched kernels from
  the full-offload plan are needed.
- The FE blocks are plain dense `Matrix`, so the kernel surface is just `gemm` +
  `syrk` + `trsm` + `potrf` on **single dense matrices** — the best-supported GPU
  LAPACK ops, and the ones that survive the portability concern above (that gap was
  specifically *batched* factorization).
- Transfer stays cheap: `A`'s `X'Z`/`X'X` blocks are constant → resident after one
  upload. Per iteration: ship **down** the small RE factors (O(q), diagonal/
  block-diagonal) + θ; ship **back** scalars. No large arrays cross PCIe in the loop.

**The two real catches:**
- **Amdahl:** speedup is capped by the fraction of `updateL!` spent in the `i = k+1`
  block-row. Large when `p` and/or `q` are large; near-zero for the common case
  (few fixed effects, one big grouping factor) where the RE diagonal factorization
  and a `X'Z` rdiv against a *diagonal* already dominate cheaply.
- **Arithmetic intensity ~`p`:** the `X'X` downdate is a `p×q · q×p` syrk
  (~`2p²q` flops over ~`pq` data). For small `p` it is bandwidth-bound and the GPU
  win is just the bandwidth ratio minus launch + mid-factorization sync overhead —
  often a wash. It becomes compute-bound (clear GPU win) only when `p` is large:
  rich spline bases, many interactions, high-dimensional fixed-effects designs.

## Variant B (recommended enabling refactor): fit in the 3-block form of `_3blockL`

The package already contains the key structural insight, used post-hoc for
`leverage`: `_3blockL` ([linearmixedmodel.jl:751](src/linearmixedmodel.jl#L751))
collapses the `k(k+1)/2` irregular blocks of `L` into three:

- **B1** = `first(L)` — the *first* random-effects term's `Diagonal` /
  `UniformBlockDiagonal` factor, dimension `q₁ = S₁ · nlevels₁`. Reterms are ordered
  by decreasing number of levels, so this is the single largest,
  most sparse-structured block.
- **B2** — a dense **tail × q₁** rectangle stacking the cross-blocks of everything
  else against term 1.
- **B3** — a dense **tail × tail** lower triangle holding all the rest, where
  `tail = q₂ + … + q_k + p + 1` (secondary RE + fixed effects + response).

I.e. a 2×2 nested partition: one big block-diagonal term, one dense tail.
Reformulating `updateL!` around this structure turns the whole per-iteration hot
computation into **one dense Schur-complement downdate + one dense Cholesky**:

1. Factor **B1** — block-diagonal, cheap, stays on **CPU** (or batched trsm).
2. Solve **B2** against B1 — batched triangular solve over `nlevels₁` blocks.
3. **B3 ← B3 − B2·B1⁻¹·B2'** — a `tail × q₁` contraction, `~tail²·q₁` flops. The
   dominant BLAS3 op, scaling with the huge dimension `q₁`, as a **single** large
   `syrk`/`gemm` rather than a loop over many small blocks. **This is the GPU win.**
4. Dense `potrf` on **B3** (`~tail³`).
5. Read `logdet` (B1 diag + B3 tail diag) and `pwrss` (last element of B3).

**Why it supersedes Variant A** (FE block ⊂ tail):
- **Amdahl:** offloads the *entire* non-B1 cost — secondary RE, FE, and response —
  not just the FE sub-block.
- **Arithmetic intensity ~`tail`**, not `~p`: compute-bound at much more realistic
  model sizes.
- **Kernel surface:** one dense `gemm`/`syrk` + one dense `potrf` + batched trsm.
  No per-block dynamic dispatch, no custom sparse kernels — the best-supported,
  most portable GPU ops.
- **Transfers:** `A` in 3-block form is constant → resident after one upload; per
  iteration θ goes down, scalars come back.

**Catches:**
- **Privileges one dominant grouping factor.** Ideal for "one huge grouping factor
  + small everything else." For large *crossed* designs (subject and item both
  huge), `q₂` lands in the tail and the `tail³` Cholesky is itself expensive — the
  3-block form relocates the crossed-effects blowup into B3 rather than dodging it
  (GPU still helps there, but the cheap-tail story is gone).
- **A core-internals rewrite, not a pure extension.** `_3blockL` currently copies
  from an already-factored `L` post-hoc. Fitting in this form means re-expressing
  `A`/`L` construction, `updateL!`, and `objective` around it — the numerical heart
  of the package, far more invasive than the stub-based ext pattern.
- **Unconditionally densifies the tail.** Today crossed off-diagonal blocks can stay
  sparse ([arraytypes.jl:73](src/arraytypes.jl#L73)); a dense tail is a CPU
  memory/compute regression for models with sparse secondary structure. Treat the
  3-block layout as a *specialization*, selected when the tail is small or a GPU is
  engaged — not the unconditional replacement layout.

## Overall recommendation

1. **Do not** pursue a general/full GPU backend over the current block structure —
   for the dominant use cases it is slower and adds maintenance surface.
2. The strongest GPU-enabling change is the **3-block reformulation (Variant B)**:
   it is the representation in which per-iteration work becomes a single dense
   downdate + factorization — portable, high-intensity, cheap-transfer — with B1
   left on CPU. Note it is a core refactor with CPU-side benefits/risks of its own,
   not a pure extension; the GPU part *of* it can still live in an ext (device tail
   arrays + the same `gemm`/`syrk`/`potrf` calls).
3. Variant A (FE-only) is the degenerate case; only worth doing standalone if the
   3-block refactor is judged too invasive.
4. A full offload of the current block structure (batched kernels over
   `UniformBlockDiagonal`, GPU sparse) is dominated by Variant B and should not be
   attempted.
5. Sequence the investment: **first build a benchmark** (below) to locate the
   empirical crossover before writing any kernels — specifically the **fraction of
   `updateL!` time outside B1** (= ceiling for Variant B) and the tail dimension of
   target models. If B1 dominates, even an ideal GPU tail is capped low. If the
   crossover lands at model sizes nobody fits in practice, stop there — a perfectly
   good result.

---

## Verification / how to test the conclusions empirically

This doc commits no code. To validate the claims before investing:

1. **Profile the hot path on representative models** to confirm where time goes:
   - A small nested/scalar model (expect: tiny blocks, bandwidth-bound — GPU pointless).
   - A large crossed model (many subjects × many items) — confirm the forced-dense
     off-diagonal block dominates `updateL!`.
   - A large-`p` fixed-effects model — confirm `X'X` dominates.
   Use `@profile`/`ProfileView` or simple timing around `updateL!` over fixed θ.

2. **Microbenchmark the candidate kernels in isolation** at realistic block sizes:
   batched potrf/syrk on `Array{T,3}` (CPU loop vs `CUDA.jl` batched) and dense
   potrf/trsm/syrk (CPU LAPACK vs cuSOLVER). This gives the crossover dimension
   without touching MixedModels internals.

2a. **Prototype the 3-block objective out-of-place** (Variant B): `_3blockL`
   already builds B1/B2/B3 from a fitted model, so a standalone function can
   evaluate the objective via the Schur-complement recipe (steps 1–5 above) and be
   checked against `objective!` at several θ. Timing that prototype — CPU vs a GPU
   tail — measures Variant B's real ceiling without rewriting `updateL!`.

3. **Estimate end-to-end** by multiplying per-iteration kernel time by typical NLopt
   iteration counts (`m.optsum`) for the target models; compare to current CPU fit
   time from `@time fit(MixedModel, ...)`.

Crossover at block sizes/level-counts that occur in real datasets → worth a
prototype. Otherwise the analysis stands as "not worth it."
