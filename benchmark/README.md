# Benchmarks

This directory contains a few different benchmarking entrypoints with different purposes.

## Scripts

`benchmarks.jl`

- Defines the `PkgBenchmark.jl` benchmark suite for model fitting.
- Uses a curated set of datasets, formulas, and contrasts.
- Intended to be consumed by `PkgBenchmark`, not usually run directly by hand.

`run.jl`

- Runs the `PkgBenchmark.jl` suite and writes a markdown comparison report.
- Intended for branch or commit comparisons of the predefined fitting benchmarks.

`fitted_predict.jl`

- Runs focused benchmarks for `fitted!`, `fitted`, and `predict`.
- Takes the dataset and model formula as command-line arguments so it is not tied to one hard-coded example.
- Can optionally benchmark `predict(...; new_re_levels=:population)` by perturbing one grouping variable.
- Supports both LMMs and GLMMs via optional `--family` and `--link` arguments.
- Supports `text`, `json`, and `markdown` output.

Example:

```bash
julia --project=. benchmark/fitted_predict.jl \
  --dataset sleepstudy \
  --formula 'reaction ~ 1 + days + (1 | subj)' \
  --group subj \
  --iterations 30 \
  --format text
```

GLMM example:

```bash
julia --project=. benchmark/fitted_predict.jl \
  --dataset contra \
  --formula 'use ~ 1 + urban + livch + age + abs2(age) + (1 | dist)' \
  --family Bernoulli \
  --link LogitLink \
  --iterations 30 \
  --format json
```

`compare_fitted_predict.jl`

- Runs `fitted_predict.jl` against two different project directories and reports the delta.
- Useful for comparing a feature branch, worktree, or detached checkout against a baseline.
- Supports `text`, `json`, and `markdown` output.

Example:

```bash
julia --project=. benchmark/compare_fitted_predict.jl \
  --lhs /tmp/MixedModels-main-bench \
  --rhs /Users/palday/Code/MixedModels.jl \
  --lhs-label baseline \
  --rhs-label current \
  --dataset sleepstudy \
  --formula 'reaction ~ 1 + days + (1 | subj)' \
  --group subj \
  --iterations 30 \
  --format markdown
```

`snoopspeed.sh`

- Small ad hoc shell script for startup and compilation timing experiments.
- Not part of the structured benchmark workflow.

## Notes

- The `fitted_predict.jl` and `compare_fitted_predict.jl` scripts intentionally benchmark warmed code paths.
- The comparison script uses the current checkout's `fitted_predict.jl` script while switching `--project` between the two target directories. This keeps the benchmark harness itself constant across comparisons.
