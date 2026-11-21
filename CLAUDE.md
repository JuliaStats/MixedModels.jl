# Understanding tasks

Feel free to ask questions.

Do not assume that requested tasks are possible: feel free to inform me that a requested task is not possible in the given form or that there are only tedious workarounds.

# Julia development

Explicit `return` statements are required and the use of `import` is forbidden.

When running tests in Julia, you need to load the test environment. For interactive/REPL use, do this with `using TestEnv; TestEnv.activate()`. When using `Pkg.test()`, do not activate the test environment first — it manages that itself.

Always invoke Julia with `--startup-file=no` unless explicitly instructed otherwise.

You may invoke a specific Julia version with `+VERSION`, e.g. `+1.10` or `+1.12`. This argument must come immediately after `julia` and before any other flags.
If you are not testing something particular to a specific Julia version, use the minimum compatible version (as specified in Project.toml).

When checking coverage, you can use LocalCoverage.jl, which writes coverage to `coverage/lcov.info`.
