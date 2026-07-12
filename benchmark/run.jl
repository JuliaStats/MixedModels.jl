using Pkg
Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()
using PkgBenchmark, MixedModels, Statistics
# Pkg.update() allows us to benchmark even when dependencies/compat requirements change
juliacmd = `$(Base.julia_cmd()) -O3 -e "using Pkg; Pkg.update()"`
config = BenchmarkConfig(; id="origin/HEAD", juliacmd)
# for many of the smaller models, we get a lot of noise at the default 5% tolerance
# TODO: specify a tune.json with per model time tolerances
export_markdown("benchmark.md", judge(MixedModels, config; verbose=true, retune=false, f=median, judgekwargs=(;time_tolerance=0.1, memory_tolerance=0.05)))
