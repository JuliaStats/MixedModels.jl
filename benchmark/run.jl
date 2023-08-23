using Pkg
Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()
using PkgBenchmark, MixedModels, Statistics
# Pkg.update() allows us to benchmark even when dependencies/compat requirements change
juliacmd = `$(Base.julia_cmd()) -O3 -e "using Pkg; Pkg.update()"`
config = BenchmarkConfig(; id="origin/HEAD", juliacmd)
export_markdown("benchmark.md", judge(MixedModels, config; verbose=true, retune=true, f=median))
