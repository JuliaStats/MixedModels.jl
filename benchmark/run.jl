using Pkg
Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()
using PkgBenchmark, MixedModels
# explicit `Pkg.add` is a crutch until we've got a good base on main
# Pkg.update() allows us to benchmark even when dependencies/compat requirements change
juliacmd = `$(Base.julia_cmd()) -O3 -e "using Pkg; Pkg.update(); Pkg.add([\"BenchmarkTools\", \"StatsModels\"])"`
config = BenchmarkConfig(; id="origin/HEAD", juliacmd)
export_markdown("benchmark.md", judge(MixedModels, config; verbose=true))
