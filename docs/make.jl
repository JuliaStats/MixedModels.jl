using Documenter, MixedModels

makedocs(modules = [MixedModels], )

deploydocs(
    repo = "github.com/dmbates/MixedModels.jl.git",
    julia = "release",
    osname = "linux"
)
