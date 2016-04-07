using Documenter, MixedModels

makedocs()

deploydocs(
    repo = "github.com/dmbates/MixedModels.jl.git",
    julia = "release",
    osname = "linux"
)
