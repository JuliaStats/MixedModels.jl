using Documenter, MixedModels

makedocs()

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "mkdocs-material"),
    repo = "github.com/dmbates/MixedModels.jl.git",
    julia = "release",
    osname = "linux"
)
