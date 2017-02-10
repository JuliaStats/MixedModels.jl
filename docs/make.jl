using Documenter, MixedModels

makedocs()

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "mkdocs-material", "python-markdown-math"),
    repo = "github.com/dmbates/MixedModels.jl.git",
    julia = "0.5",
    osname = "linux"
)
