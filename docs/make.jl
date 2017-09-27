using Documenter, MixedModels

makedocs(
    format = :html,
    sitename = "MixedModels.jl",
    modules = [MixedModels],
    pages = ["index.md",
             "constructors.md",
             "extractors.md",
             "fitting.md"]
)

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "mkdocs-material", "python-markdown-math"),
    repo = "github.com/dmbates/MixedModels.jl.git",
    julia = "0.6",
    osname = "linux"
)
