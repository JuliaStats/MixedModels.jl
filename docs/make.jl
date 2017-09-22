using Documenter, MixedModels

makedocs(
    format = :html,
    sitename = "MixedModels.jl",
    modules = [MixedModels],
    pages = ["index.md"] #=,
              "fitting.md",
              "bootstraps.md"] =#
)

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "mkdocs-material", "python-markdown-math"),
    repo = "github.com/dmbates/MixedModels.jl.git",
    julia = "0.6",
    osname = "linux"
)
