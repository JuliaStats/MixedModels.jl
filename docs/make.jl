using Documenter, MixedModels, StatsBase

makedocs(
    format = :html,
    sitename = "MixedModels",
    pages = ["index.md",
             "constructors.md",
             "extractors.md",
             "bootstrap.md"]
)

deploydocs(
    repo    = "github.com/dmbates/MixedModels.jl.git",
    julia   = "0.6",
    osname  = "linux",
    target  = "build",
    deps    = nothing,
    make    = nothing
)
