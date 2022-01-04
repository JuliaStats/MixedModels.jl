using Documenter
using StatsBase
using MixedModels

makedocs(
#    root = joinpath(dirname(pathof(MixedModels)), "..", "docs"),
    sitename = "MixedModels",
    doctest = true,
    pages = [
        "index.md",
        "constructors.md",
        "optimization.md",
        "GaussHermite.md",
        "prediction.md",
        "bootstrap.md",
        "rankdeficiency.md",
        "mime.md",
        "api.md",
    ],
)

deploydocs(;repo = "github.com/JuliaStats/MixedModels.jl.git", push_preview = true, devbranch = "main")
