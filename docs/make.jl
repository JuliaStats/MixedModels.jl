using Documenter
using StatsBase
using MixedModels

makedocs(
    root = joinpath(dirname(pathof(MixedModels)), "..", "docs"),
    sitename = "MixedModels",
    pages = [
        "index.md",
        "constructors.md",
        "optimization.md",
        "GaussHermite.md",
        "bootstrap.md",
        # "SimpleLMM.md",
        # "MultipleTerms.md",
        # "SingularCovariance.md",
        # "SubjectItem.md",
        # "benchmarks.md"
    ],
)

deploydocs(repo = "github.com/JuliaStats/MixedModels.jl.git", push_preview = true)
