using Documenter
using MixedModels
using FiniteDiff
using ForwardDiff
using StatsAPI
using StatsBase

makedocs(;
    sitename="MixedModels",
    format=Documenter.HTML(; size_threshold=500_000, size_threshold_warn=250_000),
    doctest=true,
    # pagesonly=true,
    # warnonly=true,
    # warnonly=[:cross_references],
    pages=[
        "index.md",
        "constructors.md",
        "optimization.md",
        "GaussHermite.md",
        "prediction.md",
        "bootstrap.md",
        "rankdeficiency.md",
        "mime.md",
        "derivatives.md",
        "formula_syntax.md",
        "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaStats/MixedModels.jl.git", push_preview=true, devbranch="main"
)
