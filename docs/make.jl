using Documenter, Weave, StatsBase
using MixedModels
foreach(
    filename -> weave(
        joinpath(dirname(pathof(MixedModels)), "..", "docs", "jmd", filename),
        doctype = "github",
        fig_path = joinpath("docs", "assets"),
        fig_ext = ".svg",
        out_path = joinpath("docs", "src"),
    ),
    [
        "constructors.jmd",
        # "optimization.jmd",
        # "GaussHermite.jmd",
        # "bootstrap.jmd",
        # "SimpleLMM.jmd",
        # "MultipleTerms.jmd",
        # "SingularCovariance.jmd",
        # "SubjectItem.jmd",
    ],
)

makedocs(
    root = joinpath(dirname(pathof(MixedModels)), "..", "docs"),
    sitename = "MixedModels",
    pages = [
        "index.md",
        "constructors.md",
        # "optimization.md",
        # "GaussHermite.md",
        # "bootstrap.md",
        # "SimpleLMM.md",
        # "MultipleTerms.md",
        # "SingularCovariance.md",
        # "SubjectItem.md",
        # "benchmarks.md"
    ],
)

deploydocs(repo = "github.com/JuliaStats/MixedModels.jl.git", push_preview = true)
