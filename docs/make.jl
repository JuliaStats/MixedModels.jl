
using Documenter, Weave, StatsBase
using MixedModels
foreach(filename -> weave(joinpath("docs", "jmd", filename),
                          doctype = "github",
                          fig_path = "assets",
                          fig_ext = ".svg",
                          out_path = "src"),
        ["constructors.jmd",
         "optimization.jmd",
         "GaussHermite.jmd",
         "bootstrap.jmd",
         # "SimpleLMM.jmd",
         # "MultipleTerms.jmd",
         # "SingularCovariance.jmd",
         # "SubjectItem.jmd",
        ])

makedocs(
    sitename = "MixedModels",
    pages = ["index.md",
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

deploydocs(
    repo = "github.com/JuliaStats/MixedModels.jl.git",
    push_preview = true,
    )
