using Documenter, MixedModels, StatsBase

makedocs(
    sitename = "MixedModels",
    pages = ["index.md",
             "constructors.md",
             "optimization.md",
             "GaussHermite.md",
             "bootstrap.md",
#             "SimpleLMM.md",
#             "MultipleTerms.md",
#             "SingularCovariance.md",
#             "SubjectItem.md",
#             "benchmarks.md"
    ]
)

deploydocs(
    repo    = "github.com/dmbates/MixedModels.jl.git"
)
