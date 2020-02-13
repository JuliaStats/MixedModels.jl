using Documenter, Weave, MixedModels, StatsBase

for file in ["constructors.jmd",
             "optimization.jmd",
             "GaussHermite.jmd",
             "bootstrap.jmd",
             # "SimpleLMM.jmd",
             # "MultipleTerms.jmd",
             # "SingularCovariance.jmd",
             # "SubjectItem.jmd",
             ]
    weave(joinpath("jmd", file),
          doctype = "github",
          fig_path = "assets",
          fig_ext = ".svg",
          out_path = "src")
end

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
    throw_errors = true,
    )

deploydocs(
    repo = "github.com/JuliaStats/MixedModels.jl.git",
    push_preview = true,
    )
