using Documenter
using MixedModels
using StatsAPI
using StatsBase
using DocumenterVitepress

makedocs(;
    sitename="MixedModels",
    doctest=true,
    pages=[
        "index.md",
        "Articles" => [
            "constructors.md",
            "optimization.md",
            "GaussHermite.md",
            "prediction.md",
            "bootstrap.md",
            "rankdeficiency.md",
            "mime.md"
        ],
        "api.md"
    ],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/ajinkya-k/MixedModels.jl",
    ),
    clean = true
)

deploydocs(;
    target = "build",
    repo="github.com/ajinkya-k/MixedModels.jl.git", push_preview=true, devbranch="ahk/doc-vite"
)
