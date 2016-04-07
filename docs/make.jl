using Documenter, MixedModels

makedocs()

custom.deps() = run(`pip install --user pygments mkdocs mkdocs-material`)

deploydocs(
    deps = custom.deps,
    repo = "github.com/dmbates/MixedModels.jl.git",
    julia = "release",
    osname = "linux"
)
