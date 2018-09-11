# Mixed-effects models in Julia

| **Documentation**                                                               | **PackageEvaluator**                                            | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][pkg-0.7-img]][pkg-0.7-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url] |

[![DOI](https://zenodo.org/badge/9106942.svg)](https://zenodo.org/badge/latestdoi/9106942)

## Installation

The package is registered in `METADATA.jl` and can be installed using `Pkg.add` or in the `Pkg` REPL, entered by typing `]` as the first character in a line.

```julia
(v1.0) pkg> add MixedModels
  Updating registry at `~/.julia/registries/General`
  Updating git-repo `https://github.com/JuliaRegistries/General.git`
 Resolving package versions...
  Updating `~/.julia/environments/v1.0/Project.toml`
 [no changes]
  Updating `~/.julia/environments/v1.0/Manifest.toml`
 [no changes]
```

The package provides the functions `lmm`, to create a linear mixed-effects model
from a formula/data specification, and `glmm` to create a generalized linear
mixed-effects model.  See [the documentation](http://dmbates.github.io/MixedModels.jl/latest)
for details.

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://dmbates.github.io/MixedModels.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://dmbates.github.io/MixedModels.jl/stable

[travis-img]: https://travis-ci.org/dmbates/MixedModels.jl.svg?branch=master
[travis-url]: https://travis-ci.org/dmbates/MixedModels.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/h227adt6ovd1u3sx/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/dmbates/documenter-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/github/dmbates/MixedModels.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/dmbates/MixedModels.jl?branch=master

[codecov-img]: https://codecov.io/github/dmbates/MixedModels.jl/badge.svg?branch=master
[codecov-url]: https://codecov.io/github/dmbates/MixedModels.jl?branch=master

[issues-url]: https://github.com/dmbates/MixedModels.jl/issues

[pkg-0.7-img]: http://pkg.julialang.org/badges/MixedModels_0.7.svg
[pkg-0.7-url]: http://pkg.julialang.org/?pkg=MixedModels
[pkg-1.0-img]: http://pkg.julialang.org/badges/MixedModels_1.0.svg
[pkg-1.0-url]: http://pkg.julialang.org/?pkg=MixedModels
