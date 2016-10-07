# Mixed-effects models in Julia

| **Documentation**                                                               | **PackageEvaluator**                                            | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][pkg-0.4-img]][pkg-0.4-url] [![][pkg-0.5-img]][pkg-0.5-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][coveralls-img]][coveralls-url] |


## Installation

The package is registered in `METADATA.jl` and so can be installed with `Pkg.add`.

```julia
julia> Pkg.add("MixedModels")
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

[issues-url]: https://github.com/dmbates/MixedModels.jl/issues

[pkg-0.4-img]: http://pkg.julialang.org/badges/MixedModels_0.4.svg
[pkg-0.4-url]: http://pkg.julialang.org/?pkg=MixedModels
[pkg-0.5-img]: http://pkg.julialang.org/badges/MixedModels_0.5.svg
[pkg-0.5-url]: http://pkg.julialang.org/?pkg=MixedModels
