MixedModels v4.25.4 Release Notes
==============================
- Added additional precompilation for rePCA. [#749]

MixedModels v4.25.3 Release Notes
==============================
- Fix a bug in the handling of rank deficiency in the `simulate[!]` code. This has important correctness implications for bootstrapping models with rank-deficient fixed effects (as can happen in the case of partial crossing of the fixed effects / missing cells). [#778]

MixedModels v4.25.2 Release Notes
==============================
- Use `public` keyword so that users don't see unnecessary docstring warnings on 1.11+. [#776]
- Fix accidental export of `dataset` and `datasets` and make them `public` instead. [#776]

MixedModels v4.25.1 Release Notes
==============================
- Use more sophisticated checks on property names in `restoreoptsum` to allow for optsums saved by pre-v4.25 versions to be used with this version and later. [#775]

MixedModels v4.25 Release Notes
==============================
- Add type notations in `pwrss(::LinearMixedModel)` and `logdet(::LinearMixedModel)` to enhance type inference. [#773]
- Take advantage of type parameter for `StatsAPI.weights(::LinearMixedModel{T})`. [#772]
- Fix use of kwargs in `fit!((::LinearMixedModel)`: [#772]
    - user-specified `σ` is actually used, defaulting to existing value
    - `REML` defaults to model's already specified REML value.
- Clean up code of keyword convenience constructor for `OptSummary`. [#772]
- Refactor thresholding parameters for forcing near-zero parameter values into `OptSummary`. [#772]

MixedModels v4.24.1 Release Notes
==============================
- Re-export accidentally dropped export `lrtest`. [#769]

MixedModels v4.24.0 Release Notes
==============================
* Properties for `GeneralizedLinearMixedModel` now default to delegation to the internal weighted `LinearMixedModel` when that property is not explicitly handled by `GeneralizedLinearMixedModel`. Previously, properties were delegated on an explicit basis, which meant that they had to be added manually as use cases were discovered. The downside to the new approach is that it is now possible to access properties whose definition in the LMM case doesn't match the GLMM definition when the GLMM definition hasn't been explicitly been implemented. [#767]

MixedModels v4.23.1 Release Notes
==============================
* Fix for `simulate!` when only the estimable coefficients for a rank-deficient model are provided. [#756]
* Improve handling of rank deficiency in GLMM. [#756]
* Fix display of GLMM bootstrap without a dispersion parameter. [#756]

MixedModels v4.23.0 Release Notes
==============================
* Support for rank deficiency in the parametric bootstrap. [#755]

MixedModels v4.22.5 Release Notes
==============================
* Use `muladd` where possible to enable fused multiply-add (FMA) on architectures with hardware support. FMA will generally improve computational speed and gives more accurate rounding. [#740]
* Replace broadcasted lambda with explicit loop and use `one`. This may result in a small performance improvement. [#738]

MixedModels v4.22.4 Release Notes
==============================
* Switch to explicit imports from all included packages (i.e. replace `using Foo` by `using Foo: Foo, bar, baz`) [#748]
* Reset parameter values before a `deepcopy` in a test (doesn't change test result) [#744]

MixedModels v4.22.3 Release Notes
==============================
* Comment out calls to `@debug` [#733]
* Update package versions in compat and change `Aqua.test_all` argument name [#733]

MixedModels v4.22.0 Release Notes
==============================
* Support for equal-tail confidence intervals for `MixedModelBootstrap`. [#715]
* Basic `show` methods for `MixedModelBootstrap` and `MixedModelProfile`. [#715]
* The `hide_progress` keyword argument to `parametricbootstrap` is now deprecated. Users should instead use `progress` (which is consistent with e.g. `fit`). [#717]

MixedModels v4.21.0 Release Notes
==============================
* Auto apply `Grouping()` to grouping variables that don't already have an explicit contrast. [#652]

MixedModels v4.20.0 Release Notes
==============================
* The `.tbl` property of a `MixedModelBootstrap` now includes the correlation parameters for lower triangular elements of the `λ` field. [#702]

MixedModels v4.19.0 Release Notes
==============================
* New method `StatsAPI.coefnames(::ReMat)` returns the coefficient names associated with each grouping factor. [#709]

MixedModels v4.18.0 Release Notes
==============================
* More user-friendly error messages when a formula contains variables not in the data. [#707]

MixedModels v4.17.0 Release Notes
==============================
* **EXPERIMENTAL** New kwarg `amalgamate` can be used to disable amalgation of random effects terms sharing a single grouping variable. Generally, `amalgamate=false` will result in a slower fit, but may improve convergence in some pathological cases. Note that this feature is experimental and changes to it are **not** considered breakings. [#673]
* More informative error messages when passing a `Distribution` or `Link` type instead of the desired instance. [#698]
* More informative error message on the intentional decision not to define methods for the coefficient of determination. [#698]
* **EXPERIMENTAL** Return `finitial` when PIRLS drifts into a portion of the parameter space that yields a (numerically) invalid covariance matrix. This recovery strategy may be removed in a future release. [#616]

MixedModels v4.16.0 Release Notes
==============================
* Support for check tolerances in deserialization. [#703]

MixedModels v4.15.0 Release Notes
==============================
* Support for different optimization criteria during the bootstrap. [#694]
* Support for combining bootstrap results with `vcat`. [#694]
* Support for saving and restoring bootstrap replicates with `savereplicates` and `restorereplicates`. [#694]

MixedModels v4.14.0 Release Notes
==============================
* New function `profile` for computing likelihood profiles for `LinearMixedModel`. The resultant `MixedModelProfile` can be then be used for computing confidence intervals with `confint`. Note that this API is still somewhat experimental and as such the internal storage details of `MixedModelProfile` may change in a future release without being considered breaking. [#639]
* A `confint(::LinearMixedModel)` method has been defined that returns Wald confidence intervals based on the z-statistic, i.e. treating the denominator degrees of freedom as infinite. [#639]

MixedModels v4.13.0 Release Notes
==============================
* `raneftables` returns a `NamedTuple` where the names are the grouping factor names and the values are some `Tables.jl`-compatible type.  This type has been changed to a `Table` from `TypedTables.jl`. [#682]

MixedModels v4.12.1 Release Notes
==============================
* Precompilation is now handled with `PrecompileTools` instead of `SnoopPrecompile`. [#681]
* An unnecessary explicit `Vararg` in an internal method has been removed. This removal eliminates a compiler warning about the deprecated `Vararg` pattern. [#680]

MixedModels v4.12.0 Release Notes
==============================
* The pirated method `Base.:/(a::AbstractTerm, b::AbstractTerm)` is no longer defined. This does not impact the use of `/` as a nesting term in `@formula` within MixedModels, only the programmatic runtime construction of formula, e.g. `term(:a) / term(:b)`. If you require `Base.:/`, then [`RegressionFormulae.jl`](https://github.com/kleinschmidt/RegressionFormulae.jl) provides this method. (Avoiding method redefinition when using `RegressionFormulae.jl` is the motivating reason for this change.) [#677]

MixedModels v4.11.0 Release Notes
==============================
* `raneftables` returns a `NamedTuple` where the names are the grouping factor names and the values are some `Tables.jl`-compatible type.  Currently this type is a `DictTable` from `TypedTables.jl`. [#634]

MixedModels v4.10.0 Release Notes
==============================
* Rank deficiency in prediction is now supported, both when the original model was fit to rank-deficient data and when the new data are rank deficient. The behavior is consistent but may be surprising when both old and new data are rank deficient. See the `predict` docstring for an example. [#676]
* Multithreading in `parametricbootstrap` with `use_threads` is now deprecated and a noop. With improvements in BLAS threading, multithreading at the Julia level did not help performance and sometimes hurt it. [#674]

MixedModels v4.9.0 Release Notes
==============================
* Support `StatsModels` 0.7, drop support for `StatsModels` 0.6. [#664]
* Revise code in benchmarks to work with recent Julia and PkgBenchmark.jl [#667]
* Julia minimum compat version raised to 1.8 because of BSplineKit [#665]

MixedModels v4.8.2 Release Notes
==============================
* Use `SnoopPrecompile` for better precompilation performance. This can dramatically increase TTFX, especially on Julia 1.9. [#663]

MixedModels v4.8.1 Release Notes
==============================
* Don't fit a GLM internally during construction of GLMM when the fixed effects are empty (better compatibility with
  `dropcollinear` kwarg in newer GLM.jl) [#657]

MixedModels v4.8.0 Release Notes
==============================
* Allow predicting from a single observation, as long as `Grouping()` is used for the grouping variables. The simplified implementation of `Grouping()` also removes several now unnecessary `StatsModels` methods that should not have been called directly by the user. [#653]

MixedModels v4.7.3 Release Notes
==============================
* More informative error message for formulae lacking random effects [#651]

MixedModels v4.7.2 Release Notes
==============================
* Replace separate calls to `copyto!` and `scaleinflate!` in `updateL!` with `copyscaleinflate!` [#648]

MixedModels v4.7.1 Release Notes
==============================
* Avoid repeating initial objective evaluation in `fit!` method for `LinearMixedModel`
* Ensure that the number of function evaluations from NLopt corresponds to `length(m.optsum.fitlog) when `isone(thin)`. [#637]

MixedModels v4.7.0 Release Notes
==============================
* Relax type restriction for filename in `saveoptsum` and `restoreoptsum!`. Users can now pass any type with an appropriate `open` method, e.g. `<:AbstractPath`. [#628]

MixedModels v4.6.5 Release Notes
========================
* Attempt recovery when the initial parameter values lead to an invalid covariance matrix by rescaling [#615]
* Return `finitial` when the optimizer drifts into a portion of the parameter space that yields a (numerically) invalid covariance matrix [#615]

MixedModels v4.6.4 Release Notes
========================
* Support transformed responses in `predict` [#614]
* Simplify printing of BLAS configuration in tests. [#597]

MixedModels v4.6.3 Release Notes
========================
* Add precompile statements to speed up first `LinearMixedModel` and Bernoulli `GeneralizedLinearModel` fit [#608]

MixedModels v4.6.2 Release Notes
========================
* Efficiency improvements in `predict`, both in memory and computation [#604]
* Changed the explanation of `predict`'s keyword argument `new_re_levels` in a way that is clearer about the behavior when there are multiple grouping variables. [#603]
* Fix the default behavior of `new_re_levels=:missing` to match the docstring. Previously, the default was `:population`, in disagreement with the docstring. [#603]

MixedModels v4.6.1 Release Notes
========================
* Loosen type restriction on `shortestcovint(::MixedModelBootstrap)` to `shortestcovint(::MixedModelFitCollection)`. [#598]

MixedModels v4.6.0 Release Notes
========================
* Experimental support for initializing `GeneralizedLinearMixedModel` fits from a linear mixed model instead of a marginal (non-mixed) generalized linear model. [#588]

MixedModels v4.5.0 Release Notes
========================
* Allow constructing a `GeneralizedLinearMixedModel` with constant response, but don't update the ``L`` matrix nor initialize its deviance. This allows for the model to still be used for simulation where the response will be changed before fitting. [#578]
* Catch `PosDefException` during the first optimization step and throw a more informative `ArgumentError` if the response is constant. [#578]

MixedModels v4.4.1 Release Notes
========================
* Fix type parameterization in MixedModelsBootstrap to support models with a mixture of correlation structures (i.e. `zerocorr` in some but not all RE terms) [#577]

MixedModels v4.4.0 Release Notes
========================
* Add a constructor for the abstract type `MixedModel` that delegates to `LinearMixedModel` or `GeneralizedLinearMixedModel`. [#572]
* Compat for Arrow.jl 2.0 [#573]

MixedModels v4.3.0 Release Notes
========================
* Add support for storing bootstrap results with lower precision [#566]
* Improved support for zerocorr models in the bootstrap [#570]

MixedModels v4.2.0 Release Notes
========================
* Add support for zerocorr models to the bootstrap [#561]
* Add a `Base.length(::MixedModelsFitCollection)` method  [#561]

MixedModels v4.1.0 Release Notes
========================
* Add support for specifying a fixed value of `σ`, the residual standard deviation,
  in `LinearMixedModel`. `fit` takes a keyword-argument `σ`. `fit!` does not expose `σ`,
  but `σ` can be changed after model construction by setting `optsum.sigma`. [#551]
* Add support for logging the non-linear optimizer's steps via a `thin`
  keyword-argument for `fit` and `fit!`. The default behavior is 'maximal' thinning,
  such that only the initial and final values are stored. `OptSummary` has a new field
  `fitlog` that contains the aforementioned log as a  vector of tuples of parameter and
  objective values.[#552]
* Faster version of `leverage` for `LinearMixedModel` allowing for experimentation
  using the sum of the leverage values as an empirical degrees of freedom for the
  model. [#553], see also [#535]
* Optimized version of `condVar` with an additional method for extracting only the
  conditional variances associated with a single grouping factor. [#545]

MixedModels v4.0.0 Release Notes
========================
* Drop dependency on `BlockArrays` and use a `Vector` of matrices to represent
  the lower triangle in packed, row-major order. The non-exported function `block`
  can be used for finding the corresponding `Vector` index of a block. [#456]
* `simulate!` now marks the modified model as being unfitted.
* Deprecated and unused `named` argument removed from `ranef` [#469]
* Introduce an abstract type for collections of fits `MixedModelFitCollection`,
  and make `MixedModelBootstrap` a subtype of it. Accordingly, rename the `bstr`
  field to `fits`. [#465]
* The response (dependent variable) is now stored internally as part of the
  the renamed `FeMat` field, now called `Xymat` [#464]
* Replace internal `statscholesky` and `statsqr` functions for determining the
  rank of `X` by `statsrank`. [#479]
* Artifacts are now loaded lazily: the test data loaded via `dataset` is
  downloaded on first use [#486]
* `ReMat` and `PCA` now support covariance factors (`λ`) that are `LowerTriangular`
  or `Diagonal`. This representation is both more memory efficient and
  enables additional computational optimizations for particular covariance
  structures.[#489]
* `GeneralizedLinearMixedModel` now includes the response distribution as one
  of its type parameters. This will allow dispatching on the model family and may allow
  additional specialization in the future.[#490]
* `saveoptsum` and `restoreoptsum!` provide for saving and restoring the `optsum`
  field of a `LinearMixedModel` as a JSON file, allowing for recreating a model fit
  that may take a long time for parameter optimization. [#506]
* Verbose output now uses `ProgressMeter`, which gives useful information about the timing
  of each iteration and does not flood stdio. The `verbose` argument has been renamed `progress`
  and the default changed to `true`. [#539]
* Support for Julia < 1.6 has been dropped. [#539]
* New `simulate`, `simulate!` and `predict` methods for simulating and
  predicting responses to new data. [#427]

Run-time formula syntax
-----------------------

* It is now possible to construct `RandomEffectsTerm`s at run-time from `Term`s
  (methods for `Base.|(::AbstractTerm, ::AbstractTerm)` added) [#470]
* `RandomEffectsTerm`s can have left- and right-hand side terms that are
  "non-concrete", and `apply_schema(::RandomEffectsTerm, ...)` works more like
  other StatsModels.jl `AbstractTerm`s [#470]
* Methods for `Base./(::AbstractTerm, ::AbstractTerm)` are added, allowing
  nesting syntax to be used with `Term`s at run-time as well [#470]

MixedModels v3.9.0 Release Notes
========================
* Add support for `StatsModels.formula` [#536]
* Internal method `allequal` renamed to `isconstant` [#537]

MixedModels v3.8.0 Release Notes
========================
* Add support for NLopt `maxtime` option to `OptSummary` [#524]

MixedModels v3.7.1 Release Notes
========================
* Add support for `condVar` for models with a BlockedSparse structure [#523]

MixedModels v3.7.0 Release Notes
========================
* Add `condVar` and `condVartables` for computing the conditional variance on the random effects [#492]
* Bugfix: store the correct lower bound for GLMM bootstrap, when the original model was fit with `fast=false` [#518]

MixedModels v3.6.0 Release Notes
========================
* Add `likelihoodratiotest` method for comparing non-mixed (generalized) linear models to (generalized) linear mixed models [#508].

MixedModels v3.5.2 Release Notes
========================
* Explicitly deprecate vestigial `named` kwarg in `ranef` in favor of `raneftables` [#507].

MixedModels v3.5.1 Release Notes
========================
* Fix MIME show methods for models with random-effects not corresponding to a fixed effect [#501].

MixedModels v3.5.0 Release Notes
========================
* The Progressbar for `parametricbootstrap` and `replicate` is not displayed
  when in a non-interactive (i.e. logging) context. The progressbar can also
  be manually disabled with `hide_progress=true`.[#495]
* Threading in `parametricbootstrap` now uses a `SpinLock` instead of a `ReentrantLock`.
  This improves performance, but care should be taken when nesting spin locks. [#493]
* Single-threaded use of `paramatricbootstrap` now works when nested within a larger
  multi-threaded context (e.g. `Threads.@threads for`). (Multi-threaded `parametricbootstrap`
  worked and continues to work within a nested threading context.) [#493]

MixedModels v3.4.1 Release Notes
========================
* The value of a named `offset` argument to `GeneralizedLinearMixedModel`,
  which previously was ignored [#453], is now handled properly. [#482]

MixedModels v3.4.0 Release Notes
========================
* `shortestcovint` method for `MixedModelsBootstrap` [#484]

MixedModels v3.3.0 Release Notes
========================
* HTML and LaTeX `show` methods for `MixedModel`, `BlockDescription`,
  `LikelihoodRatioTest`, `OptSummary` and `VarCorr`. Note that the interface for
  these is not yet completely stable. In particular, rounding behavior may
  change. [#480]

MixedModels v3.2.0 Release Notes
========================
* Markdown `show` methods for `MixedModel`, `BlockDescription`,
  `LikelihoodRatioTest`, `OptSummary` and `VarCorr`. Note that the interface for
  these is not yet completely stable. In particular, rounding behavior may
  change. White-space padding within Markdown may also change, although this
  should not impact rendering of the Markdown into HTML or LaTeX.  The
  Markdown presentation of a `MixedModel` is much more compact than the
  REPL summary. If the REPL-style presentation is desired, then this can
  be assembled from the Markdown output from `VarCorr` and `coeftable` [#474].

MixedModels v3.1.4 Release Notes
========================
* [experimental] Additional convenience constructors for `LinearMixedModel` [#449]

MixedModels v3.1.3 Release Notes
========================
* Compatibility updates
* `rankUpdate!` method for `UniformBlockDiagonal` by `Dense` [#447]

MixedModels v3.1.2 Release Notes
========================
* Compatibility updates
* `rankUpdate!` method for `Diagonal` by `Dense` [#446]
* use eager (install-time) downloading of `TestData` artifact to avoid compatibility
  issues with `LazyArtifacts` in Julia 1.6 [#444]

MixedModels v3.1.1 Release Notes
========================
* Compatibility updates
* Better `loglikelihood(::GeneralizedLinearMixedModel)` which will work for models with
  dispersion parameters [#419]. Note that fitting such models is still problematic.

MixedModels v3.1 Release Notes
========================

* `simulate!` and thus `parametricbootstrap` methods for `GeneralizedLinearMixedModel` [#418].
* Documented inconsistent behavior in `sdest` and `varest` `GeneralizedLinearMixedModel` [#418].

MixedModels v3.0.2 Release Notes
========================

* Compatibility updates
* Minor updates for formatting in various `show` method for `VarCorr`.

MixedModels v3.0 Release Notes
========================

New formula features
---------------------

* Nested grouping factors can be written using the `/` notation, as in
  `@formula(strength ~ 1 + (1|batch/cask))` as a model for the `pastes` dataset.
* The `zerocorr` function converts a vector-valued random effects term from
  correlated random effects to uncorrelated. (See the `constructors` section of the docs.)
* The `fulldummy` function can be applied to a factor to obtain a redundant encoding
  of a categorical factor as a complete set of indicators plus an intercept.  This is only
  practical on the left-hand side of a random-effects term.  (See the `constructors` section
  of the docs.)
* `Grouping()` can be used as a contrast for a categorical array in the `contrasts` dictionary.
  Doing so bypasses creation of contrast matrix, which, when the number of levels is large,
  may cause memory overflow.  As the name implies, this is used for grouping factors. [#339]

Rank deficiency
--------------------

* Checks for rank deficiency in the model matrix for the fixed-effects
  parameters have been improved.
* The distinction between `coef`, which always returns a full set of coefficients
  in the original order, and `fixef`, which returns possibly permuted and
  non-redundant coefficients, has been made consistent across models.

Parametric bootstrap
--------------------

* The `parametricbootstrap` function and the struct it produces have been
  extensively reworked for speed and convenience.
  See the `bootstrap` section of the docs.

Principal components
--------------------

* The PCA property for `MixedModel` types provides principal components from
  the correlation of the random-effects distribution (as opposed to the covariance)
* Factor loadings are included in the `print` method for the `PCA` struct.

`ReMat` and `FeMat` types
----------------------------------

* An `AbstractReMat` type has now been introduced to support [#380] work on constrained
  random-effects structures and random-effects structures appropriate for applications
  in GLM-based deconvolution as used in fMRI and EEG (see e.g. [unfold.jl](https://github.com/unfoldtoolbox/unfold.jl).)
* Similarly, a constructor for `FeMat{::SparseMatrixCSC,S}` has been introduced [#309].
  Currently, this constructor assumes a full-rank matrix, but the work on rank
  deficiency may be extended to this constructor as well.
* Analogous to `AbstractReMat`, an `AbstractReTerm <: AbstractTerm` type  has been introduced [#395].
  Terms created with `zerocorr` are of type `ZeroCorr <: AbstractReTerm`.

Availability of test data sets
------------------------------

* Several data sets from the literature were previously saved in `.rda` format
  in the `test` directory and read using the `RData` package. These are now available
  in an `Artifact` in the [`Arrow`](https://github.com/JuliaData/Arrow.jl.git) format [#382].
* Call `MixedModels.datasets()` to get a listing of the names of available datasets
* To load, e.g. the `dyestuff` data, use `MixedModels.dataset(:dyestuff)`
* Data sets are loaded using `Arrow.Table` which returns a column table.  Wrap the
  call in `DataFrame` if you prefer a `DataFrame`.
* Data sets are cached and multiple calls to `MixedModels.dataset()` for the same
  data set have very low overhead after the first call.

Replacement capabilities
------------------------

* `describeblocks` has been dropped in favor of the `BlockDescription` type
* The `named` argument to `ranef` has been dropped in favor of `raneftables`

Package dependencies
--------------------

* Several package dependencies have been dropped, including `BlockDiagonals`, `NamedArrays` [#390],
  `Printf` and `Showoff`
* Dependencies on `Arrow` [#382], `DataAPI` [#384], and `PooledArrays` have been added.

<!--- generated by NEWS-update.jl: -->
[#309]: https://github.com/JuliaStats/MixedModels.jl/issues/309
[#339]: https://github.com/JuliaStats/MixedModels.jl/issues/339
[#380]: https://github.com/JuliaStats/MixedModels.jl/issues/380
[#382]: https://github.com/JuliaStats/MixedModels.jl/issues/382
[#384]: https://github.com/JuliaStats/MixedModels.jl/issues/384
[#390]: https://github.com/JuliaStats/MixedModels.jl/issues/390
[#395]: https://github.com/JuliaStats/MixedModels.jl/issues/395
[#418]: https://github.com/JuliaStats/MixedModels.jl/issues/418
[#419]: https://github.com/JuliaStats/MixedModels.jl/issues/419
[#427]: https://github.com/JuliaStats/MixedModels.jl/issues/427
[#444]: https://github.com/JuliaStats/MixedModels.jl/issues/444
[#446]: https://github.com/JuliaStats/MixedModels.jl/issues/446
[#447]: https://github.com/JuliaStats/MixedModels.jl/issues/447
[#449]: https://github.com/JuliaStats/MixedModels.jl/issues/449
[#453]: https://github.com/JuliaStats/MixedModels.jl/issues/453
[#456]: https://github.com/JuliaStats/MixedModels.jl/issues/456
[#464]: https://github.com/JuliaStats/MixedModels.jl/issues/464
[#465]: https://github.com/JuliaStats/MixedModels.jl/issues/465
[#469]: https://github.com/JuliaStats/MixedModels.jl/issues/469
[#470]: https://github.com/JuliaStats/MixedModels.jl/issues/470
[#474]: https://github.com/JuliaStats/MixedModels.jl/issues/474
[#479]: https://github.com/JuliaStats/MixedModels.jl/issues/479
[#480]: https://github.com/JuliaStats/MixedModels.jl/issues/480
[#482]: https://github.com/JuliaStats/MixedModels.jl/issues/482
[#484]: https://github.com/JuliaStats/MixedModels.jl/issues/484
[#486]: https://github.com/JuliaStats/MixedModels.jl/issues/486
[#489]: https://github.com/JuliaStats/MixedModels.jl/issues/489
[#490]: https://github.com/JuliaStats/MixedModels.jl/issues/490
[#492]: https://github.com/JuliaStats/MixedModels.jl/issues/492
[#493]: https://github.com/JuliaStats/MixedModels.jl/issues/493
[#495]: https://github.com/JuliaStats/MixedModels.jl/issues/495
[#501]: https://github.com/JuliaStats/MixedModels.jl/issues/501
[#506]: https://github.com/JuliaStats/MixedModels.jl/issues/506
[#507]: https://github.com/JuliaStats/MixedModels.jl/issues/507
[#508]: https://github.com/JuliaStats/MixedModels.jl/issues/508
[#518]: https://github.com/JuliaStats/MixedModels.jl/issues/518
[#523]: https://github.com/JuliaStats/MixedModels.jl/issues/523
[#524]: https://github.com/JuliaStats/MixedModels.jl/issues/524
[#535]: https://github.com/JuliaStats/MixedModels.jl/issues/535
[#536]: https://github.com/JuliaStats/MixedModels.jl/issues/536
[#537]: https://github.com/JuliaStats/MixedModels.jl/issues/537
[#539]: https://github.com/JuliaStats/MixedModels.jl/issues/539
[#545]: https://github.com/JuliaStats/MixedModels.jl/issues/545
[#551]: https://github.com/JuliaStats/MixedModels.jl/issues/551
[#552]: https://github.com/JuliaStats/MixedModels.jl/issues/552
[#553]: https://github.com/JuliaStats/MixedModels.jl/issues/553
[#561]: https://github.com/JuliaStats/MixedModels.jl/issues/561
[#566]: https://github.com/JuliaStats/MixedModels.jl/issues/566
[#570]: https://github.com/JuliaStats/MixedModels.jl/issues/570
[#572]: https://github.com/JuliaStats/MixedModels.jl/issues/572
[#573]: https://github.com/JuliaStats/MixedModels.jl/issues/573
[#577]: https://github.com/JuliaStats/MixedModels.jl/issues/577
[#578]: https://github.com/JuliaStats/MixedModels.jl/issues/578
[#588]: https://github.com/JuliaStats/MixedModels.jl/issues/588
[#597]: https://github.com/JuliaStats/MixedModels.jl/issues/597
[#598]: https://github.com/JuliaStats/MixedModels.jl/issues/598
[#603]: https://github.com/JuliaStats/MixedModels.jl/issues/603
[#604]: https://github.com/JuliaStats/MixedModels.jl/issues/604
[#608]: https://github.com/JuliaStats/MixedModels.jl/issues/608
[#614]: https://github.com/JuliaStats/MixedModels.jl/issues/614
[#615]: https://github.com/JuliaStats/MixedModels.jl/issues/615
[#616]: https://github.com/JuliaStats/MixedModels.jl/issues/616
[#628]: https://github.com/JuliaStats/MixedModels.jl/issues/628
[#634]: https://github.com/JuliaStats/MixedModels.jl/issues/634
[#637]: https://github.com/JuliaStats/MixedModels.jl/issues/637
[#639]: https://github.com/JuliaStats/MixedModels.jl/issues/639
[#648]: https://github.com/JuliaStats/MixedModels.jl/issues/648
[#651]: https://github.com/JuliaStats/MixedModels.jl/issues/651
[#652]: https://github.com/JuliaStats/MixedModels.jl/issues/652
[#653]: https://github.com/JuliaStats/MixedModels.jl/issues/653
[#657]: https://github.com/JuliaStats/MixedModels.jl/issues/657
[#663]: https://github.com/JuliaStats/MixedModels.jl/issues/663
[#664]: https://github.com/JuliaStats/MixedModels.jl/issues/664
[#665]: https://github.com/JuliaStats/MixedModels.jl/issues/665
[#667]: https://github.com/JuliaStats/MixedModels.jl/issues/667
[#673]: https://github.com/JuliaStats/MixedModels.jl/issues/673
[#674]: https://github.com/JuliaStats/MixedModels.jl/issues/674
[#676]: https://github.com/JuliaStats/MixedModels.jl/issues/676
[#677]: https://github.com/JuliaStats/MixedModels.jl/issues/677
[#680]: https://github.com/JuliaStats/MixedModels.jl/issues/680
[#681]: https://github.com/JuliaStats/MixedModels.jl/issues/681
[#682]: https://github.com/JuliaStats/MixedModels.jl/issues/682
[#694]: https://github.com/JuliaStats/MixedModels.jl/issues/694
[#698]: https://github.com/JuliaStats/MixedModels.jl/issues/698
[#702]: https://github.com/JuliaStats/MixedModels.jl/issues/702
[#703]: https://github.com/JuliaStats/MixedModels.jl/issues/703
[#707]: https://github.com/JuliaStats/MixedModels.jl/issues/707
[#709]: https://github.com/JuliaStats/MixedModels.jl/issues/709
[#715]: https://github.com/JuliaStats/MixedModels.jl/issues/715
[#717]: https://github.com/JuliaStats/MixedModels.jl/issues/717
[#733]: https://github.com/JuliaStats/MixedModels.jl/issues/733
[#738]: https://github.com/JuliaStats/MixedModels.jl/issues/738
[#740]: https://github.com/JuliaStats/MixedModels.jl/issues/740
[#744]: https://github.com/JuliaStats/MixedModels.jl/issues/744
[#748]: https://github.com/JuliaStats/MixedModels.jl/issues/748
[#749]: https://github.com/JuliaStats/MixedModels.jl/issues/749
[#755]: https://github.com/JuliaStats/MixedModels.jl/issues/755
[#756]: https://github.com/JuliaStats/MixedModels.jl/issues/756
[#767]: https://github.com/JuliaStats/MixedModels.jl/issues/767
[#769]: https://github.com/JuliaStats/MixedModels.jl/issues/769
[#772]: https://github.com/JuliaStats/MixedModels.jl/issues/772
[#773]: https://github.com/JuliaStats/MixedModels.jl/issues/773
[#775]: https://github.com/JuliaStats/MixedModels.jl/issues/775
[#776]: https://github.com/JuliaStats/MixedModels.jl/issues/776
[#778]: https://github.com/JuliaStats/MixedModels.jl/issues/778
