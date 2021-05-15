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
* added `Distributions.Distribution` and `GLM.Link` methods for `GeneralizedLinearMixedModel` [#510]

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
  in GLM-based decovolution as used in fMRI and EEG (see e.g. [unfold.jl](https://github.com/unfoldtoolbox/unfold.jl).)
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
[#444]: https://github.com/JuliaStats/MixedModels.jl/issues/444
[#446]: https://github.com/JuliaStats/MixedModels.jl/issues/446
[#447]: https://github.com/JuliaStats/MixedModels.jl/issues/447
[#449]: https://github.com/JuliaStats/MixedModels.jl/issues/449
[#453]: https://github.com/JuliaStats/MixedModels.jl/issues/453
[#474]: https://github.com/JuliaStats/MixedModels.jl/issues/474
[#480]: https://github.com/JuliaStats/MixedModels.jl/issues/480
[#482]: https://github.com/JuliaStats/MixedModels.jl/issues/482
[#484]: https://github.com/JuliaStats/MixedModels.jl/issues/484
[#492]: https://github.com/JuliaStats/MixedModels.jl/issues/492
[#493]: https://github.com/JuliaStats/MixedModels.jl/issues/493
[#495]: https://github.com/JuliaStats/MixedModels.jl/issues/495
[#501]: https://github.com/JuliaStats/MixedModels.jl/issues/501
[#507]: https://github.com/JuliaStats/MixedModels.jl/issues/507
[#508]: https://github.com/JuliaStats/MixedModels.jl/issues/508
[#510]: https://github.com/JuliaStats/MixedModels.jl/issues/510
[#518]: https://github.com/JuliaStats/MixedModels.jl/issues/518
[#523]: https://github.com/JuliaStats/MixedModels.jl/issues/523
