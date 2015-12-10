MixedModels.jl --- Mixed-effects (statistical) models
=====================================================

.. toctree::
   :maxdepth: 2

.. highlight:: julia

.. .. module:: MixedModels.jl
   :synopsis: Fit and analyze mixed-effects models

MixedEffects.jl provides functions and methods to fit `mixed-effects
models <http://en.wikipedia.org/wiki/Mixed_model>`__ using a
specification similar to that of the `lme4
<https://github.com/lme4/lme4>`__ package for `R
<http://www.R-project.org>`__.  Currently only linear mixed models
(LMMs) are implemented.

-------
Example
-------

The :func:`lmm()` function creates a linear mixed model
representation that inherits from :class:`LinearMixedModel`.

    julia> using DataFrames, MixedModels, RCall

    julia> @rimport lme4
    WARNING: RCall.jl Loading required package: Matrix

    julia> ds = rcopy(lme4.Dyestuff);

    julia> dump(ds)
    DataFrames.DataFrame  30 observations of 2 variables
      Batch: DataArrays.PooledDataArray{ASCIIString,UInt8,1}(30) ASCIIString["A","A","A","A"]
      Yield: DataArrays.DataArray{Float64,1}(30) [1545.0,1440.0,1440.0,1520.0]

    julia> m = lmm(Yield ~ 1|Batch, ds);

    julia> typeof(m)
    MixedModels.LinearMixedModel{Float64}

    julia> fit!(m,true);
    f_1: 327.76702, [1.0]
    f_2: 331.03619, [1.75]
    f_3: 330.64583, [0.25]
    f_4: 327.69511, [0.97619]
    f_5: 327.56631, [0.928569]
    f_6: 327.3826, [0.833327]
    f_7: 327.35315, [0.807188]
    f_8: 327.34663, [0.799688]
    f_9: 327.341, [0.792188]
    f_10: 327.33253, [0.777188]
    f_11: 327.32733, [0.747188]
    f_12: 327.32862, [0.739688]
    f_13: 327.32706, [0.752777]
    f_14: 327.32707, [0.753527]
    f_15: 327.32706, [0.752584]
    f_16: 327.32706, [0.752509]
    f_17: 327.32706, [0.752591]
    f_18: 327.32706, [0.752581]
    FTOL_REACHED

    julia> m
    Linear mixed model fit by maximum likelihood
     logLik: -163.663530, deviance: 327.327060, AIC: 333.327060, BIC: 337.530652

    Variance components:
               Variance  Std.Dev.
     Batch    1388.3332 37.260344
     Residual 2451.2500 49.510100
     Number of obs: 30; levels of grouping factors: 6

      Fixed-effects parameters:
                 Estimate Std.Error z value
    (Intercept)    1527.5   17.6946  86.326

------------
Constructors
------------

.. function:: lmm(f, fr)

   Create the representation for a linear mixed-effects model with
   formula ``f`` (of :type:`Formula`) evaluated in the :type:`DataFrame`
   ``fr``.

-------
Setters
-------

These setters or mutating functions are defined for ``m`` of type
:type:`LinearMixedModel`.  By convention their names end in ``!``.  The
:func:`fit` function is an exception, because the name was
already established in the ``StatsBase`` package.

.. function:: fit!(m, verbose=false) -> m

   Fit the parameters of the model by maximum likelihood.

.. function:: m[:θ] = th -> th

   Set a new value of the variance-component parameter and update the
   blocked Cholesky factor.

----------
Extractors
----------

These extractors are defined for ``m`` of type
:type:`LMMGeneral`.

.. function:: cholfact(m) -> UpperTriangular{Float64,Array{Float64,2}}

   The Cholesky factor, ``RX``, of the downdated X'X.

.. function:: coef(m) -> Vector{Float64}

   A synonym for :func:`fixef`

.. function:: coeftable(m) -> StatsBase.CoefTable

   A CoefTable with the current fixed-effects parameter vector, the
   standard errors and their ratio.

.. function:: cor(m) -> Vector{Matrix{Float64}}

   Vector of correlation matrices for the random effects

.. function:: deviance(m) -> Float64

   Value of the deviance (throws an error if :func:`isfit` is ``false``).

.. function:: fixef(m) -> Vector{Float64}

   Fixed-effects parameter vector

.. function:: lowerbd(m) -> Vector{Float64}

   Vector of lower bounds on the variance-component parameters

.. function:: objective(m) -> Float64

   Value of the profiled deviance at current parameter values

.. function:: pwrss(m) -> Float64

   The penalized, weighted residual sum of squares.

.. function:: ranef(m, uscale=false) -> Vector{Matrix{Float64}}

   Vector of matrices of random effects on the original scale or on the U scale

.. function:: sdest(m, sqr=false) -> Float64

   Estimate, ``s``, of the residual scale parameter or its square.

.. function:: std(m) -> Vector{Float64}

   Estimated standard deviations of random effects.

.. function:: stderr(m) -> Vector{Float64}

   Standard errors of the fixed-effects parameters

.. function:: vcov(m) -> Matrix{Float64}

   Estimated variance-covariance matrix of the fixed-effects parameters

----------
Predicates
----------

The following predicates (functions that return boolean values,
:type:`Bool`) are defined for `m` of type :type:`LinearMixedModel`

.. function:: isfit(m)

   Has the model been fit?

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
