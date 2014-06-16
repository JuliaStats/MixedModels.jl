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
<http://www.R-project.org>`__.  Currently the linear mixed models
(LMMs) are implemented.

-------
Example
-------

The :func:`lmer()` function creates a linear mixed model
representation that inherits from :class:`LinearMixedModel`.  The
:class:`LMMGeneral` type can represent any LMM expressed in the
formula language.  Other types are used for better performance in
special cases::

    julia> using MixedModels, RDatasets

    julia> ds = dataset("lme4","Dyestuff");

    julia> dump(ds)
    DataFrame  30 observations of 2 variables
      Batch: PooledDataArray{ASCIIString,Uint8,1}(30) ASCIIString["A","A","A","A"]
      Yield: DataArray{Int32,1}(30) Int32[1545,1440,1440,1520]

    julia> m = lmm(Yield ~ 1|Batch, ds);

    julia> typeof(m)
    LinearMixedModel{PLSOne} (constructor with 2 methods)

    julia> fit(m, true);
    f_1: 327.76702, [1.0]
    f_2: 328.63496, [0.428326]
    f_3: 327.33773, [0.787132]
    f_4: 328.27031, [0.472809]
    f_5: 327.33282, [0.727955]
    f_6: 327.32706, [0.752783]
    f_7: 327.32706, [0.752599]
    f_8: 327.32706, [0.752355]
    f_9: 327.32706, [0.752575]
    f_10: 327.32706, [0.75258]
    FTOL_REACHED

    julia> m
    Linear mixed model fit by maximum likelihood
     logLik: -163.663530, deviance: 327.327060

     Variance components:
		    Variance    Std.Dev.
     Batch        1388.342960   37.260474
     Residual     2451.247052   49.510070
     Number of obs: 30; levels of grouping factors: 6

      Fixed-effects parameters:
	 Estimate Std.Error z value
    [1]    1527.5   17.6946 86.3258

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

.. function:: fit(m, verbose=false) -> m

   Fit the parameters of the model by maximum likelihood or by the REML criterion.

.. function:: reml!(m, v=true]) -> m

   Set the REML flag in ``m`` to ``v``.

.. function:: plssolve!(m.s, u, β) -> m

   Update the spherical random-effects values and the fixed-effects by
   solving the penalized least squares (PLS) problem.  On input the
   arguments ``u`` should contain ``Z'y`` and ``β`` should contain ``X'y``

.. function:: θ!(m, th) -> m

   Set a new value of the variance-component parameter and update the
   sparse Cholesky factor.

----------
Extractors
----------

These extractors are defined for ``m`` of type
:type:`LMMGeneral`.

.. function:: cholfact(m,RX=true) -> Cholesky{Float64} or CholmodFactor{Float64}

   The Cholesky factor, ``RX``, of the downdated X'X or the sparse
   Cholesky factor, ``L``, of the random-effects model matrix in the U
   scale.  These are returned as references and should not be modified.

.. function:: coef(m) -> Vector{Float64}

   A synonym for :func:`fixef`

.. function:: coeftable(m) -> DataFrame

   A dataframe with the current fixed-effects parameter vector, the
   standard errors and their ratio.

.. function:: cor(m) -> Vector{Matrix{Float64}}

   Vector of correlation matrices for the random effects

.. function:: deviance(m) -> Float64

   Value of the deviance (returns ``NaN`` if :func:`isfit` is ``false`` or
   :func:`isreml` is ``true``).

.. function:: fixef(m) -> Vector{Float64}

   Fixed-effects parameter vector

.. function:: grplevels(m) -> Vector{Int}

   Vector of number of levels in random-effect terms

.. function:: lower(m) -> Vector{Float64}

   Vector of lower bounds on the variance-component parameters

.. function:: objective(m) -> Float64

   Value of the profiled deviance or REML criterion at current parameter values

.. function:: pwrss(m) -> Float64

   The penalized, weighted residual sum of squares.

.. function:: ranef(m, uscale=false) -> Vector{Matrix{Float64}}

   Vector of matrices of random effects on the original scale or on the U scale

.. function:: scale(m, sqr=false) -> Float64

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
:type:`Bool`) are defined for `m` of type :type:`LMMGeneral`

.. function:: isfit(m)

   Has the model been fit?

.. function:: isreml(m)

   Is the model to be fit by REML?

.. function:: isscalar(m)

   Are all the random-effects terms scalar?

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
