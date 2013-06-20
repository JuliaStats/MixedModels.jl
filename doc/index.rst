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

    julia> ds = data("lme4", "Dyestuff");

    julia> dump(ds)
    DataFrame  30 observations of 2 variables
      Batch: PooledDataArray{ASCIIString,Uint8,1}(30) ["A","A","A","A"]
      Yield: DataArray{Float64,1}(30) [1545.0,1440.0,1440.0,1520.0]

    julia> m = lmer(:(Yield ~ 1|Batch), ds);

    julia> typeof(m)
    LMMGeneral{Int32}

    julia> fit(m, true);
    f_1: 327.7670216246145, [1.0]
    f_2: 331.0361932224437, [1.75]
    f_3: 330.6458314144857, [0.25]
    f_4: 327.69511270610866, [0.9761896354668361]
    f_5: 327.56630914532184, [0.9285689064005083]
    f_6: 327.3825965130752, [0.8333274482678525]
    f_7: 327.3531545408492, [0.8071883308459398]
    f_8: 327.34662982410276, [0.7996883308459398]
    f_9: 327.34100192001785, [0.7921883308459399]
    f_10: 327.33252535370985, [0.7771883308459397]
    f_11: 327.32733056112147, [0.7471883308459397]
    f_12: 327.3286190977697, [0.7396883308459398]
    f_13: 327.32706023603697, [0.7527765100471926]
    f_14: 327.3270681545395, [0.7535265100471926]
    f_15: 327.3270598812218, [0.7525837539477753]
    FTOL_REACHED

---------
Functions
---------

.. function:: lmer(f, fr)

   Create the representation for a linear mixed-effects model with
   formula ``f`` evaluated in the :class:`DataFrame` ``fr``.  The
   primary method is for ``f`` of type :class:`Formula` but more
   commonly ``f`` is an expression (:class:`Expr`) as in the example
   above.

-------
Setters
-------

These setters or mutating functions are defined for ``m`` of type
:class:`LMMGeneral`.  By convention their names end in ``!``.  The
:func:`fit` function is an exception, because the name was
already established in the ``Distributions`` package.

.. function:: fit(m[, verbose]) -> m

   Fit the parameters of the model by maximum likelihood or by the REML
   criterion.

.. function:: reml!(m[, v]) -> m

   Set or unset (if ``v`` is ``false``) fitting according to the REML
   criterion.

.. function:: solve!(m[, ubeta]) -> m

   Update the random-effects values (and the fixed-effects, when
   ``ubeta`` is ``true``) by solving the penalized least squares (PLS)
   problem.

.. function:: theta!(m, th) -> m

   Set a new value of the variance-component parameter and update the
   sparse Cholesky factor.

----------
Extractors
----------

These extractors are defined for ``m`` of type
:class:`LMMGeneral`.

.. function:: cor(m)

   Vector of correlation matrices for the random effects

.. function:: deviance(m)

   Value of the deviance (returns ``NaN`` if :func:`isfit` is ``false`` or
   :func:`reml` is ``true``).

.. function:: grplevels(m)

   Vector of number of levels in random-effect terms

.. function:: fixef(m)

   Fixed-effects parameter vector

.. function:: linpred(m[, minusy])

   The linear predictor vector or the negative residual vector

.. function:: lower(m)

   Vector of lower bounds on the variance-component parameters

.. function:: pwrss(m)

   The penalized, weighted residual sum of squares.

.. function:: RX(m)

   The Cholesky factor of the downdated X'X (can be a reference)

.. function:: ranef(m[, uscale])

   Vector of matrices of random effects on the original scale or the U
   scale

.. function:: scale(m[, sqr])

   Estimate, ``s``, of the residual scale parameter or its square.

.. function:: std(m)

   Estimated standard deviations of random effects, in the form of a
   vector of vectors.

.. function:: stderr(m)

   Standard errors of the fixed-effects parameters

.. function:: theta(m)

   Vector of variance-component parameters

.. function:: vcov(m)

   Estimated variance-covariance matrix of the fixed-effects
   parameters

----------
Predicates
----------

The following predicates (functions that return boolean values,
:class:`Bool`) are defined for `m` of type :class:`LMMGeneral`

.. function:: isfit(m)

   Has the model been fit?

.. function:: isscalar(m)

   Are all the random-effects terms scalar?

.. function:: reml(m)

   Is the model to be fit by REML?

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
