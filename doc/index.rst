MixedModels.jl --- Mixed-effects (statistical) models
=====================================================

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
special cases.::

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

----------
Extractors for types :class:`LMMGeneral`
----------

.. function:: theta(m)

   Vector of variance-component parameters

.. function:: linpred(m[, minusy])

   The linear predictor vector or the negative residual vector

.. function:: pwrss(m)

   The penalized, weighted residual sum of squares.

.. function:: reml(m)

   Is the model to be fit by REML?

.. function:: lower(m)

   Vector of lower bounds on the variance-component parameters

.. function:: RX(m)

   The Cholesky factor of the downdated X'X (can be a reference)

.. function:: ranef(m[, uscale])

   Vector of matrices of random effects on the original scale or the U
   scale

.. function:: grplevels(m)

   Vector of number of levels in random-effect terms

.. function:: fixef(m)

   Fixed-effects parameter vector

.. function:: isfit(m)

   Has the model been fit?

.. function:: scale(m[, sqr])

   Estimate, ``s``, of the residual scale parameter or its square.

.. function:: std(m)

   :class:`Vector{Vector{Float64}}` estimated standard deviations of
	  random effects

.. function:: cor(m)

   Vector of correlation matrices for the random effects

.. function:: deviance(m)

   Value of the deviance.

.. function:: vcov(m)

   Estimated variance-covariance matrix of the fixed-effects
   parameters

.. function:: stderr(m)

   Standard errors of the fixed-effects parameters

