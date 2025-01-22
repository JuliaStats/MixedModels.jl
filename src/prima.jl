
"""
    prfit!(m::LinearMixedModel)

Fit a mixed model using the [PRIMA](https://github.com/libprima/PRIMA.jl) implementation
of the BOBYQA optimizer.

!!! warning "Experimental feature"
    This function is an experimental feature that will go away in the future.
    Do **not** rely on it, unless you are willing to pin the precise MixedModels.jl
    version. The purpose of the function is to provide the MixedModels developers
    a chance to explore the performance of the PRIMA implementation without the large
    and potentially breaking changes it would take to fully replace the current NLopt
    backend with a PRIMA backend or a backend supporting a range of optimizers.

!!! note "OptSummary"
    As part of this experimental foray, the structure of [`OptSummary`](@ref) is
    not changed. This means that some fields of `OptSummary` are inaccurate when
    examining a model fit with PRIMA. The following fields are unused with PRIMA
    fits:

    - all tolerances (`ftol_rel`, `ftol_abs`, `xtol_rel`, `xtol_abs`)
    - optimization timeouts (`maxfeval`, `maxtime`)
    - `initial_step`

    The following fields have a different meaning when used with PRIMA:

    -  `returnvalue` is populated with a symbol representing the PRIMA
        return value, which PRIMA represents as an enum.
    - `optimizer` is populated with a dummy value, indicating that a model was
       fit with PRIMA. If you wish to refit the model with the NLOpt backend,
       you will need to update the field with an appropriate NLOpt optimizer,
       e.g. `:LN_BOBYQA`

!!! note "Package extension"
    In order to reduce the dependency burden, all methods of this function are
    implemented in a package extension and are only defined when PRIMA.jl is loaded
    by the user.
"""
function prfit! end
