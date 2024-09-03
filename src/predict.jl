"""
    StatsAPI.predict(m::LinearMixedModel, newdata;
                    new_re_levels=:missing)
    StatsAPI.predict(m::GeneralizedLinearMixedModel, newdata;
                    new_re_levels=:missing, type=:response)

Predict response for new data.

!!! note
    Currently, no in-place methods are provided because these methods
    internally construct a new model and therefore allocate not just a
    response vector but also many other matrices.

!!! warning
    `newdata` should contain a column for the response (dependent variable)
    initialized to some numerical value (not `missing`), because this is
    used to construct the new model used in computing the predictions.
    `missing` is not valid because `missing` data are dropped before
    constructing the model matrices.

!!! warning
    These methods construct an entire MixedModel behind the scenes and
    as such may use a large amount of memory when `newdata` is large.

!!! warning
    Rank-deficiency can lead to surprising but consistent behavior.
    For example, if there are two perfectly collinear predictors `A`
    and `B` (e.g. constant multiples of each other), then it is possible
    that `A` will be pivoted out in the fitted model and thus the
    associated coefficient is set to zero. If predictions are then
    generated on new data where `B` has been set to zero but `A` has
    not, then there will no contribution from neither `A` nor `B`
    in the resulting predictions.

The keyword argument `new_re_levels` specifies how previously unobserved
values of the grouping variable are handled. Possible values are:

- `:population`: return population values for the relevant grouping variable.
   In other words, treat the associated random effect as 0.
   If all grouping variables have new levels, then this is equivalent to
   just the fixed effects.
- `:missing`: return `missing`.
- `:error`: error on this condition. The error type is an implementation detail:
   you should not rely on a particular type of error being thrown.

If you want simulated values for unobserved levels of the grouping variable,
consider the [`simulate!`](@ref) and `simulate` methods.

Predictions based purely on the fixed effects can be obtained by
specifying previously unobserved levels of the random effects and setting
`new_re_levels=:population`. Similarly, the contribution of any
grouping variable can be excluded by specifying previously unobserved levels,
while including previously observed levels of the other grouping variables.
In the future, it may be possible to specify a subset of the grouping variables
or overall random-effects structure to use, but not at this time.

!!! note
    `new_re_levels` impacts only the behavior for previously unobserved random
    effects levels, i.e. new RE levels. For previously observed random effects
    levels, predictions take both the fixed and random effects into account.

For `GeneralizedLinearMixedModel`, the `type` parameter specifies
whether the predictions should be returned on the scale of linear predictor
(`:linpred`) or on the response scale (`:response`). If you don't know the
difference between these terms, then you probably want `type=:response`.

Regression weights are not yet supported in prediction.
Similarly, offsets are also not supported for `GeneralizedLinearMixedModel`.
"""
function StatsAPI.predict(
    m::LinearMixedModel, newdata::Tables.ColumnTable; new_re_levels=:missing
)
    return _predict(m, newdata, coef(m)[pivot(m)]; new_re_levels)
end

function StatsAPI.predict(
    m::GeneralizedLinearMixedModel,
    newdata::Tables.ColumnTable;
    new_re_levels=:population,
    type=:response,
)
    type in (:linpred, :response) || throw(ArgumentError("Invalid value for type: $(type)"))
    # want pivoted but not truncated
    y = _predict(m.LMM, newdata, coef(m)[pivot(m)]; new_re_levels)

    return type == :linpred ? y : broadcast!(Base.Fix1(linkinv, Link(m)), y, y)
end

# β is separated out here because m.β != m.LMM.β depending on how β is estimated for GLMM
# also β should already be pivoted but NOT truncated in the rank deficient case
function _predict(m::MixedModel{T}, newdata, β; new_re_levels) where {T}
    new_re_levels in (:population, :missing, :error) ||
        throw(ArgumentError("Invalid value for new_re_levels: $(new_re_levels)"))

    # if we ever support simulation, here some old bits from the docstrings
    # `new_re_levels=:simulate` is also not yet available for `GeneralizedLinearMixedModel`.
    # , `:simulate` (simulate new values).
    # For `:simulate`, the values are determined by solving for their values by
    # using the existing model's estimates for the new data. (These are in general
    # *not* the same values as the estimates computed on the new data.)

    # the easiest thing here is to just assemble a new model and
    # pass that to the other predict methods....
    # this can probably be made much more efficient

    # note that the contrasts don't matter for prediction purposes
    # (at least for the response)

    # add a response column
    # we get type stability via constant propagation on `new_re_levels`
    y, mnew = let ytemp = ones(T, length(first(newdata)))
        f, contr = _abstractify_grouping(m.formula)
        respvars = StatsModels.termvars(f.lhs)
        if !issubset(respvars, Tables.columnnames(newdata)) ||
            any(any(ismissing, Tables.getcolumn(newdata, col)) for col in respvars)
            throw(
                ArgumentError(
                    "Response column must be initialized to a non-missing numeric value."
                ),
            )
        end
        lmm = LinearMixedModel(f, newdata; contrasts=contr)
        ytemp =
            new_re_levels == :missing ? convert(Vector{Union{T,Missing}}, ytemp) : ytemp

        ytemp, lmm
    end

    pivotmatch = pivot(mnew)[pivot(m)]
    grps = fnames(m)
    mul!(y, view(mnew.X, :, pivotmatch), β)
    # mnew.reterms for the correct Z matrices
    # ranef(m) for the BLUPs from the original fit

    # because the reterms are sorted during model construction by
    # number of levels and that number may not be the same for the
    # new data, we need to permute the reterms from both models to be
    # in the same order
    newreperm = sortperm(mnew.reterms; by=x -> string(x.trm))
    oldreperm = sortperm(m.reterms; by=x -> string(x.trm))
    newre = view(mnew.reterms, newreperm)
    oldre = view(m.reterms, oldreperm)

    if new_re_levels == :error
        for (grp, known_levels, data_levels) in
            zip(grps, levels.(m.reterms), levels.(mnew.reterms))
            if sort!(known_levels) != sort!(data_levels)
                throw(ArgumentError("New level encountered in $grp"))
            end
        end

        # we don't have to worry about the BLUP ordering within a given
        # grouping variable because we are in the :error branch
        blups = ranef(m)[oldreperm]
    elseif new_re_levels == :population
        blups = [
            Matrix{T}(undef, size(t.z, 1), nlevs(t)) for t in view(mnew.reterms, newreperm)
        ]
        blupsold = ranef(m)[oldreperm]

        for (idx, B) in enumerate(blups)
            oldlevels = levels(oldre[idx])
            for (lidx, ll) in enumerate(levels(newre[idx]))
                oldloc = findfirst(isequal(ll), oldlevels)
                if oldloc === nothing
                    # setting a BLUP to zero gives you the population value
                    B[:, lidx] .= zero(T)
                else
                    B[:, lidx] .= @view blupsold[idx][:, oldloc]
                end
            end
        end
    elseif new_re_levels == :missing
        blups = [
            Matrix{Union{T,Missing}}(undef, size(t.z, 1), nlevs(t)) for
            t in view(mnew.reterms, newreperm)
        ]
        blupsold = ranef(m)[oldreperm]
        for (idx, B) in enumerate(blups)
            oldlevels = levels(oldre[idx])
            for (lidx, ll) in enumerate(levels(newre[idx]))
                oldloc = findfirst(isequal(ll), oldlevels)
                if oldloc === nothing
                    # missing is poisonous so propagates
                    B[:, lidx] .= missing
                else
                    B[:, lidx] .= @view blupsold[idx][:, oldloc]
                end
            end
        end
        # elseif new_re_levels == :simulate
        #     @show m.θ
        #     updateL!(setθ!(mnew, m.θ))
        #     blups = ranef(mnew)[newreperm]
        #     blupsold = ranef(m)[oldreperm]
        #     for (idx, B) in enumerate(blups)
        #         oldlevels = levels(oldre[idx])
        #         for (lidx, ll) in enumerate(levels(newre[idx]))
        #             oldloc = findfirst(isequal(ll), oldlevels)
        #             if oldloc === nothing
        #                 # keep the new value
        #             else
        #                 B[:, lidx] = @view blupsold[idx][:, oldloc]
        #             end
        #         end
        #     end
    else
        throw(ErrorException("Impossible branch reached. Please report an issue on GitHub"))
    end

    for (rt, bb) in zip(newre, blups)
        mul!(y, rt, bb, one(T), one(T))
    end

    return y
end

# yup, I got lazy on this one -- let the dispatched method handle kwarg checking
function StatsAPI.predict(m::MixedModel, newdata; kwargs...)
    return predict(m, columntable(newdata); kwargs...)
end

## should we add in code for prediction intervals?
# we don't directly implement (Wald) confidence intervals, so directly
# supporting (Wald) prediction intervals seems a step too far right now
