"""
    StatsBase.predict(m::LinearMixedModel{T}, newdata;
                    new_re_levels=:missing)
    StatsBase.predict(m::GeneralizedLinearMixedModel{T}, newdata;
                    new_re_levels=:missing, type=:response)

Predict response for new data.

!!! note
    Currently, no in-place methods are provided because these methods
    internally construct a new model and therefore allocate not just a
    response vector but also many other matrices.

!!! warning
    Models are assumed to be full rank.

!!! warning
    These methods construct an entire MixedModel behind the scenes and
    as such may use a large amount of memory when `newdata` is large.

The keyword argument `new_re_levels` specifies how previously unobserved
values of the grouping variable are handled. Possible values are
`:population` (return population values, i.e. fixed-effects only),
`:missing` (return `missing`), `:error` (error on this condition;
the error type is an implementation detail). If you want simulated values
for unobserved levels of the grouping variable, consider the
[`simulate!`](@ref) and `simulate` methods.

Predictions based purely on the fixed effects can be obtained by
specifying previously unobserved levels of the random effects and setting
`new_re_levels=:population`. In the future, it may be possible to specify
a subset of the grouping variables or overall random-effects structure to use,
but not at this time.

For `GeneralizedLinearMixedModel`, the `type` parameter specifies
whether the predictions should be returned on the scale of linear predictor
(`:linpred`) or on the response scale (`:response`). If you don't know the
difference between these terms, then you probably want `type=:response`.

Regression weights are not yet supported in prediction.
Similarly, offsets are also not supported for `GeneralizedLinearMixedModel`.
"""
function StatsBase.predict(m::LinearMixedModel{T}, newdata::Tables.ColumnTable;
                           new_re_levels=:population) where T

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
    # we get type stability via constant propogation on `new_re_levels`
    y, mnew = let ytemp = ones(T, length(first(newdata)))
        f, contr = _abstractify_grouping(m.formula)
        sch = schema(f, newdata, contr)
        form = apply_schema(f, sch, LinearMixedModel)
        mnewXs = modelcols(form.rhs, newdata)
        lmm = LinearMixedModel(ytemp, mnewXs, form)

        ytemp = new_re_levels == :missing ? convert(Vector{Union{T, Missing}}, ytemp) : ytemp

        ytemp, lmm
    end

    grps = fnames(m)
    mul!(y, mnew.X, m.β)
    # mnew.reterms for the correct Z matrices
    # ranef(m) for the BLUPs from the original fit

    # because the reterms are sorted during model construction by
    # number of levels and that number may not be the same for the
    # new data, we need to permute the reterms from both models to be
    # in the same order
    newreperm = sort(1:length(mnew.reterms), by=x -> string(mnew.reterms[x].trm))
    oldreperm = sort(1:length(m.reterms), by=x -> string(m.reterms[x].trm))
    newre = mnew.reterms[newreperm]
    oldre = m.reterms[oldreperm]

    if new_re_levels == :error
        for (grp, known_levels, data_levels) in zip(grps,
                                                    levels.(m.reterms),
                                                    levels.(mnew.reterms))
            if sort!(known_levels) != sort!(data_levels)
                throw(ArgumentError("New level enountered in $grp"))
            end
        end

        # we don't have to worry about the BLUP ordering within a given
        # grouping variable because we are in the :error branch
        blups = ranef(m)[oldreperm]
    elseif new_re_levels == :population
        blups = ranef(mnew)[newreperm]
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
        # we can't quite use ranef! because we need
        # Union{T, Missing} and not just T
        blups = Vector{Matrix{Union{T,Missing}}}(undef, length(m.reterms))
        copyto!(blups, ranef(mnew)[newreperm])
        blupsold = ranef(m)[oldreperm]
        for (idx, B) in enumerate(blups)
            oldlevels = levels(oldre[idx])
            for (lidx, ll) in enumerate(levels(newre[idx]))
                oldloc = findfirst(isequal(ll), oldlevels)
                if oldloc === nothing
                    # missing is poisonous so propogates
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

    y
end

function StatsBase.predict(m::GeneralizedLinearMixedModel{T}, newdata::Tables.ColumnTable;
                           new_re_levels=:population, type=:response) where T
    type in (:linpred, :response) || throw(ArgumentError("Invalid value for type: $(type)"))

    y = predict(m.LMM, newdata; new_re_levels)

    type == :linpred ? y : broadcast!(Base.Fix1(linkinv, Link(m)), y, y)
end

# yup, I got lazy on this one -- let the dispatched method handle kwarg checking
StatsBase.predict(m::MixedModel, newdata; kwargs...) =
    predict(m, columntable(newdata); kwargs...)

## should we add in code for prediction intervals?
# we don't directly implement (Wald) confidence intervals, so directly
# supporting (Wald) prediction intervals seems a step too far right now
