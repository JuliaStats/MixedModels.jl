"""
    StatsBase.predict!(y::AbstractVector{T}, m::LinearMixedModel{T},
                    newX::AbstractMatrix{T}=m.X; use_re=true)
    StatsBase.predict(m::LinearMixedModel{T},
                    newX::AbstractMatrix{T}=m.X; use_re=true)
    StatsBase.predict!(y::AbstractVector{T}, m::GeneralizedLinearMixedModel{T},
                    newX::AbstractMatrix{T}=m.X; use_re=true, type=:response)
    StatsBase.predict(m::GeneralizedLinearMixedModel{T},
                    newX::AbstractMatrix{T}=m.X; use_re=true, type=:response)

Predict response for new values of the fixed-effects matrix X.

The mutating methods overwrite `y` in place, while the non mutating methods
allocate a new `y`. Predictions based purely on the fixed effects can be
obtained with `use_re=false`. In the future, it may be possible to specify
a subset of the grouping variables to use, but not at this time.

For `GeneralizedLinearMixedModel`, the `type` parameter specifies
whether the predictions should be returned on the scale of linear predictor
(`:linpred`) or on the response scale (`:response`). If you don't know the
difference between these terms, then you probably want `type=:response`.

!!! warning
    Models are assumed to be full rank.

!!! note
    The `predict` and `predict!` methods with `newX` as a fixed effects matrix
    differ from the methods with `newdata` in tabular format. The matrix
    methods are more efficient than the tabular methods. The tabular methods
    can accomodate new values of the grouping variable(s), but the matrix methods
    cannot.
"""
function StatsBase.predict!(y::AbstractVector{<:Union{T, Missing}}, m::LinearMixedModel{T}, newX::AbstractMatrix{T}=m.X;
                           use_re=true) where T
    # this is called `use_re` in case we later decide to support prediction
    # with only a subset of the RE

    mul!(y, newX, m.β)

    if use_re
        for (rt, bb) in zip(m.reterms, ranef(m))
            mul!(y, rt, bb, one(T), one(T))
        end
    end

    y
end

function StatsBase.predict!(y::AbstractVector{<:Union{T, Missing}}, m::GeneralizedLinearMixedModel{T}, newX::AbstractMatrix{T}=m.X;
                            use_re=true, type=:response) where T
    # this is called `use_re` in case we later decide to support prediction
    # with only a subset of the RE

    type in (:linpred, :response) || throw(ArgumentError("Invalid value for `type`: $(type)"))

    mul!(y, newX, m.β)

    if use_re
        for (rt, bb) in zip(m.reterms, ranef(m))
            mul!(y, rt, bb, one(T), one(T))
        end
    end

    type == :linpred && return y

    @inbounds for (idx, val) in enumerate(y)
        y[idx] = linkinv(Link(m.resp), val)
    end

    y
end

function StatsBase.predict(m::LinearMixedModel{T}, newX::AbstractMatrix{T}=m.X;
                          use_re=true) where T
    # this is called `use_re` in case we later decide to support prediction
    # with only a subset of the RE

    y = zeros(T, nobs(m))
    predict!(y, m, newX; use_re=use_re)
end

function StatsBase.predict(m::GeneralizedLinearMixedModel{T}, newX::AbstractMatrix{T}=m.X;
                           use_re=true, type=:response) where T
    # this is called `use_re` in case we later decide to support prediction
    # with only a subset of the RE

    type in (:linpred, :response) || throw(ArgumentError("Invalid value for type: $(type)"))

    if use_re && newX === m.X
        # should we add a kwarg to fitted to allow returning eta?
        return type == :response ? fitted(m) : m.resp.eta
    end
    y = zeros(T, nobs(m))
    predict!(y, m, newX; use_re=use_re, type=type)
end

"""
    StatsBase.predict(m::LinearMixedModel{T}, newdata;
                    new_re_levels=:missing)
    StatsBase.predict(m::GeneralizedLinearMixedModel{T}, newdata;
                    new_re_levels=:missing, type=:response)

Predict response for new values of the fixed-effects matrix X.

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

!!! note
    The `predict` and `predict!` methods with `newX` as a fixed effects matrix
    differ from the methods with `newdata` in tabular format. The matrix
    methods are more efficient than the tabular methods. The tabular methods
    can accomodate new values of the grouping variable(s), but the matrix methods
    cannot.
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
    y = let ytemp = ones(T, length(first(newdata)))
        new_re_levels == :missing ? convert(Vector{Union{T, Missing}}, ytemp) : ytemp
    end

    newdata = merge(newdata, NamedTuple{(m.formula.lhs.sym,)}((y,)))

    f, contr = _abstractify_grouping(m.formula)
    mnew = LinearMixedModel(f, newdata; contrasts=contr)

    grps = getproperty.(m.reterms, :trm)
    y = predict!(y, m, mnew.X; use_re=false)
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

    # # the trick we use to simulate in LMM will give probably give garbage for GLMM
    # # because the weights aren't passed
    # new_re_levels == :simulate &&
    #     throw(ArgumentError("Simulation of new RE levels not available for GeneralizedLinearMixedModel"))
    y = predict(m.LMM, newdata; new_re_levels=new_re_levels)

    type == :linpred && return y

    y .= linkinv.(Link(m.resp), y)

    y
end

# yup, I got lazy on this one -- let the dispatched method handle kwarg checking
StatsBase.predict(m::MixedModel, newdata; kwargs...) =
    predict(m, columntable(newdata); kwargs...)

## should we add in code for prediction intervals?
# we don't directly implement (Wald) confidence intervals, so directly
# supporting (Wald) prediction intervals seems a step too far right now
