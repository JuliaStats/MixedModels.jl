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

The modifying methods overwrite `y` in place, while the non modifying methods
allocate a new `y`. Predictions based purely on the fixed effects can be
obtained with `use_re=false`. In the future, it may be possible to specify
a subset of the grouping variables to use, but not at this time.

For [`GeneralizedLinearMixedModel`](@ref), the `type` parameter specifies
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
function StatsBase.predict!(y::AbstractVector{T}, m::LinearMixedModel{T}, newX::AbstractMatrix{T}=m.X;
                           use_re=true) where T
    # this is called `use_re` in case we later decide to support prediction
    # with only a subset of the RE

    mul!(y, newX, m.β)

    use_re || return y

    # because we're using ranef(), this actually
    # adds the *scaled* random effects
    for (rt, bb) in zip(m.reterms, ranef(m))
        unscaledre!(y, rt, bb)
    end

    y
end

function StatsBase.predict!(y::AbstractVector{T}, m::GeneralizedLinearMixedModel{T}, newX::AbstractMatrix{T}=m.X;
                            use_re=true, type=:response) where T
    # this is called `use_re` in case we later decide to support prediction
    # with only a subset of the RE

    type in (:linpred, :response) || throw(ArgumentError("Invalid value for `type`: $(type)"))

    mul!(y, newX, m.β)

    if use_re
        # because we're using ranef(), this actually
        # adds the *scaled* random effects
        for (rt, bb) in zip(m.reterms, ranef(m))
            unscaledre!(y, rt, bb)
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
        type == :response && return fitted(m)
        return m.resp.eta
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
the error type is an implementation detail), `:simulate` (simulate new values).
For `:simulate`, the values are determined by solving for their values by
using the existing model's estimates for the new data. (These are in general
*not* the same values as the estimates computed on the new data.)

Predictions based purely on the fixed effects can be obtained by
specifying previously unobserved levels of the random effects and setting
`new_re_levels=:population`. In the future, it may be possible to specify
a subset of the grouping variables or overall random-effects structure to use,
but not at this time.

For [`GeneralizedLinearMixedModel`](@ref), the `type` parameter specifies
whether the predictions should be returned on the scale of linear predictor
(`:linpred`) or on the response scale (`:response`). If you don't know the
difference between these terms, then you probably want `type=:response`.

Regression weights are not yet supported in prediction. As a consequence of this,
`new_re_levels=:simulate` is also not yet available for `GeneralizedLinearMixedModel`.
Similarly, offset are also not support for `GeneralizedLinearMixedModel`.

!!! note
    The `predict` and `predict!` methods with `newX` as a fixed effects matrix
    differ from the methods with `newdata` in tabular format. The matrix
    methods are more efficient than the tabular methods. The tabular methods
    can accomodate new values of the grouping variable(s), but the matrix methods
    cannot.
"""
function StatsBase.predict(m::LinearMixedModel{T}, newdata::Tables.ColumnTable;
                           new_re_levels=:population) where T

    new_re_levels in (:population, :missing, :)
    # the easiest thing here is to just assemble a new model and
    # pass that to the other predict methods....
    # this can probably be made much more efficient

    # note that the contrasts don't matter for prediction purposes
    # (at least for the response)

    # add a response column
    # we use the Union here so that we have type stability
    y = ones(Union{T, Missing}, length(first(newdata)))
    newdata = merge(newdata, NamedTuple{(m.formula.lhs.sym,)}((y,)))

    mnew = LinearMixedModel(m.formula, newdata)

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
                throw(KeyError("New level enountered in $grp"))
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
                    B[lidx] = zero(T)
                else
                    B[lidx] = blupsold[idx][oldloc]
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
                    B[lidx] = missing
                else
                    B[lidx] = blupsold[idx][oldloc]
                end
            end
        end
    elseif new_re_levels == :simulate
        updateL!(setθ!(mnew, m.θ))
        blupsold = ranef(m)[oldreperm]
        for (idx, B) in enumerate(blups)
            oldlevels = levels(oldre[idx])
            for (lidx, ll) in enumerate(levels(newre[idx]))
                oldloc = findfirst(isequal(ll), oldlevels)
                if oldloc === nothing
                    # keep the new value
                else
                    B[lidx] = blupsold[idx][oldloc]
                end
            end
        end
    else
        throw(ErrorException("Impossible branch reached. Please report an issue on GitHub"))
    end

    for (rt, bb) in zip(newre, blups)
        unscaledre!(y, rt, bb)
    end

    y
end

function StatsBase.predict(m::GeneralizedLinearMixedModel{T}, newdata::Tables.ColumnTable;
                           new_re_levels=:population, type=:response) where T
    type in (:linpred, :response) || throw(ArgumentError("Invalid value for type: $(type)"))

    # the trick we use to simulate in LMM will give probably give garbage for GLMM
    # because the weights aren't passed
    new_re_levels == :simulate &&
        throw(ArgumentError("Simulation of new RE levels not available for GeneralizedLinearMixedModel"))
    y = predict(m.LMM, newdata; new_re_levels=new_re_levels)

    type == :linpred && return y

    @inbounds for (idx, val) in enumerate(y)
        y[idx] = linkinv(Link(m.resp), val)
    end

    y
end

# yup, I got lazy on this one -- let the dispatched method handle kwarg checking
StatsBase.predict(m::MixedModel, newdata; kwargs...) =
    predict(m, columntable(newdata); kwargs...)

"""
    simulate!([rng::AbstractRNG,] y::AbstractVector, m::MixedModel{T},
                    newX::AbstractArray{T} = m.X;
                    β = coef(m), σ = m.σ, θ = T[], wts=m.wts)
    simulate!([rng::AbstractRNG,] y::AbstractVector, m::MixedModel{T},
                    newdata;
                    β = coef(m), σ = m.σ, θ = T[], wts=m.wts)
    simulate([rng::AbstractRNG,] m::MixedModel{T},
                    newX::AbstractArray{T} = m.X;
                    β = coef(m), σ = m.σ, θ = T[], wts=m.wts)
    simulate([rng::AbstractRNG,] m::MixedModel{T},
                    newdata;
                    β = coef(m), σ = m.σ, θ = T[], wts=m.wts)

Simulate a new response vector, optionally overwriting a pre-allocated vector.

New data can be optionally provided, either as a fixed-effects model matrix or
in tabular format. Currently, the tabular format is the only way to specify
different observations for the random effects than in the original model.

This simulation includes sampling new values for the random effects. Thus in
contrast to [`predict`](`@ref`), there is no distinction in between "new" and
"old" / previously observed random-effects levels.

Unlike [`predict`](`@ref`), there is `type` parameter for [`GeneralizedLinearMixedModel`](@ref)
because the noise term in the model and simulation is always on the response
scale.

The `wts` argument is currently ignored except for `GeneralizedLinearMixedModel`
models with a `Binomial` distribution.

!!! warning
    Models are assumed to be full rank.

!!! note
    Note that `simulate!` methods with a `y::AbstractVector` as the first argument
    (besides the RNG) and `simulate` methods return the simulated response. This is
    in contrast to `simulate!` methods with a `m::MixedModel` as the first argument,
    which modify the model's response and return the entire modified model.
"""
function simulate!(rng::AbstractRNG,
                   y::AbstractVector,
                   m::LinearMixedModel{T},
                   newX::AbstractArray{T} = m.X;
                   β = coef(m),
                   σ = m.σ,
                   θ = T[],
                   wts = m.sqrtwts .^ 2
               ) where {T}

    length(β) == length(fixef(m)) ||
        length(β) == length(coef(m)) ||
            throw(ArgumentError("You must specify all (non-singular) βs"))

    β = convert(Vector{T},β)
    σ = T(σ)
    θ = convert(Vector{T},θ)
    isempty(θ) || setθ!(m, θ)

    if length(β) ≠ length(coef(m))
        padding = length(coef(m)) - length(β)
        for ii in 1:padding
            push!(β, -0.0)
        end
    end

    # initialize y to standard normal
    randn!(rng, y)

    # add the unscaled random effects
    for trm in m.reterms
        unscaledre!(rng, y, trm)
    end

    # scale by σ and add fixed-effects contribution
    mul!(y, m.X, β, one(T), σ)

    y
end

function simulate!(rng::AbstractRNG,
    y::AbstractVector,
    m::GeneralizedLinearMixedModel{T},
    newX::AbstractMatrix{T} = m.X;
    β = coef(m),
    σ = m.σ,
    θ = T[],
    wts = m.resp.wts) where {T}

    resp = deepcopy(m.resp)
    η = fill!(similar(m.LMM.y), zero(T))
    _simulate!(rng, y, η, resp, m, newX, β, σ, θ, wts)
end


function _simulate!(rng::AbstractRNG,
    y::AbstractVector, # modified
    η::AbstractVector, # modified
    resp::GLM.GlmResp, # modified
    m::GeneralizedLinearMixedModel{T},
    newX::AbstractArray{T},
    β, σ, θ, wts # note that these are not kwargs for the internal method!
) where {T}
    length(β) == length(fixef(m)) ||
        length(β) == length(coef(m)) ||
            throw(ArgumentError("You must specify all (non-singular) βs"))

    dispersion_parameter(m) || ismissing(σ) ||
        throw(ArgumentError("You must not specify a dispersion parameter for model families without a dispersion parameter"))

    β = convert(Vector{T},β)
    if σ !== missing
        σ = T(σ)
    end
    θ = convert(Vector{T},θ)

    d = m.resp.d

    if length(β) ≠ length(coef(m))
        padding = length(coef(m)) - length(β)
        for ii in 1:padding
            push!(β, -0.0)
        end
    end

    fast = (length(m.θ) == length(m.optsum.final))
    setpar! = fast ? setθ! : setβθ!
    params = fast ? θ : vcat(β, θ)
    setpar!(m, params)

    lm = m.LMM

    # assemble the linear predictor

    # add the unscaled random effects
    # note that unit scaling may not be correct for
    # families with a dispersion parameter
    @inbounds for trm in m.reterms
        unscaledre!(rng, η, trm)
    end

    # add fixed-effects contribution
    # note that unit scaling may not be correct for
    # families with a dispersion parameter
    mul!(η, lm.X, β, one(T), one(T))

    # from η to μ
    GLM.updateμ!(resp, η)

    # convert to the distribution / add in noise
    @inbounds for (idx, val) in enumerate(resp.mu)
        n = isempty(m.wt) ? 1 : m.wt[idx]
        y[idx] = _rand(rng, d, val, σ, n)
    end

    y
end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::LinearMixedModel, newdata::Tables.ColumnTable;
                   kwargs...)
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    # (for one thing, this still allocates for the model's response)
    # note that the contrasts don't matter for simulation purposes
    # (at least for the response)
    mnew = LinearMixedModel(m.formula, newdata; REML=m.optsum.REML)
    # this solves for the BLUPs
    updateL!(setθ!(mnew, θ))
    simulate!(rng, y, mnew,
              β=β, σ=σ, θ=θ, wts=wts)
    y
end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::GeneralizedLinearMixedModel, newdata::Tables.ColumnTable;
                   kwargs...)
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    # (for one thing, this still allocates for the model's response)
    # note that the contrasts don't matter for simulation purposes
    # (at least for the response)
    mnew = GeneralizedLinearMixedModel(m.formula, newdata, m.resp.d, m.resp.l)
    # this solves for the BLUPs
    deviance!(setθ!(mnew, θ))
    simulate!(rng, y, mnew,
              β=β, σ=σ, θ=θ, wts=wts)
end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::MixedModel, newdata;
                   kwargs...)

    simulate!(rng, y, m, Tables.columntable(newdata); kwargs...)
end

function simulate!(y::AbstractVector, m::MixedModel, newdata;
                   kwargs...)
    simulate!(Random.GLOBAL_RNG, y, m, Tables.columntable(newdata);
              kwargs...)
end

function simulate(rng::AbstractRNG, m::MixedModel, X::AbstractMatrix = m.X;
                  kwargs...)
    y = zeros(T, nobs(m))
    simulate!(rng, y, m, X; kwargs...)
    y
end

function simulate(m::MixedModel, X::AbstractMatrix = m.X; kwargs...)
    simulate(Random.GLOBAL_RNG, m, X; kwargs...)
end

## should we add in code for prediction intervals?
# we don't directly implement (Wald) confidence intervals, so directly
# supporting (Wald) prediction intervals seems a step too far right now
