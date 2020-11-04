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

    use_re || mul!(y, newX, m.β)

    y .= zero(T)
    # add the unscaled random effects
    for trm in m.reterms
        unscaledre!(y, trm)
    end

    # scale by σ and add fixed-effects contribution
    mul!(y, m.X, β, one(T), σ)
end

function StatsBase.predict!(y::AbstractVector{T}, m::GeneralizedLinearMixedModel{T}, newX::AbstractMatrix{T}=m.X;
                            use_re=true, type=:response) where T
    # this is called `use_re` in case we later decide to support prediction
    # with only a subset of the RE

    type in (:linpred, :response) || throw(ArgumentError("Invalid value for `type`: $(type)"))

    if use_re
        mul!(y, newX, m.β)
    else
        y .= zero(T)
        # add the unscaled random effects
        for trm in m.reterms
            unscaledre!(y, trm)
        end

        # scale by σ and add fixed-effects contribution
        mul!(y, m.X, β, one(T), σ)
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

    use_re && newX === m.X && return fitted(m)
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
                  new_re_levels=:missing,
                  rng=Random.GLOBAL_RNG)
StatsBase.predict(m::GeneralizedLinearMixedModel{T}, newdata;
                  new_re_levels=:missing, type=:response,
                  rng=Random.GLOBAL_RNG)

Predict response for new values of the fixed-effects matrix X.

!!! note
    Currently, no in-place methods are provided because these methods
    internally construct a new model and therefore allocate not just a
    response vector but also many other matrices.

The keyword argument `new_re_levels` specifies how previously unobserved
values of the grouping variable are handled. Possible values are
`:population` (return population values, i.e. fixed-effects only),
`:missing` (return `missing`), `:error` (error on this condition),
`:simulate` (simulate new values). For reproducibility with`:simulate`,
a random-number generator can be specified with `rng`.

Predictions based purely on the fixed effects can be obtained with by
specifying previously unobserved levels of the random effects and setting
`new_re_levels=:population`. In the future, it may be possible to specify
a subset of the grouping variables or overall random-effects structure to use,
but not at this time.

For [`GeneralizedLinearMixedModel`](@ref), the `type` parameter specifies
whether the predictions should be returned on the scale of linear predictor
(`:linpred`) or on the response scale (`:response`). If you don't know the
difference between these terms, then you probably want `type=:response`.

!!! note
    The `predict` and `predict!` methods with `newX` as a fixed effects matrix
    differ from the methods with `newdata` in tabular format. The matrix
    methods are more efficient than the tabular methods. The tabular methods
    can accomodate new values of the grouping variable(s), but the matrix methods
    cannot.

"""
function StatsBase.predict(m::LinearMixedModel, newdata::Tables.ColumnTable;
                           new_re_levels=:population)

end

function StatsBase.predict(m::GeneralizedLinearMixedModel, newdata::Tables.ColumnTable;
                           type=:linpred)

end


StatsBase.predict(m::MixedModel, newdata; kwargs...) =
    predict(m, columntable(newdata); kwargs...)

"""
    simulate!([rng::AbstractRNG,] y::AbstractVector, m::MixedModel{T},
                    newX::AbstractArray{T} = m.X;
                    β = coef(m), σ = m.σ, θ = T[])
    simulate!([rng::AbstractRNG,] y::AbstractVector, m::MixedModel{T},
                    newdata;
                    β = coef(m), σ = m.σ, θ = T[])
    simulate([rng::AbstractRNG,] m::MixedModel{T},
                    newX::AbstractArray{T} = m.X;
                    β = coef(m), σ = m.σ, θ = T[])
    simulate([rng::AbstractRNG,] m::MixedModel{T},
                    newdata;
                    β = coef(m), σ = m.σ, θ = T[])


Simulate a new response vector, optionally overwriting a pre-allocated vector.

New data can be optionally provided, either as a fixed-effects model matrix or
in tabular format. Currently, the tabular format is the only way to specify
different observations for the random effects than in the original model.

This simulation includes sampling new values for the random effects. Thus in
contrast to [`predict`](`@ref`), there is no distinction in between "new" and
"old" / previously observed random-effects levels.


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

function simulate!(rng::AbstractRNG, y::AbstractVector, m::LinearMixedModel, newdata::Tables.ColumnTable;
                   β = coef(m), σ = m.σ, θ = T[])
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    # (for one thing, this still allocates for the model's response)
    # note that the contrasts don't matter for simulation purposes
    # (at least for the response)
    m = LinearMixedModel(m.formula, newdata; REML=m.optsum.REML)
    simulate!(rng, y, m,
              β=β, σ=σ, θ=θ)
    y
end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::GeneralizedLinearMixedModel, newdata::Tables.ColumnTable;
                   β = coef(m), σ = m.σ, θ = T[])
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    # (for one thing, this still allocates for the model's response)
    # note that the contrasts don't matter for simulation purposes
    # (at least for the response)
    m = GeneralizedLinearMixedModel(m.formula, newdata, m.resp.d, m.resp.l)
    simulate!(rng, y, m,
              β=β, σ=σ, θ=θ)
    y
end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::MixedModel, newdata;
                   β = coef(m), σ = m.σ, θ = T[])

    simulate!(rng, y, m, Tables.columntable(newdata);
              β=β, σ=σ, θ=θ)
end

function simulate!(y::AbstractVector, m::MixedModel, newdata;
                   β = coef(m), σ = m.σ, θ = T[])
    simulate!(Random.GLOBAL_RNG, y, m, Tables.columntable(newdata);
              β=β, σ=σ, θ=θ)
end

function simulate(rng::AbstractRNG, m::MixedModel, X = m.X;
                  β = coef(m), σ = m.σ, θ = T[])
    y = zeros(T, nobs(m))
    simulate!(rng, y, m, X;
              β=β, σ=σ, θ=θ)
    y
end

function simulate(m::MixedModel, X = m.X;
                  β = coef(m), σ = m.σ, θ = T[])
    simulate(Random.GLOBAL_RNG, m, X;
             β=β, σ=σ, θ=θ)
end

## should we add in code for prediction intervals?
# we don't directly implement (Wald) confidence intervals, so directly
# supporting (Wald) prediction intervals seems a step too far right now