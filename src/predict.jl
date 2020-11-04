StatsBase.predict(m::MixedModel) = fitted(m)

function StatsBase.predict(m::MixedModel, newX::AbstractMatrix)

end


"""

`new_re_levels`
    - `:population`
    - `:error`
    - `:simulate`

`scale`
    - `:linpred`
    - `:link`

"""
function StatsBase.predict(m::LinearMixedModel, newdata::ColumnTable;
                           new_re_levels=:population)

end

function StatsBase.predict(m::GeneralizedLinearMixedModel, newdata::ColumnTable;
                           scale=:linpred)

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
                   m::MixedModel{T},
                   newX::AbstractArray{T} = m.X;
                   β = coef(m),
                   σ = m.σ,
                   θ = T[],
               ) where {T}
# TODO: move the simulate!(::MixedModel) code here and
#       pass the model's own response as the pre-allocated vector

end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::LinearMixedModel, newdata::ColumnTable)
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    m = LinearMixedModel(m.formula, newdata, REML)
end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::MixedModel, newdata;
                   β = coef(m), σ = m.σ, θ = T[])

    simulate!(rng, y, m, columntable(newdata);
              β=β, σ=σ, θ=θ)
end

function simulate!(y::AbstractVector, m::MixedModel, newdata;
                   β = coef(m), σ = m.σ, θ = T[])
    simulate!(Random.GLOBAL_RNG, y, m, columntable(newdata);
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