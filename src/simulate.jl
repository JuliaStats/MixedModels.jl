"""
See [`simulate!`](@ref)
"""
function simulate end

function simulate(rng::AbstractRNG, m::MixedModel{T}, newdata; kwargs...) where {T}
    dat = Tables.columntable(newdata)
    y = zeros(T, length(first(dat)))
    return simulate!(rng, y, m, newdata; kwargs...)
end

function simulate(rng::AbstractRNG, m::MixedModel; kwargs...)
    return simulate!(rng, similar(response(m)), m; kwargs...)
end

function simulate(m::MixedModel, args...; kwargs...)
    return simulate(Random.GLOBAL_RNG, m, args...; kwargs...)
end

"""
    simulate!(rng::AbstractRNG, m::MixedModel{T}; β=fixef(m), σ=m.σ, θ=T[])
    simulate!(m::MixedModel; β=fixef(m), σ=m.σ, θ=m.θ)

Overwrite the response (i.e. `m.trms[end]`) with a simulated response vector from model `m`.

This simulation includes sampling new values for the random effects.

`β` can be specified either as a pivoted, full rank coefficient vector (cf. [`fixef`](@ref))
or as an unpivoted full dimension coefficient vector (cf. [`coef`](@ref)), where the entries
corresponding to redundant columns will be ignored.

!!! note
    Note that `simulate!` methods with a `y::AbstractVector` as the first argument
    (besides the RNG) and `simulate` methods return the simulated response. This is
    in contrast to `simulate!` methods with a `m::MixedModel` as the first argument,
    which modify the model's response and return the entire modified model.
"""
function simulate!(
    rng::AbstractRNG, m::LinearMixedModel{T}; β=fixef(m), σ=m.σ, θ=T[]
) where {T}
    # XXX should we add support for doing something with weights?
    simulate!(rng, m.y, m; β, σ, θ)
    return unfit!(m)
end

function simulate!(
    rng::AbstractRNG, m::GeneralizedLinearMixedModel{T}; β=fixef(m), σ=m.σ, θ=T[]
) where {T}
    # note that these m.resp.y and m.LMM.y will later be synchronized in (re)fit!()
    # but for now we use them as distinct scratch buffers to avoid allocations

    # the noise term is actually in the GLM and not the LMM part so no noise
    # at the LMM level
    η = fill!(copy(m.LMM.y), zero(T))  # ensure that η is a vector - needed for GLM.updateμ! below
    # A better approach is to change the signature for updateμ!
    y = m.resp.y

    _simulate!(rng, y, η, m, β, σ, θ, m.resp)

    return unfit!(m)
end

"""
    _rand(rng::AbstractRNG, d::Distribution, location, scale=missing, n=1)

A convenience function taking a draw from a distribution.

Note that `d` is specified as an existing distribution, such as
from the `GlmResp.d` field. This isn't vectorized nicely because
for distributions where the scale/dispersion is dependent on the
location (e.g. Bernoulli, Binomial, Poisson), it's not really
possible to avoid creating multiple `Distribution` objects.

Note that `n` is the `n` parameter for the Binomial distribution,
*not* the number of draws from the RNG. It is then used to change the
random draw (an integer in [0, n]) into a probability (a float in [0,1]).
"""
function _rand(rng::AbstractRNG, d::Distribution, location, scale=NaN, n=1)
    if !ismissing(scale)
        throw(ArgumentError("Families with a dispersion parameter not yet supported"))
    end

    if d isa Binomial
        dist = Binomial(Int(n), location)
    else
        dist = typeof(d)(location)
    end

    return rand(rng, dist) / n
end

function simulate!(m::MixedModel{T}; β=fixef(m), σ=m.σ, θ=T[]) where {T}
    return simulate!(Random.GLOBAL_RNG, m; β, σ, θ)
end

"""
    simulate!([rng::AbstractRNG,] y::AbstractVector, m::MixedModel{T}[, newdata];
                    β = coef(m), σ = m.σ, θ = T[], wts=m.wts)
    simulate([rng::AbstractRNG,] m::MixedModel{T}[, newdata];
                    β = coef(m), σ = m.σ, θ = T[], wts=m.wts)

Simulate a new response vector, optionally overwriting a pre-allocated vector.

New data can be optionally provided in tabular format.

This simulation includes sampling new values for the random effects. Thus in
contrast to `predict`, there is no distinction in between "new" and
"old" / previously observed random-effects levels.

Unlike `predict`, there is no `type` parameter for `GeneralizedLinearMixedModel`
because the noise term in the model and simulation is always on the response
scale.

The `wts` argument is currently ignored except for `GeneralizedLinearMixedModel`
models with a `Binomial` distribution.

!!! note
    Note that `simulate!` methods with a `y::AbstractVector` as the first argument
    (besides the RNG) and `simulate` methods return the simulated response. This is
    in contrast to `simulate!` methods with a `m::MixedModel` as the first argument,
    which modify the model's response and return the entire modified model.
"""
function simulate!(
    rng::AbstractRNG,
    y::AbstractVector,
    m::LinearMixedModel,
    newdata::Tables.ColumnTable;
    β=fixef(m),
    σ=m.σ,
    θ=m.θ,
)
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    # (for one thing, this still allocates for the model's response)
    # note that the contrasts get copied over with the formula
    # (as part of the applied schema)
    # contr here are the fast Grouping contrasts
    f, contr = _abstractify_grouping(m.formula)
    mnew = LinearMixedModel(f, newdata; contrasts=contr)
    # XXX why not do simulate!(rng, y, mnew; β=β, σ=σ, θ=θ)
    # instead of simulating the model and then copying?
    # Well, it turns out that the call to randn!(rng, y)
    # gives different results at the tail end of the array
    # for y <: view(::Matrix{Float64}, :, 3) than y <: Vector{Float64}
    # I don't know why, but this doesn't actually incur an
    # extra computation and gives consistent results at the price
    # of an allocationless copy
    simulate!(rng, mnew; β, σ, θ)
    return copy!(y, mnew.y)
end

function simulate!(
    rng::AbstractRNG, y::AbstractVector, m::LinearMixedModel{T}; β=fixef(m), σ=m.σ, θ=m.θ
) where {T}
    length(β) == length(pivot(m)) || length(β) == rank(m) ||
        throw(ArgumentError("You must specify all (non-singular) βs"))

    β = convert(Vector{T}, β)
    σ = T(σ)
    θ = convert(Vector{T}, θ)
    isempty(θ) || setθ!(m, θ)

    if length(β) == length(pivot(m))
        β = view(view(β, pivot(m)), 1:rank(m))
    end

    # initialize y to standard normal
    randn!(rng, y)

    # add the unscaled random effects
    for trm in m.reterms
        unscaledre!(rng, y, trm)
    end

    # scale by σ and add fixed-effects contribution
    return mul!(y, fullrankx(m), β, one(T), σ)
end

function simulate!(
    rng::AbstractRNG,
    y::AbstractVector,
    m::GeneralizedLinearMixedModel,
    newdata::Tables.ColumnTable;
    β=fixef(m),
    σ=m.σ,
    θ=m.θ,
)
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    # (for one thing, this still allocates for the model's response)
    # note that the contrasts get copied over with the formula
    # (as part of the applied schema)
    # contr here are the fast Grouping contrasts
    f, contr = _abstractify_grouping(m.formula)
    mnew = GeneralizedLinearMixedModel(f, newdata, m.resp.d, Link(m.resp); contrasts=contr)
    # XXX why not do simulate!(rng, y, mnew; β, σ, θ)
    # instead of simulating the model and then copying?
    # Well, it turns out that the call to randn!(rng, y)
    # gives different results at the tail end of the array
    # for y <: view(::Matrix{Float64}, :, 3) than y <: Vector{Float64}
    # I don't know why, but this doesn't actually incur an
    # extra computation and gives consistent results at the price
    # of an allocationless copy
    simulate!(rng, mnew; β, σ, θ)
    return copy!(y, mnew.y)
end

function simulate!(
    rng::AbstractRNG,
    y::AbstractVector,
    m::GeneralizedLinearMixedModel{T};
    β=fixef(m),
    σ=m.σ,
    θ=m.θ,
) where {T}
    # make sure both scratch arrays are init'd to zero
    η = zeros(T, size(y))
    copyto!(y, η)
    return _simulate!(rng, y, η, m, β, σ, θ)
end

function _simulate!(
    rng::AbstractRNG,
    y::AbstractVector,
    η::AbstractVector,
    m::GeneralizedLinearMixedModel{T},
    β,
    σ,
    θ,
    resp=nothing,
) where {T}
    length(β) == length(pivot(m)) || length(β) == m.feterm.rank ||
        throw(ArgumentError("You must specify all (non-singular) βs"))

    dispersion_parameter(m) ||
        ismissing(σ) ||
        throw(
            ArgumentError(
                "You must not specify a dispersion parameter for model families without a dispersion parameter"
            ),
        )

    β = convert(Vector{T}, β)
    if σ !== missing
        σ = T(σ)
    end
    θ = convert(Vector{T}, θ)

    d = m.resp.d

    if length(β) == length(pivot(m))
        # unlike LMM, GLMM stores the truncated, pivoted vector directly
        β = view(view(β, pivot(m)), 1:rank(m))
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
    mul!(η, fullrankx(lm), β, one(T), one(T))

    μ = resp === nothing ? linkinv.(Link(m), η) : GLM.updateμ!(resp, η).mu

    # convert to the distribution / add in noise
    @inbounds for (idx, val) in enumerate(μ)
        n = isempty(m.wt) ? 1 : m.wt[idx]
        y[idx] = _rand(rng, d, val, σ, n)
    end

    return y
end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::MixedModel, newdata; kwargs...)
    return simulate!(rng, y, m, Tables.columntable(newdata); kwargs...)
end
function simulate!(y::AbstractVector, m::MixedModel, newdata; kwargs...)
    return simulate!(Random.GLOBAL_RNG, y, m, Tables.columntable(newdata); kwargs...)
end

"""
    unscaledre!(y::AbstractVector{T}, M::ReMat{T}) where {T}
    unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, M::ReMat{T}) where {T}

Add unscaled random effects simulated from `M` to `y`.

These are unscaled random effects (i.e. they incorporate λ but not σ) because
the scaling is done after the per-observation noise is added as a standard normal.
"""
function unscaledre! end

function unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,S}) where {T,S}
    return mul!(y, A, vec(lmul!(A.λ, randn(rng, S, nlevs(A)))), one(T), one(T))
end

function unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,1}) where {T}
    return mul!(y, A, lmul!(first(A.λ), randn(rng, nlevs(A))), one(T), one(T))
end

unscaledre!(y::AbstractVector, A::ReMat) = unscaledre!(Random.GLOBAL_RNG, y, A)
