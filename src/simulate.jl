"""
See [`simulate!`](@ref)
"""
function simulate end

function simulate(rng::AbstractRNG, m::MixedModel{T}, newX::AbstractMatrix = m.X;
                  kwargs...) where {T}
    size(newX, 1) == nobs(m) ||
        throw(DimensionMismatch("New fixed-effect model matrix must have the same number of observations as the original."))
    size(newX, 2) == size(m.X, 2) ||
    throw(DimensionMismatch("New fixed-effect model matrix must have the same predictors as the original."))
    y = zeros(T, nobs(m))
    simulate!(rng, y, m, newX; kwargs...)
    y
end

function simulate(m::MixedModel, newX::AbstractMatrix = m.X; kwargs...)
    simulate(Random.GLOBAL_RNG, m, newX; kwargs...)
end

function simulate(rng::AbstractRNG, m::MixedModel{T}, newdata; kwargs...) where {T}
    dat = Tables.columntable(newdata)
    y = zeros(T, length(first(dat)))
    simulate!(rng, y, m, newdata; kwargs...)
end

function simulate(m::MixedModel, newdata; kwargs...)
    simulate(Random.GLOBAL_RNG, m, newdata; kwargs...)
end

"""
    simulate!(rng::AbstractRNG, m::MixedModel{T}; β=m.β, σ=m.σ, θ=T[])
    simulate!(m::MixedModel; β=m.β, σ=m.σ, θ=m.θ)

Overwrite the response (i.e. `m.trms[end]`) with a simulated response vector from model `m`.

This simulation includes sampling new values for the random effects.

!!! note
    Note that `simulate!` methods with a `y::AbstractVector` as the first argument
    (besides the RNG) and `simulate` methods return the simulated response. This is
    in contrast to `simulate!` methods with a `m::MixedModel` as the first argument,
    which modify the model's response and return the entire modified model.
"""
function simulate!(
    rng::AbstractRNG,
    m::LinearMixedModel{T};
    β = coef(m),
    σ = m.σ,
    θ = T[],
) where {T}
    simulate!(rng, m.y, m; β=β, σ=σ, θ=θ)
    unfit!(m)
end

function simulate!(
    rng::AbstractRNG,
    m::GeneralizedLinearMixedModel{T};
    β = coef(m),
    σ = m.σ,
    θ = T[],
) where {T}
    # note that these m.resp.y and m.LMM.y will later be sychronized in (re)fit!()
    # but for now we use them as distinct scratch buffers to avoid allocations

    # the noise term is actually in the GLM and not the LMM part so no noise
    # at the LMM level
    η = fill!(copy(m.LMM.y), zero(T))  # ensure that η is a vector - needed for GLM.updateμ! below
                                       # A better approach is to change the signature for updateμ!
    y = m.resp.y

    _simulate!(rng, y, η, m.resp, m, m.X, β, σ, θ, m.resp.wts)

    unfit!(m)
end

"""
    _rand(rng::AbstractRNG, d::Distribution, location, scale=missing, n=1)

A convenience function taking a draw from a distrbution.

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

    rand(rng, dist) / n
end

function simulate!(m::MixedModel{T}; β = coef(m), σ = m.σ, θ = T[]) where {T}
    simulate!(Random.GLOBAL_RNG, m, β = β, σ = σ, θ = θ)
end

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

Unlike [`predict`](`@ref`), there is no `type` parameter for [`GeneralizedLinearMixedModel`](@ref)
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
                   β=m.β, σ=m.σ, θ=m.θ)
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    # (for one thing, this still allocates for the model's response)
    # note that the contrasts get copied over with the formula
    # (as part of the applied schema)
    # contr here are the fast Grouping contrasts
    f, contr = _abstractify_grouping(m.formula)
    mnew = LinearMixedModel(f, newdata; contrasts=contr)

    simulate!(rng, y, mnew; β, σ, θ)
    y
end

function simulate!(rng::AbstractRNG, y::AbstractVector, m::GeneralizedLinearMixedModel, newdata::Tables.ColumnTable;
                   β=m.β, σ=m.σ, θ=m.θ)
    # the easiest thing here is to just assemble a new model and
    # pass that to the other simulate methods....
    # this can probably be made much more efficient
    # (for one thing, this still allocates for the model's response)
    # note that the contrasts get copied over with the formula
    # (as part of the applied schema)
    # contr here are the fast Grouping contrasts
    f, contr = _abstractify_grouping(m.formula)
    mnew = GeneralizedLinearMixedModel(f, newdata, m.resp.d, Link(m.resp); contrasts=contr)
    simulate!(rng, y, mnew; β, σ, θ)
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

"""
    unscaledre!(y::AbstractVector{T}, M::ReMat{T}, b) where {T}
    unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, M::ReMat{T}) where {T}

Add unscaled random effects defined by `M` and `b` to `y`.  When `rng` is present the `b`
vector is generated as `randn(rng, size(M, 2))`
"""
function unscaledre! end

function unscaledre!(y::AbstractVector{<:Union{T, Missing}}, A::ReMat{T,1}, b::AbstractVector{<:Union{T, Missing}}) where {T}
    m, n = size(A)
    length(y) == m && length(b) == n || throw(DimensionMismatch(""))
    z = A.z
    @inbounds for (i, r) in enumerate(A.refs)
        y[i] += b[r] * z[i]
    end
    y
end

unscaledre!(y::AbstractVector{<:Union{T, Missing}}, A::ReMat{T,1}, B::AbstractMatrix{<:Union{T, Missing}}) where {T} =
    unscaledre!(y, A, vec(B))

# the compiler will actually create distinct methods for each of the types in
# the outer Union
function unscaledre!(y::AbstractVector{<:Union{T, Missing}}, A::ReMat{T,S},
                     b::AbstractMatrix{<:Union{T, Missing}}) where {T,S}
    Z = A.z
    k, n = size(Z)
    l = nlevs(A)
    length(y) == n && size(b) == (k, l) || throw(DimensionMismatch(""))
    @inbounds for (i, ii) in enumerate(A.refs)
        for j = 1:k
            y[i] += Z[j, i] * b[j, ii]
        end
    end
    y
end

function unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,S}) where {T,S}
    rng_nums = randn(rng, S, nlevs(A))

    unscaledre!(y, A, lmul!(A.λ, rng_nums))
end

function unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,1}) where {T}
    rng_nums = randn(rng, 1, nlevs(A))

    unscaledre!(y, A, lmul!(first(A.λ), rng_nums))
end

unscaledre!(y::AbstractVector, A::ReMat) = unscaledre!(Random.GLOBAL_RNG, y, A)
