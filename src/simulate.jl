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

!!! warning
    The modified model is currently not marked as being unfitted.
    More precisely, its internal `OptSum` structure is not modified.
    As such, displaying the model will show the previous fit, even though
    this no longer corresponds to the new, modified response.

    This behavior may change in a future release
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
    unscaledre!(y::AbstractVector{T}, M::ReMat{T}, b) where {T}
    unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, M::ReMat{T}) where {T}

Add unscaled random effects defined by `M` and `b` to `y`.  When `rng` is present the `b`
vector is generated as `randn(rng, size(M, 2))`
"""
function unscaledre! end

function unscaledre!(y::AbstractVector{T}, A::ReMat{T,1}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    length(y) == m && length(b) == n || throw(DimensionMismatch(""))
    z = A.z
    @inbounds for (i, r) in enumerate(A.refs)
        y[i] += b[r] * z[i]
    end
    y
end

unscaledre!(y::AbstractVector{T}, A::ReMat{T,1}, B::AbstractMatrix{T}) where {T} =
    unscaledre!(y, A, vec(B))

# the compiler will actually create distinct methods for each of the types in
# the outer Union
function unscaledre!(y::AbstractVector{T}, A::ReMat{T,S},
                     b::Union{AbstractMatrix{T},
                              AbstractMatrix{Union{T, Missing}} }) where {T,S}
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
