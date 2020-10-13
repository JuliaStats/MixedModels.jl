"""
    simulate!(rng::AbstractRNG, m::LinearMixedModel{T}; β=m.β, σ=m.σ, θ=T[])
    simulate!(m::LinearMixedModel; β=m.β, σ=m.σ, θ=m.θ)
    simulate!(m::MixedModel; β=m.β, σ=(diom.σ, θ=m.θ)

Overwrite the response (i.e. `m.trms[end]`) with a simulated response vector from model `m`.
"""
function simulate!(
    rng::AbstractRNG,
    m::LinearMixedModel{T};
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

    y = randn!(rng, response(m))      # initialize y to standard normal

    for trm in m.reterms              # add the unscaled random effects
        unscaledre!(rng, y, trm)
    end
                    # scale by σ and add fixed-effects contribution
    BLAS.gemv!('N', one(T), m.X, β, σ, y)
    m
end

function simulate!(
    rng::AbstractRNG,
    m::GeneralizedLinearMixedModel{T};
    β = coef(m),
    σ = m.σ,
    θ = T[],
) where {T}
    length(β) == length(fixef(m)) ||
        length(β) == length(coef(m)) ||
            throw(ArgumentError("You must specify all (non-singular) βs"))

    dispersion_parameter(m) || isnan(σ) ||
        throw(ArgumentError("You must not specify a dispersion parameter for model families without a dispersion parameter"))

    β = convert(Vector{T},β)
    σ = T(σ)
    θ = convert(Vector{T},θ)

    d = m.resp.d
    l = GLM.Link(m.resp)

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

    # note that these m.resp.y and m.LMM.y will later be sychronized in (re)fit!()
    # but for now we use them as distinct scratch buffers to avoid allocations

    # should this be initialized with zeros or standard normal?
    # the noise term is actually in the GLM and not the LMM part....
    # so no noise at the LMM level
    lmy = fill!(m.LMM.y, zero(T))
    # lmy = randn!(rng, m.LMM.y)
    y = m.resp.y

    for trm in m.reterms             # add the unscaled random effects
        unscaledre!(rng, lmy, trm)
    end

    # do we need to worry about weights?

    # assemble the linear predictor
    # scale by lm.σ and add fixed-effects contribution
    BLAS.gemv!('N', one(T), lm.X, β, lm.σ, lmy)

    # from η to μ
    @inbounds for  idx in 1:length(y)
        y[idx] = GLM.linkinv(l, lmy[idx])
    end

    # convert to the distribution / add in noise
    @inbounds for (idx, val) in enumerate(y)
        y[idx] = _rand(rng, d, val, σ)
    end

    m
end

"""
    _rand(rng::AbstractRNG, d::Distribution, location, scale=NaN)

A convenience function taking a draw from a distrbution.

Note that `d` is specified as an existing distribution, such as
from the `GlmResp.d` field. This isn't vectorized nicely because
for distributions where the scale/dispersion is dependent on the
location (e.g. Bernoulli, Binomial, Poisson), it's not really
possible to avoid creating multiple `Distribution` objects.
"""
function _rand(rng::AbstractRNG, d::Distribution, location, scale=NaN)
    if !isnan(scale)
        throw(ArgumentError("Families with a dispersion parameter not yet supported"))
    end

    rand(rng, typeof(d)(location))
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

function unscaledre!(y::AbstractVector{T}, A::ReMat{T,S}, b::AbstractMatrix{T}) where {T,S}
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
