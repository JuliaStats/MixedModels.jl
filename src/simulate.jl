struct MixedModelBootstrap{T<:AbstractFloat}
    m::LinearMixedModel{T}
    bstr::Tables.ColumnTable
    cnamedict::Dict{String,Int}
end

function MixedModelBootstrap(m::LinearMixedModel, bstr::NamedTuple) where {T}
    dict = Dict{String,Int}()
    k = findfirst(isequal(:β₁), keys(bstr))
    isnothing(k) && throw(ArgumentError("No :β₁ in keys(bstr)"))
    for nm in coefnames(m)
        dict[nm] = k
        k += 1
    end
    MixedModelBootstrap(m, bstr, dict)
end

function Base.getproperty(bsamp::MixedModelBootstrap, s::Symbol)
    if s == :model
        getfield(bsamp, :m)
    elseif s ∈ [:objective, :σ, :θ]
        getproperty(getfield(bsamp, :bstr), s)
    elseif s == :σs
        σs(bsamp)
    elseif s == :σρs
        σρs(bsamp)
    elseif haskey(getfield(bsamp, :cnamedict), string(s))
        getfield(bsamp, :bstr)[getfield(bsamp, :cnamedict)[string(s)]]
    else
        getfield(bsamp, s)
    end
end


"""
    parametricbootstrap(rng::AbstractRNG, nsamp::Integer, m::LinearMixedModel,
        props=(:objective, :σ, :β, :θ); β = m.β, σ = m.σ, θ = m.θ)
    parametricbootstrap(nsamp::Integer, m::LinearMixedModel,
        props=(:objective, :σ, :β, :θ); β = m.β, σ = m.σ, θ = m.θ)

Perform `nsamp` parametric bootstrap replication fits of `m`, returning a
`Tables.ColumnTable` of parameter estimates of the refit model.

The default random number generator is `Random.GLOBAL_RNG`.

# Named Arguments

`β`, `σ`, and `θ` are the values of `m`'s parameters for simulating the responses.
"""
function parametricbootstrap(rng::AbstractRNG, nsamp::Integer, m::LinearMixedModel{T};
    β = m.β, σ = m.σ, θ = m.θ) where {T}
    y₀ = copy(response(m))  # to restore original state of m
    θscr = copy(θ)
    βscr = copy(β)
    k = length(θ)
    bnms = Symbol.(subscriptednames("β", length(β)))
    vnms = (:objective, :σ, bnms..., :θ)
    value = (;(nm => nm == :θ ? SVector{k,T}[] : Vector{T}(undef, nsamp)
        for nm in vnms)...)
    try
        @showprogress 1 for i in 1:nsamp
            refit!(simulate!(rng, m, β = β, σ = σ, θ = θ))
            value.objective[i] = objective(m)
            value.σ[i] = sdest(m)
            fixef!(βscr, m)
            for (j, bnm) in enumerate(bnms)
                getproperty(value, bnm)[i] = βscr[j]
            end
            push!(value.θ, SVector{k, T}(getθ!(θscr, m)))
        end
    finally
        refit!(m, y₀)
    end
    MixedModelBootstrap(deepcopy(m), value)
end

function mktable(nsamp, p, k, T)
    nms = (:objective, :σ, Symbol.(subscriptednames("β", p))..., :θ)
    (; (nm => nm == :θ ? Vector{SVector{k,T}}(undef, nsamp) : Vector{T}(undef, nsamp) for nm in nms)...)
end


function parametricbootstrap(nsamp::Integer, m::LinearMixedModel, β = m.β, σ = m.σ, θ = m.θ)
    parametricbootstrap(Random.GLOBAL_RNG, nsamp, m, β = β, σ = σ, θ = θ)
end

function Base.propertynames(bsamp::MixedModelBootstrap)
    append!([:model, :objective, :σ, :θ, :σs, :σρs], Symbol.(keys(bsamp.cnamedict)))
end

function byreterm(bsamp::MixedModelBootstrap{T}, f::Function) where {T}
    m = bsamp.m
    oldθ = getθ(m)     # keep a copy to restore later
    retrms = m.reterms
    value = [typeof(v)[] for v in f.(retrms, m.σ)]
    for (σ,θ) in zip(bsamp.bstr.σ, bsamp.bstr.θ)
        setθ!(m, θ)
        for (i, v) in enumerate(f.(retrms, σ))
            push!(value[i], v)
        end
    end
    refit!(setθ!(m, oldθ))
    NamedTuple{(Symbol.(fnames(m))...,)}(value)
end

σs(bsamp::MixedModelBootstrap) = byreterm(bsamp, σs)

σρs(bsamp::MixedModelBootstrap) = byreterm(bsamp, σρs)

"""
    shortestCovInt(v, level = 0.95)

Return the shortest interval containing `level` proportion of the values of `v`
"""
function shortestCovInt(v, level = 0.95)
    n = length(v)
    0 < level < 1 || throw(ArgumentError("level = $level should be in (0,1)"))
    vv = issorted(v) ? v : sort(v)
    ilen = Int(ceil(n * level))   # the length of the interval in indices
    len, i = findmin([vv[i + ilen - 1] - vv[i] for i in 1:(n + 1 - ilen)])
    vv[[i, i + ilen - 1]]
end

"""
    simulate!(rng::AbstractRNG, m::LinearMixedModel{T}; β=m.β, σ=m.σ, θ=T[])
    simulate!(m::LinearMixedModel; β=m.β, σ=m.σ, θ=m.θ)

Overwrite the response (i.e. `m.trms[end]`) with a simulated response vector from model `m`.
"""
function simulate!(rng::AbstractRNG, m::LinearMixedModel{T}; β = coef(m), σ=m.σ, θ=T[]) where {T}
    isempty(θ) || setθ!(m, θ)
    y = randn!(rng, response(m))      # initialize y to standard normal
    for trm in m.reterms              # add the unscaled random effects
        unscaledre!(rng, y, trm)
    end
                    # scale by σ and add fixed-effects contribution
    BLAS.gemv!('N', one(T), m.X, β, σ, y)
    m
end

function simulate!(m::LinearMixedModel{T}; β=m.β, σ=m.σ, θ=T[]) where {T}
    simulate!(Random.GLOBAL_RNG, m, β=β, σ=σ, θ=θ)
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
        for j in 1:k
            y[i] += Z[j, i] * b[j, ii]
        end
    end
    y
end

unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,S}) where {T,S} =
    unscaledre!(y, A, lmul!(A.λ, randn(rng, S, nlevs(A))))

unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,1}) where {T} =
    unscaledre!(y, A, lmul!(first(A.λ), randn(rng, 1, nlevs(A))))

unscaledre!(y::AbstractVector, A::ReMat) = unscaledre!(Random.GLOBAL_RNG, y, A)
