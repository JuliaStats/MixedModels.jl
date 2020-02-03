"""
    MixedModelBootstrap{T<:AbstractFloat}

Object returned by `parametericbootstrap` with fields
- `m`: a copy of the model that was bootstrapped
- `bstr`: the parameter estimates from the bootstrap replicates as a vector of named tuples.

The schema of `bstr` is, by default,
```
Tables.Schema:
 :objective  T
 :σ          T
 :β          StaticArrays.SArray{Tuple{2},T,1,2}
 :θ          StaticArrays.SArray{Tuple{3},T,1,3}
```
where the sizes of the `β` and `θ` elements are determined by the model.

Characteristics of the bootstrap replicates can be extracted as properties.  The `σs` and
`σρs` properties unravel the `σ` and `θ` estimates into estimates of the standard deviations
and correlations of the random-effects terms.
"""
struct MixedModelBootstrap{T<:AbstractFloat}
    m::LinearMixedModel{T}
    bstr::Vector
end

function Base.getproperty(bsamp::MixedModelBootstrap, s::Symbol)
    if s == :model
        getfield(bsamp, :m)
    elseif s ∈ [:objective, :β, :σ, :θ]
        getproperty.(getfield(bsamp, :bstr), s)
    elseif s == :σs
        σs(bsamp)
    elseif s == :σρs
        σρs(bsamp)
    else
        getfield(bsamp, s)
    end
end

issingular(bsamp::MixedModelBootstrap) = issingular.(Ref(bsamp.m), bsamp.θ)

"""
    parametricbootstrap(rng::AbstractRNG, nsamp::Integer, m::LinearMixedModel;
        β = m.β, σ = m.σ, θ = m.θ)
    parametricbootstrap(nsamp::Integer, m::LinearMixedModel;
        β = m.β, σ = m.σ, θ = m.θ)

Perform `nsamp` parametric bootstrap replication fits of `m`, returning a `MixedModelBootstrap`.

The default random number generator is `Random.GLOBAL_RNG`.

# Named Arguments

`β`, `σ`, and `θ` are the values of `m`'s parameters for simulating the responses.
"""
function parametricbootstrap(
    rng::AbstractRNG,
    n::Integer,
    morig::LinearMixedModel{T};
    β = morig.β,
    σ = morig.σ,
    θ = morig.θ,
) where {T}
    βsc, θsc, p, k, m = similar(β), similar(θ), length(β), length(θ), deepcopy(morig)
    y₀ = copy(response(m))
    # we need to do for in-place operations to work across threads
    if Threads.nthreads() > 1
        m_threads = [deepcopy(m) for _ in Base.OneTo(Threads.nthreads())]
    else
        m_threads = [m]
    end
    # this assumes a thread-safe RNG. The default MersenneTwister seems to be ok on Linux
    # TODO: check thread safety issues
    # rng = MersenneTwister(42); replicate(10) do; [rand(rng,1),Threads.threadid()] ; end
    samp = replicate(n) do
        refit!(simulate!(rng, m_threads[Threads.threadid()], β = β, σ = σ, θ = θ))
        (
         objective = m.objective,
         σ = m.σ,
         β = SVector{p,T}(fixef!(βsc, m)),
         θ = SVector{k,T}(getθ!(θsc, m)),
        )
    end
    MixedModelBootstrap(refit!(m, y₀), samp)
end

function parametricbootstrap(nsamp::Integer, m::LinearMixedModel, β = m.β, σ = m.σ, θ = m.θ)
    parametricbootstrap(Random.GLOBAL_RNG, nsamp, m, β = β, σ = σ, θ = θ)
end

function Base.propertynames(bsamp::MixedModelBootstrap)
    [:model, :objective, :σ, :β, :θ, :σs, :σρs]
end

function byreterm(bsamp::MixedModelBootstrap{T}, f::Function) where {T}
    m = bsamp.m
    oldθ = getθ(m)     # keep a copy to restore later
    retrms = m.reterms
    value = [typeof(v)[] for v in f.(retrms, m.σ)]
    for r in bsamp.bstr
        setθ!(m, r.θ)
        for (i, v) in enumerate(f.(retrms, r.σ))
            push!(value[i], v)
        end
    end
    refit!(setθ!(m, oldθ))
    NamedTuple{(Symbol.(fnames(m))...,)}(value)
end

σs(bsamp::MixedModelBootstrap) = byreterm(bsamp, σs)

σρs(bsamp::MixedModelBootstrap) = byreterm(bsamp, σρs)

"""
    shortestcovint(v, level = 0.95)

Return the shortest interval containing `level` proportion of the values of `v`
"""
function shortestcovint(v, level = 0.95)
    n = length(v)
    0 < level < 1 || throw(ArgumentError("level = $level should be in (0,1)"))
    vv = issorted(v) ? v : sort(v)
    ilen = Int(ceil(n * level))   # the length of the interval in indices
    len, i = findmin([vv[i+ilen-1] - vv[i] for i = 1:(n+1-ilen)])
    vv[[i, i + ilen - 1]]
end

"""
    simulate!(rng::AbstractRNG, m::LinearMixedModel{T}; β=m.β, σ=m.σ, θ=T[])
    simulate!(m::LinearMixedModel; β=m.β, σ=m.σ, θ=m.θ)

Overwrite the response (i.e. `m.trms[end]`) with a simulated response vector from model `m`.
"""
function simulate!(
    rng::AbstractRNG,
    m::LinearMixedModel{T};
    β = coef(m),
    σ = m.σ,
    θ = T[],
) where {T}
    isempty(θ) || setθ!(m, θ)
    y = randn!(rng, response(m))      # initialize y to standard normal
    for trm in m.reterms              # add the unscaled random effects
        unscaledre!(rng, y, trm)
    end
                    # scale by σ and add fixed-effects contribution
    BLAS.gemv!('N', one(T), m.X, β, σ, y)
    m
end

function simulate!(m::LinearMixedModel{T}; β = m.β, σ = m.σ, θ = T[]) where {T}
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

unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,S}) where {T,S} =
    unscaledre!(y, A, lmul!(A.λ, randn(rng, S, nlevs(A))))

unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,1}) where {T} =
    unscaledre!(y, A, lmul!(first(A.λ), randn(rng, 1, nlevs(A))))

unscaledre!(y::AbstractVector, A::ReMat) = unscaledre!(Random.GLOBAL_RNG, y, A)
