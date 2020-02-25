"""
    MixedModelBootstrap{T<:AbstractFloat}

Object returned by `parametericbootstrap` with fields
- `bstr`: the parameter estimates from the bootstrap replicates as a vector of named tuples.
- ``
The schema of `bstr` is, by default,
```
Tables.Schema:
 :objective  T
 :σ          T
 :β          StaticArrays.SArray{Tuple{p},T,1,p}
 :θ          StaticArrays.SArray{Tuple{p},T,1,k}
```
where the sizes, `p` and `k`, of the `β` and `θ` elements are determined by the model.

Characteristics of the bootstrap replicates can be extracted as properties.  The `σs` and
`σρs` properties unravel the `σ` and `θ` estimates into estimates of the standard deviations
and correlations of the random-effects terms.
"""
struct MixedModelBootstrap{T<:AbstractFloat}
    bstr::Vector
    λ::Vector{LowerTriangular{T,Matrix{T}}}
    inds::Vector{Vector{Int}}
    lowerbd::Vector{T}
    fcnames::NamedTuple
end

"""
    parametricbootstrap(rng::AbstractRNG, nsamp::Integer, m::LinearMixedModel;
        β = coef(m), σ = m.σ, θ = m.θ, use_threads=false)
    parametricbootstrap(nsamp::Integer, m::LinearMixedModel;
        β = coef(m), σ = m.σ, θ = m.θ, use_threads=false)

Perform `nsamp` parametric bootstrap replication fits of `m`, returning a `MixedModelBootstrap`.

The default random number generator is `Random.GLOBAL_RNG`.

# Named Arguments

`β`, `σ`, and `θ` are the values of `m`'s parameters for simulating the responses.
`use_threads` determines whether or not to use thread-based parallelism.
"""
function parametricbootstrap(
    rng::AbstractRNG,
    n::Integer,
    morig::LinearMixedModel{T};
    β = coef(morig),
    σ = morig.σ,
    θ = morig.θ,
    use_threads = false,
) where {T}
    β = convert(Vector{T},β)
    σ = T(σ)
    θ = convert(Vector{T},θ)
    βsc, θsc, p, k, m = similar(β), similar(θ), length(β), length(θ), deepcopy(morig)

    β_names = (Symbol.(fixefnames(morig))..., )
    rank = length(β_names)

    # we need arrays of these for in-place operations to work across threads
    m_threads = [m]
    βsc_threads = [βsc]
    θsc_threads = [θsc]

    if use_threads
        Threads.resize_nthreads!(m_threads)
        Threads.resize_nthreads!(βsc_threads)
        Threads.resize_nthreads!(θsc_threads)
    end
    # we use locks to guarantee thread-safety, but there might be better ways to do this for some RNGs
    # see https://docs.julialang.org/en/v1.3/manual/parallel-computing/#Side-effects-and-mutable-function-arguments-1
    # see https://docs.julialang.org/en/v1/stdlib/Future/index.html
    rnglock = ReentrantLock()
    samp = replicate(n, use_threads=use_threads) do
        mod = m_threads[Threads.threadid()]
        local βsc = βsc_threads[Threads.threadid()]
        local θsc = θsc_threads[Threads.threadid()]
        lock(rnglock)
        mod = simulate!(rng, mod, β = β, σ = σ, θ = θ)
        unlock(rnglock)
        refit!(mod)
        (
         objective = mod.objective,
         σ = mod.σ,
         β = NamedTuple{β_names}(fixef!(βsc, mod)),
         θ = SVector{k,T}(getθ!(θsc, mod)),
        )
    end
    MixedModelBootstrap(
        samp,
        deepcopy(morig.λ),
        getfield.(morig.reterms, :inds),
        copy(morig.optsum.lowerbd),
        NamedTuple{Symbol.(fnames(morig))}(map(t -> Symbol.((t.cnames...,)), morig.reterms)),
    )
end

function parametricbootstrap(
    nsamp::Integer,
    m::LinearMixedModel;
    β = m.β,
    σ = m.σ,
    θ = m.θ,
    use_threads = false
)
    parametricbootstrap(Random.GLOBAL_RNG, nsamp, m, β=β, σ=σ, θ=θ, use_threads=use_threads)
end

function Base.getproperty(bsamp::MixedModelBootstrap, s::Symbol)
    if s ∈ [:objective, :σ, :θ]
        getproperty.(getfield(bsamp, :bstr), s)
    elseif s == :β
        tidyβ(bsamp)
    elseif s == :σs
        tidyσs(bsamp)
    elseif s == :σρs
        σρs(bsamp)
    else
        getfield(bsamp, s)
    end
end

issingular(bsamp::MixedModelBootstrap) = map(θ -> any(θ .≈ bsamp.lowerbd), bsamp.θ)

function Base.propertynames(bsamp::MixedModelBootstrap)
    [:objective, :σ, :β, :θ, :σs, :σρs, :λ, :inds, :lowerbd, :bstr, :fcnames]
end

"""
    setθ!(bsamp::MixedModelsBootstrap, i::Integer)

Install the values of the i'th θ value of `bsamp.bstr` in `bsamp.λ`
"""
function setθ!(bsamp::MixedModelBootstrap, i::Integer) where {T}
    θ = bsamp.bstr[i].θ
    offset = 0
    for (λ, inds) in zip(bsamp.λ, bsamp.inds)
        λdat = λ.data
        fill!(λdat, false)
        for j in eachindex(inds)
            λdat[inds[j]] = θ[j + offset]
        end
        offset += length(inds)
    end
    bsamp
end

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
    Tuple(vv[[i, i + ilen - 1]])
end

function tidyβ(bsamp::MixedModelBootstrap{T}) where {T}
    bstr = bsamp.bstr
    colnms = (:iter, :coefname, :β)
    result = sizehint!(
        NamedTuple{colnms,Tuple{Int,Symbol,T}}[],
        length(bstr) * length(first(bstr).β),
    )
    for (i, r) in enumerate(bstr)
        for (k, v) in pairs(r.β)
            push!(result, NamedTuple{colnms}((i, k, v)))
        end
    end
    result
end

function tidyσs(bsamp::MixedModelBootstrap{T}) where {T}
    bstr = bsamp.bstr
    fcnames = bsamp.fcnames
    λ = bsamp.λ
    colnms = (:iter, :group, :column, :σ)
    result = sizehint!(
        NamedTuple{colnms,Tuple{Int,Symbol,Symbol,T}}[],
        length(bstr) * sum(length, fcnames),
    )
    for (iter, r) in enumerate(bstr)
        setθ!(bsamp, iter)    # install r.θ in λ
        σ = r.σ
        for (grp, ll) in zip(keys(fcnames), λ)
            for (cn, col) in zip(getproperty(fcnames, grp), eachrow(ll))
                push!(result, NamedTuple{colnms}((iter, grp, cn, σ * norm(col))))
            end
        end
    end
    result
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

function simulate!(m::LinearMixedModel{T}; β = coef(m), σ = m.σ, θ = T[]) where {T}
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
