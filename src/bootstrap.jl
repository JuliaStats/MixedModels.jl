"""
    MixedModelBootstrap{T<:AbstractFloat}

Object returned by `parametericbootstrap` with fields
- `bstr`: the parameter estimates from the bootstrap replicates as a vector of named tuples.
- `λ`: `Vector{LowerTriangular{T,Matrix{T}}}` containing copies of the λ field from `ReMat` model terms
- `inds`: `Vector{Vector{Int}}` containing copies of the `inds` field from `ReMat` model terms
- `lowerbd`: `Vector{T}` containing the vector of lower bounds (corresponds to the identically named field of [`OptSummary`](@ref))
- `fcnames`: NamedTuple whose keys are the grouping factor names and whose values are the column names

The schema of `bstr` is, by default,
```
Tables.Schema:
 :objective  T
 :σ          T
 :β          NamedTuple{β_names}{NTuple{p,T}}
 :se         StaticArrays.SArray{Tuple{p},T,1,p}
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
    β::AbstractVector=coef(morig),
    σ=morig.σ,
    θ::AbstractVector=morig.θ,
    use_threads::Bool=false,
) where {T}
    β, σ, θ = convert(Vector{T}, β), T(σ), convert(Vector{T}, θ)
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
         se = SVector{p,T}(stderror!(βsc, mod)),
         θ = SVector{k,T}(getθ!(θsc, mod)),
        )
    end
    MixedModelBootstrap(
        samp,
        deepcopy(morig.λ),
        getfield.(morig.reterms, :inds),
        copy(morig.optsum.lowerbd),
        NamedTuple{Symbol.(fnames(morig))}(map(t -> (t.cnames...,), morig.reterms)),
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

"""
    allpars(bsamp::MixedModelBootstrap)

Return a tidy (row)table with the parameter estimates spread into columns
of `iter`, `type`, `group`, `name` and `value`.
"""
function allpars(bsamp::MixedModelBootstrap{T}) where {T}
    bstr, λ, fcnames = bsamp.bstr, bsamp.λ, bsamp.fcnames
    npars = 2 + length(first(bstr).β) + sum(map(k -> (k * (k + 1)) >> 1, size.(bsamp.λ, 2)))
    nresrow = length(bstr) * npars
    cols = (
        sizehint!(Int[], nresrow),
        sizehint!(String[], nresrow), 
        sizehint!(Union{Missing,String}[], nresrow),
        sizehint!(Union{Missing,String}[], nresrow),
        sizehint!(T[], nresrow),
    )
    nrmdr = Vector{T}[]  # normalized rows of λ
    for (i, r) in enumerate(bstr)
        σ = r.σ
        for (nm, v) in pairs(r.β)
            push!.(cols, (i, "β", missing, String(nm), v))
        end
        setθ!(bsamp, i)
        for (grp, ll) in zip(keys(fcnames), λ)
            rownms = getproperty(fcnames, grp)
            grpstr = String(grp)
            empty!(nrmdr)
            for (j, rnm, row) in zip(eachindex(rownms), rownms, eachrow(ll))
                push!.(cols, (i, "σ", grpstr, rnm, σ * norm(row)))
                push!(nrmdr, normalize(row))
                for k in 1:(j - 1)
                    push!.(cols, (
                        i,
                        "ρ",
                        grpstr,
                        string(rownms[k], ", ", rnm),
                        dot(nrmdr[j], nrmdr[k])
                    ))
                end
            end
        end
        push!.(cols, (i, "σ", "residual", missing, σ))
    end
    (
        iter=cols[1],
        type=PooledArray(cols[2]),
        group=PooledArray(cols[3]),
        names=PooledArray(cols[4]),
        value=cols[5],
    )
end

function Base.getproperty(bsamp::MixedModelBootstrap, s::Symbol)
    if s ∈ [:objective, :σ, :θ]
        getproperty.(getfield(bsamp, :bstr), s)
    elseif s == :β
        tidyβ(bsamp)
    elseif s == :σs
        tidyσs(bsamp)
    elseif s == :allpars
        allpars(bsamp)
    else
        getfield(bsamp, s)
    end
end

issingular(bsamp::MixedModelBootstrap) = map(θ -> any(θ .≈ bsamp.lowerbd), bsamp.θ)

function Base.propertynames(bsamp::MixedModelBootstrap)
    [:allpars, :objective, :σ, :β, :θ, :σs, :λ, :inds, :lowerbd, :bstr, :fcnames]
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
    ilen = Int(ceil(n * level)) # number of elements (counting endpoints) in interval
                                # skip non-finite elements at the ends of sorted vv
    start = findfirst(isfinite, vv)
    stop = findlast(isfinite, vv)
    if stop < (start + ilen - 1)
        return (vv[1], vv[end])
    end
    len, i = findmin([vv[i+ilen-1] - vv[i] for i = start:(stop+1-ilen)])
    (vv[i], vv[i + ilen - 1])
end

"""
    tidyβ(bsamp::MixedModelBootstrap)
Return a tidy (row)table with the parameter estimates spread into columns
of `iter`, `coefname` and `β`
"""
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

"""
    tidyσs(bsamp::MixedModelBootstrap)
Return a tidy (row)table with the estimates of the variance components (on the standard deviation scale) spread into columns
of `iter`, `group`, `column` and `σ`.
"""
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
                push!(result, NamedTuple{colnms}((iter, grp, Symbol(cn), σ * norm(col))))
            end
        end
    end
    result
end
