"""
    bootstrap(N, m::LinearMixedModel; β::Vector=fixef(m), σ=sdest(m), θ::Vector=getθ(m))

Perform `N` parametric bootstrap replication fits of `m`, returning a `Tables.RowTable`
of properties of the refit model given by the tuple of symbols `props`.

# Named Arguments

`β`, `σ`, and `θ` are the values of the parameters in `m` for simulation of the responses.
"""
function parametricbootstrap(N::Integer, m::LinearMixedModel,
        rng::AbstractRNG=Random.GLOBAL_RNG, props=(:objective, :σ, :β, :θ),
        β = copy(m.β), σ = m.σ, θ = copy(m.θ))
    y₀ = copy(response(m))          # to restore original state of m
    n, p, q, nre = size(m)
    length(β) == p && length(θ) == (k = length(getθ(m))) || throw(DimensionMismatch(""))
    baseval = getproperty.(Ref(m), props)
    ptype = typeof(baseval)
    val = [NamedTuple{props, ptype}(
        getproperty.(Ref(refit!(simulate!(rng, m, β=β, σ=σ, θ=θ))), props)) for _ in 1:N]
    refit!(m, y₀)                   # restore original state
    val
end

"""
    unscaledre!(y::AbstractVector{T}, M::ReMat{T}, b) where {T}
    unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, M::AbstractFactorReTerm{T}) where {T}

Add unscaled random effects defined by `M` and `b` to `y`.  When `rng` is present the `b`
vector is generated as `randn(rng, size(M, 2))`
"""
function unscaledre! end

function unscaledre!(y::AbstractVector{T}, A::ReMat{T,R,1}, b::AbstractVector{T}) where {T,R}
    m, n = size(A)
    length(y) == m && length(b) == n || throw(DimensionMismatch(""))
    z = A.z
    for (i, r) in enumerate(A.refs)
        y[i] += b[r] * z[i]
    end
    y
end

unscaledre!(y::AbstractVector{T}, A::ReMat{T,R,1}, B::AbstractMatrix{T}) where {T,R} = 
    unscaledre!(y, A, vec(B))

function unscaledre!(y::AbstractVector{T}, A::ReMat{T,R,S}, b::AbstractMatrix{T}) where {T,R,S}
    Z = A.z
    k, n = size(Z)
    l = nlevs(A)
    length(y) == n && size(b) == (k, l) || throw(DimensionMismatch(""))
    for (i, ii) in enumerate(A.refs)
        for j in 1:k
            y[i] += Z[j, i] * b[j, ii]
        end
    end
    y
end

unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T}) where {T} =
    unscaledre!(y, A, lmul!(A.λ, randn(rng, vsize(A), nlevs(A))))

unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, A::ReMat{T,R,1}) where {T,R} =
    unscaledre!(y, A, lmul!(first(A.λ), randn(rng, vsize(A), nlevs(A))))

unscaledre!(y::AbstractVector, A::ReMat) = unscaledre!(Base.GLOBAL_RNG, y, A)

"""
    simulate!(m::LinearMixedModel; β=fixef(m), σ=sdest(m), θ=getθ(m))

Overwrite the response (i.e. `m.trms[end]`) with a simulated response vector from model `m`.
"""
function simulate!(rng::AbstractRNG, m::LinearMixedModel{T}; 
        β=coef(m), σ=sdest(m), θ=T[]) where {T}
    if !isempty(θ)
        setθ!(m, θ)
    end
    y = randn!(rng, response(m))      # initialize to standard normal noise
    for trm in m.reterms              # add the unscaled random effects
        unscaledre!(rng, y, trm)
    end
                                  # scale by σ and add fixed-effects contribution
    BLAS.gemv!('N', one(T), first(m.feterms).x, β, σ, y)
    m
end

simulate!(m::LinearMixedModel{T}; β=coef(m), σ=sdest(m), θ=T[]) where {T} =
    simulate!(Random.GLOBAL_RNG, m, β=β, σ=σ, θ=θ)
