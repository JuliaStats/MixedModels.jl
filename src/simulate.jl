"""
    parametricbootstrap(N::Integer, m::LinearMixedModel,
        rng=Random.GLOBAL_RNG, props=(:objective:σ,:β,:θ), β=m.β, σ=m.σ, θ=m.θ)

Perform `N` parametric bootstrap replication fits of `m`, returning a `Tables.RowTable`
of properties of the refit model given by the tuple of symbols `props`.

# Named Arguments

`β`, `σ`, and `θ` are the values of the parameters in `m` for simulation of the responses.
"""
function parametricbootstrap(N::Integer, m::LinearMixedModel{T},
        rng::AbstractRNG=Random.GLOBAL_RNG, props=(:objective, :σ, :β, :θ),
        β::Vector{T} = m.β, σ::T = m.σ, θ::Vector{T} = m.θ) where {T}
    y₀ = copy(response(m))          # to restore original state of m
    n, p, q, nre = size(m)
    length(β) == p && length(θ) == (k = length(getθ(m))) || throw(DimensionMismatch(""))
    baseval = getproperty.(Ref(m), props)
    ptype = typeof(baseval)
    val = [NamedTuple{props, ptype}(
        getproperty.(Ref(refit!(simulate!(rng, m, β=β, σ=σ, θ=θ))), props)) for _ in Base.OneTo(N)]
    refit!(m, y₀)                   # restore original state
    val
end

#=
function stddevcor!(σ::Vector{T}, ρ::Matrix{T}, scr::Matrix{T}, L::Cholesky{T}) where {T}
    @argcheck(length(σ) == (k = size(L, 2)) && size(ρ) == (k, k) && size(scr) == (k, k),
        DimensionMismatch)
    if L.uplo == 'L'
        copyto!(scr, L.factors)
        for i in 1 : k
            σ[i] = σi = norm(view(scr, i, 1:i))
            for j in 1 : i
                scr[i, j] /= σi
            end
        end
        mul!(ρ, LowerTriangular(scr), adjoint(LowerTriangular(scr)))
    elseif L.uplo == 'U'
        copyto!(scr, L.factors)
        for j in 1 : k
            σ[j] = σj = norm(view(scr, 1:j, j))
            for i in 1 : j
                scr[i, j] /= σj
            end
        end
        mul!(ρ, UpperTriangular(scr)', UpperTriangular(scr))
    else
        throw(ArgumentError("L.uplo should be 'L' or 'U'"))
    end
    σ, ρ
end

function stddevcor!(σ::Vector, ρ::Matrix, scr::Matrix, L::ScalarFactorReTerm)
    σ[1] = L.Λ
    ρ[1] = 1
end

function stddevcor!(σ::Vector{T}, ρ::Matrix{T}, scr::Matrix{T},
    L::VectorFactorReTerm{T}) where {T}
    stddevcor!(σ, ρ, scr, Cholesky(L.Λ))
end

function stddevcor(L::Cholesky{T}) where {T}
    k = size(L, 1)
    stddevcor!(Vector{T}(undef, k), Matrix{T}(undef, k, k), Matrix{T}(undef, k, k), L)
end

stddevcor(L::LowerTriangular) = stddevcor(Cholesky(L))
stddevcor(L::VectorFactorReTerm) = stddevcor(L.Λ)
stddevcor(L::ScalarFactorReTerm{T}) where {T} = [L.Λ], ones(T, 1, 1)

function LinearAlgebra.Cholesky(L::LowerTriangular)  # FIXME: this is type piracy 
    info = 0
    for k in 1:size(L,2)
        if iszero(L[k, k])
            info = k
            break
        end
    end
    Cholesky(L, 'L', LinearAlgebra.BlasInt(info))
end
=#
"""
    unscaledre!(y::AbstractVector{T}, M::AbstractFactorReTerm{T}, b) where {T}
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

simulate!(m::LinearMixedModel{T}; β=m.β, σ=m.σ, θ=T[]) where {T} =
    simulate!(Random.GLOBAL_RNG, m, β=β, σ=σ, θ=θ)
