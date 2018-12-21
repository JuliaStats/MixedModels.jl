"""
    bootstrap(N, m::LinearMixedModel; β::Vector=fixef(m), σ=sdest(m), θ::Vector=getθ(m))

Perform `N` parametric bootstrap replication fits of `m`, returning a data frame
with column names `:obj`, the objective function at convergence, `:σ`, the estimated
standard deviation of the residuals, `βᵢ, i = 1,...,p`, the fixed-effects coefficients,
and `θᵢ, i = 1,...,k` the covariance parameters.

# Named Arguments

`β`, `σ`, and `θ` are the values of the parameters in `m` for simulation of the responses.
"""
function bootstrap(N, m::LinearMixedModel{T};
        β = fixef(m), σ = sdest(m), θ = getθ(m)) where {T}
    y₀ = copy(response(m))          # to restore original state of m
    n, p, q, nre = size(m)
    length(β) == p && length(θ) == (k = length(getθ(m))) || throw(DimensionMismatch(""))
    trms = m.reterms
    Λsize = vsize.(trms)
    cnms = vcat([:obj, :σ], Symbol.(subscriptednames('β', p)),
        Symbol.(subscriptednames('θ', k)), Symbol.(subscriptednames('σ', sum(Λsize))))
    nρ = [(l * (l - 1)) >> 1 for l in Λsize]
    if (nρtot = sum(nρ)) > 0
        append!(cnms, Symbol.(subscriptednames('ρ', nρtot)))
    end

    dfr = DataFrame(Any[Vector{T}(undef, N) for _ in eachindex(cnms)], cnms)
    scrβ = Vector{T}(undef, p)
    scrθ = Vector{T}(undef, k)
    scrσ = [Vector{T}(undef, l) for l in Λsize]
    scrρ = [Matrix{T}(undef, l, l) for l in Λsize]
    scr = similar.(scrρ)
    @showprogress for i in 1 : N
        j = 0
        refit!(simulate!(m, β = β, σ = σ, θ = θ))
        dfr[j += 1][i] = objective(m)
        dfr[j += 1][i] = σest = sdest(m)
        for x in fixef!(scrβ, m)
            dfr[j += 1][i] = x
        end
        for x in getθ!(scrθ, m)
            dfr[j += 1][i] = x
        end
        for l in eachindex(trms)
            stddevcor!(scrσ[l], scrρ[l], scr[l], trms[l])
            for x in scrσ[l]
                dfr[j += 1][i] = σest * x
            end
            ρl = scrρ[l]
            sz = size(ρl, 1)
            for jj in 1 : (sz - 1), ii in (jj + 1) : sz
                dfr[j += 1][i] = ρl[ii, jj]
            end
        end
    end
    refit!(m, y₀)
    dfr
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

simulate!(m::LinearMixedModel{T}; β=coef(m), σ=sdest(m), θ=T[]) where {T} =
    simulate!(Random.GLOBAL_RNG, m, β=β, σ=σ, θ=θ)
