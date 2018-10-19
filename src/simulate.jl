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
    y₀ = copy(model_response(m)) # to restore original state of m
    p = size(m.trms[end - 1], 2)
    @argcheck(length(β) == p, DimensionMismatch)
    @argcheck(length(θ) == (k = length(getθ(m))), DimensionMismatch)
    trms = reterms(m)
    Λsize = vsize.(trms)
    cnms = vcat([:obj, :σ], Symbol.(subscriptednames('β', p)),
        Symbol.(subscriptednames('θ', k)), Symbol.(subscriptednames('σ', sum(Λsize))))
    nρ = [(l * (l - 1)) >> 1 for l in Λsize]  # FIXME: Do this more carefully.
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

"""
    subscriptednames(nm, len)

Return a `Vector{String}` of `nm` with subscripts from `₁` to `len`
"""
function subscriptednames(nm, len)
    nd = ndigits(len)
    nd == 1 ?
        [string(nm, '₀' + j) for j in 1:len] :
        [string(nm, lpad(string(j), nd, '0')) for j in 1:len]
end

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

"""
    reevaluateAend!(m::LinearMixedModel)

Reevaluate the last column of `m.A` from `m.trms`.  This function should be called
after updating the response, `m.trms[end]`.
"""
function reevaluateAend!(m::LinearMixedModel)
    A = m.A
    trms = m.trms
    trmn = reweight!(trms[end], m.sqrtwts)
    nblk = nblocks(A, 2)
    for (i, trm) in enumerate(trms)
        mul!(A[Block(nblk, i)], trmn', trm)
    end
    m
end

"""
    refit!(m::LinearMixedModel[, y::Vector])

Refit the model `m` after installing response `y`.

If `y` is omitted the current response vector is used.
"""
refit!(m::LinearMixedModel) = fit!(updateL!(resetθ!(reevaluateAend!(m))))
function refit!(m::LinearMixedModel, y)
    resp = m.trms[end]
    @argcheck length(y) == size(resp, 1) DimensionMismatch
    copyto!(resp, y)
    refit!(m)
end

"""
    resetθ!(m::LinearMixedModel)

Reset the value of `m.θ` to the initial values and mark the model as not having been fit
"""
function resetθ!(m::LinearMixedModel)
    opt = m.optsum
    opt.feval = -1
    opt.fmin = Inf
    updateL!(setθ!(m, opt.initial))
end

"""
    unscaledre!(y::AbstractVector{T}, M::AbstractFactorReTerm{T}, b) where {T}
    unscaledre!(rng::AbstractRNG, y::AbstractVector{T}, M::AbstractFactorReTerm{T}) where {T}

Add unscaled random effects defined by `M` and `b` to `y`.  When `rng` is present the `b`
vector is generated as `randn(rng, size(M, 2))`
"""
function unscaledre! end

function unscaledre!(y::AbstractVector, A::ScalarFactorReTerm, b::AbstractVecOrMat)
    m, n = size(A)
    @argcheck(length(y) == m && length(b) == n, DimensionMismatch)
    z = A.z
    for (i, r) in enumerate(A.refs)
        y[i] += b[r] * z[i]
    end
    y
end

function unscaledre!(y::AbstractVector{T}, A::VectorFactorReTerm{T,R,S},
                     b::DenseMatrix) where {T,R,S}
    Z = A.z
    k, n = size(Z)
    l = nlevs(A)
    @argcheck(length(y) == n && size(b) == (k, l), DimensionMismatch)
    for (i, ii) in enumerate(A.refs)
        for j in 1:k
            y[i] += Z[j, i] * b[j, ii]
        end
    end
    y
end

function unscaledre!(rng::AbstractRNG, y::AbstractVector{T},
                     A::VectorFactorReTerm{T}) where {T}
    unscaledre!(y, A, lmul!(LowerTriangular(A.Λ), randn(rng, vsize(A), nlevs(A))))
end

function unscaledre!(rng::AbstractRNG, y::AbstractVector{T},
                     A::ScalarFactorReTerm{T}) where {T}
    unscaledre!(y, A, lmul!(A.Λ, randn(rng, vsize(A), nlevs(A))))
end

unscaledre!(y::AbstractVector, A::AbstractFactorReTerm) = unscaledre!(Base.GLOBAL_RNG, y, A)

"""
    simulate!(m::LinearMixedModel; β=fixef(m), σ=sdest(m), θ=getθ(m))

Overwrite the response (i.e. `m.trms[end]`) with a simulated response vector from model `m`.
"""
function simulate!(rng::AbstractRNG, m::LinearMixedModel{T};
                   β=coef(m), σ=sdest(m), θ=T[]) where {T}
    if !isempty(θ)
        setθ!(m, θ)
    end
    y = randn!(rng, model_response(m)) # initialize to standard normal noise
    for trm in reterms(m)              # add the unscaled random effects
        unscaledre!(rng, y, trm)
    end
                                  # scale by σ and add fixed-effects contribution
    BLAS.gemv!('N', one(T), m.trms[end - 1].x, β, σ, y)
    m
end

simulate!(m::LinearMixedModel{T}; β=coef(m), σ=sdest(m), θ=T[]) where {T} =
    simulate!(Random.GLOBAL_RNG, m, β=β, σ=σ, θ=θ)
