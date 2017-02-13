"""
    bootstrap!{T}(r::Matrix{T}, m::LinearMixedModel{T}, f!::Function;
        β=fixef(m), σ=sdest(m), θ=getθ(m))

Overwrite columns of `r` with the results of applying the mutating extractor `f!`
to parametric bootstrap replications of model `m`.

The signature of `f!` should be
    f!{T}(v::AbstractVector{T}, m::LinearMixedModel{T})

# Named Arguments

`β::Vector{T}`, `σ::T`, and `θ::Vector{T}` are the values of the parameters in `m`
for simulation of the responses.
"""
function bootstrap!{T}(r::Matrix{T}, m::LinearMixedModel{T}, f!::Function;
    β=fixef(m), σ=sdest(m), θ=getθ(m))
    y₀ = copy(model_response(m)) # to restore original state of m
    for i in 1 : size(r, 2)
        f!(view(r, :, i), refit!(simulate!(m, β = β, σ = σ, θ = θ)))
    end
    refit!(m, y₀)               # restore original state of m
    r
end

"""
    bootstrap{T}(N, m::LinearMixedModel{T},
        β::Vector{T}=fixef(m), σ::T=sdest(m), θ::Vector{T}=getθ(m))

Perform `N` parametric bootstrap replication fits of `m`, returning a data frame
with column names `:obj`, the objective function at convergence, `:σ`, the estimated
standard deviation of the residuals, `βᵢ, i = 1,...,p`, the fixed-effects coefficients,
and `θᵢ, i = 1,...,k` the covariance parameters.

# Named Arguments

`β::Vector{T}`, `σ::T`, and `θ::Vector{T}` are the values of the parameters in `m`
for simulation of the responses.
"""
function bootstrap{T}(N, m::LinearMixedModel{T};
    β=fixef(m), σ=sdest(m), θ=getθ(m))
    y₀ = copy(model_response(m)) # to restore original state of m
    p = size(m.trms[end - 1], 2)
    length(β) == p || throw(DimensionMismatch("length(β) should be $p"))
    k = length(getθ(m))
    length(θ) == k || throw(DimensionMismatch("length(θ) should be $k"))
    Λ = m.Λ
    Λsize = [isa(λ, UniformScaling) ? 1 : size(λ, 2) for λ in Λ]
    cnms = vcat([:obj, :σ], Symbol.(subscriptednames('β', p)),
        Symbol.(subscriptednames('θ', k)), Symbol.(subscriptednames('σ', sum(Λsize))))
    nρ = [(l * (l - 1)) >> 1 for l in Λsize]
    if (nρtot = sum(nρ)) > 0
        append!(cnms, Symbol.(subscriptednames('ρ', nρtot)))
    end
    dfr = DataFrame(Any[Array(T, (N,)) for _ in eachindex(cnms)], cnms)
    scrβ, scrθ = Array(T, (p, )), Array(T, (k, ))
    scrσ = [Array(T, (l, )) for l in Λsize]
    scrρ = [Array(T, (l, l)) for l in Λsize]
    scr = [similar(sρ) for sρ in scrρ]
    for i in 1 : N
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
        for l in eachindex(Λ)
            λ = Λ[l]
            stddevcor!(scrσ[l], scrρ[l], scr[l],
                LinAlg.Cholesky(isa(λ, UniformScaling) ? λ * ones(1,1) : λ, :L))
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

function stddevcor!{T}(σ::Vector{T}, ρ::Matrix{T}, scr::Matrix{T}, L::LinAlg.Cholesky{T})
    if length(σ) != (k = size(L, 2)) || size(ρ) ≠ (k, k) || size(scr) ≠ (k, k)
        throw(DimensionMismatch(string("size(ρ) = $(size(ρ)) and size(scr) = $(size(scr)) ",
            "should be ($k, $k) and length(σ) = $(length(σ)) should be $k")))
    end
    if k == 1
        copy!(σ, L.factors)
        ρ[1, 1] = one(T)
    elseif L.uplo == 'L'
        copy!(scr, L.factors)
        for i in 1 : k
            σ[i] = σi = norm(view(scr, i, 1 : i))
            for j in 1 : i
                scr[i, j] /= σi
            end
        end
        A_mul_Bc!(ρ, LowerTriangular(scr), LowerTriangular(scr))
    elseif L.uplo == 'U'
        copy!(scr, L.factors)
        for j in 1 : k
            σ[j] = σj = norm(view(scr, 1 : j, j))
            for i in 1 : j
                scr[i, j] /= σj
            end
        end
        Ac_mul_B!(ρ, UpperTriangular(scr), UpperTriangular(scr))
    else
        throw(ArgumentError("L.uplo should be 'L' or 'U'"))
    end
    σ, ρ
end

function stddevcor{T}(L::LinAlg.Cholesky{T})
    k = size(L, 1)
    stddevcor!(Array(T, (k,)), Array(T, (k, k)), Array(T, (k, k)), L)
end

stddevcor(L::LowerTriangular) = stddevcor(LinAlg.Cholesky(L, :L))
stddevcor{T}(L::UniformScaling{T}) = [abs(L.λ)], eye(T, 1)

"""
    reevaluateAend!(m::LinearMixedModel)

Reevaluate the last column of `m.A` from `m.trms`.  This function should be called
after updating the response, `m.trms[end]`.
"""
function reevaluateAend!(m::LinearMixedModel)
    A, trms, sqrtwts, wttrms = m.A, m.trms, m.sqrtwts, m.wttrms
    wttrmn = wttrms[end]
    if !isempty(sqrtwts)
        A_mul_B!(wttrmn, sqrtwts, trms[end])
    end
    for i in eachindex(wttrms)
        Ac_mul_B!(A[end, i], wttrmn, wttrms[i])
    end
    m
end

"""
    refit!{T}(m::LinearMixedModel{T}[, y::Vector{T}])

Refit the model `m` after installing response `y`.

If `y` is omitted the current response vector is used.
"""
refit!(m::LinearMixedModel) = fit!(cholBlocked!(resetθ!(reevaluateAend!(m))))
function refit!{T}(m::LinearMixedModel{T}, y)
    resp = m.trms[end]
    if length(y) ≠ size(resp, 1)
        throw(DimensionMismatch("length(y) = $(length(y)), should be $(size(resp, 1))"))
    end
    copy!(resp, y)
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
    cholBlocked!(setθ!(m, opt.initial))
end

"""
    unscaledre!{T}(y::Vector{T}, M::ReMat{T}, b::Matrix{T})

Add unscaled random effects defined by `M` and `b` to `y`.
"""
function unscaledre!{T<:AbstractFloat}(y::Vector{T}, M::ScalarReMat{T}, b::Matrix{T})
    z = M.z
    if length(y) ≠ length(z) || size(b, 1) ≠ 1
        throw(DimensionMismatch())
    end
    inds = M.f.refs
    @inbounds for i in eachindex(y)
        y[i] += b[inds[i]] * z[i]
    end
    y
end

"""
    unscaledre!{T}(y::AbstractVector{T}, M::ReMat{T}, L::LowerTriangular{T})

Add unscaled random effects defined by `M` and `L * randn(1, length(M.f.pool))` to `y`.
"""
function unscaledre!{T}(y::AbstractVector{T}, M::ScalarReMat{T}, L::UniformScaling{T})
    re = randn(1, length(M.f.pool))
    unscaledre!(y, M, (re *= L))
end

function unscaledre!{T}(y::AbstractVector{T}, M::VectorReMat{T}, b::DenseMatrix{T})
    Z = M.z
    k, n = size(Z)
    l = length(M.f.pool)
    if length(y) ≠ n || size(b) ≠ (k, l)
        throw(DimensionMismatch("length(y) = $(length(y)), size(M) = $(size(M)), size(b) = $(size(b))"))
    end
    inds = M.f.refs
    for i in eachindex(y)
        ii = inds[i]
        for j in 1:k
            y[i] += Z[j,i] * b[j, ii]
        end
    end
    y
end

unscaledre!(y::AbstractVector, M::VectorReMat, L::LowerTriangular) =
    unscaledre!(y, M, A_mul_B!(L, randn(size(M.z, 1), length(M.f.pool))))

"""
    simulate!(m::LinearMixedModel; β=fixef(m), σ=sdest(m), θ=getθ(m))

Overwrite the response (i.e. `m.trms[end]`) with a simulated response vector from model `m`.
"""
function simulate!{T}(m::LinearMixedModel{T}; β = coef(m), σ = sdest(m), θ = T[])
    if !isempty(θ)
        setθ!(m, θ)
    end
    trms, Λ = m.trms, m.Λ
    y = randn!(model_response(m)) # initialize to standard normal noise
    for j in eachindex(Λ)         # add the unscaled random effects
        unscaledre!(y, trms[j], Λ[j])
    end
                                  # scale by σ and add fixed-effects contribution
    BLAS.gemv!('N', 1.0, trms[end - 1], β, σ, y)
    m
end

StatsBase.model_response(m::LinearMixedModel) = vec(m.trms[end])
