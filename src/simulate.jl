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
        f!(Compat.view(r, :, i), refit!(simulate!(m, β = β, σ = σ, θ = θ)))
    end
    refit!(m, y₀)               # restore original state of m
    r
end

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
        Ac_mul_B!(A[i, end], wttrms[i], wttrmn)
    end
    m
end

"""
    refit!{T}(m::LinearMixedModel{T}[, y::Vector{T}])

Refit the model `m` after installing response `y`.

If `y` is omitted the current response vector is used.
"""
refit!(m::LinearMixedModel) = fit!(cfactor!(resetθ!(reevaluateAend!(m))))
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
    opt = m.opt
    opt.feval = -1
    opt.fmin = Inf
    setθ!(m, opt.initial) |> cfactor!
end

"""
    unscaledre!{T}(y::Vector{T}, M::ReMat{T}, b::Matrix{T})

Add unscaled random effects defined by `M` and `b` to `y`.
"""
function unscaledre!{T<:AbstractFloat,S,R<:Integer}(y::Vector{T}, M::ScalarReMat{T,S,R}, b::Matrix{T})
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
function unscaledre!{T}(y::AbstractVector{T}, M::ScalarReMat{T}, L::LowerTriangular{T})
    unscaledre!(y, M, A_mul_B!(L, randn(1, length(M.f.pool))))
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
    simulate!(m::LinearMixedModel; β=fixef(m), σ=sdest(m), θ=getΘ(m))

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
