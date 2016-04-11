"""
    bootstrap(m, N, saveresults)

Simulate `N` response vectors from `m`, refitting the model.  The function saveresults
is called after each refit.

To save space `m.trms[end]`, which is the response vector, is overwritten
by each simulation.  The original response is restored before returning.

Args:

- `m`: a `LinearMixedModel` that has been fit.
- `N`: the number of bootstrap samples to simulate
- `savresults`: a function with arguments `i` and `m` called after each bootstrap simulation.
   As the name indicates, this function should save the results of interest.
"""
function bootstrap(m::LinearMixedModel, N::Integer, saveresults::Function)
    y0 = copy(model_response(m))
    β = coef(m)
    σ = sdest(m)
    θ = m[:θ]
    for i in 1:N
        saveresults(i, simulate!(m; β = β, σ = σ, θ = θ))
    end
    refit!(m,y0)
end


"""
    reevaluateAend!(m)

Reevaluate the last column of `m.A` from `m.trms`

Args:

- `m`: a `LinearMixedModel`

Returns:
  `m` with the last column of `m.A` reevaluated

Note: This function should be called after updating the response, `m.trms[end]`.
"""
function reevaluateAend!(m::LinearMixedModel)
    A, trms, sqrtwts, wttrms = m.A, m.trms, m.sqrtwts, m.wttrms
    wttrmn = wttrms[end]
    if !isempty(sqrtwts)
        scale!(wttrmn, sqrtwts, trms[end])
    end
    for i in eachindex(wttrms)
        Ac_mul_B!(A[i, end], wttrms[i], wttrmn)
    end
    m
end

"""
    resetθ!(m)

Reset the value of `m.θ` to the initial values and mark the model as not having been fit

Args:

- `m`: a `LinearMixedModel`

Returns:
  `m`
"""
function resetθ!(m::LinearMixedModel)
    m[:θ] = m.opt.initial
    m.opt.feval = -1
    m.opt.fmin = Inf
    m
end

"""
    unscaledre!(y, M, L, u)

Add unscaled random effects to `y`.

Args:

- `y`: response vector to which the random effects are to be added
- `M`: an `ReMat`
- `L`: the `LowerTriangular` matrix defining `Λ` for this term
- `u`: a `Matrix` of random effects on the `u` scale. Defaults to a standard multivariate normal of the appropriate size.

Returns:
  the updated `y`
"""
function unscaledre!(y::AbstractVector,
      M::ScalarReMat,
      L::LowerTriangular,
      u::DenseMatrix)
    z = M.z
    if length(y) ≠ length(z) || size(L) ≠ (1, 1) || size(u, 1) ≠ 1
        throw(DimensionMismatch())
    end
    re = L[1, 1] * vec(u)
    inds = M.f.refs
    for i in eachindex(y)
        y[i] += re[inds[i]] * z[i]
    end
    y
end

function unscaledre!(y::AbstractVector, M::ScalarReMat, L::LowerTriangular)
    unscaledre!(y, M, L, randn(1, length(M.f.pool)))
end

function unscaledre!(y::AbstractVector, M::VectorReMat, L::LowerTriangular,
    u::DenseMatrix)

    Z = M.z
    k, n = size(Z)
    l = length(M.f.pool)
    if length(y) ≠ n || size(u) != (k, l) || size(L) ≠ (k, k)
        throw(DimensionMismatch())
    end
    re = L * u
    inds = M.f.refs
    for i in eachindex(y)
        y[i] += dot(sub(Z, :, i), sub(re, :, Int(inds[i])))
    end
    y
end

unscaledre!(y::AbstractVector, M::VectorReMat, L::LowerTriangular) =
    unscaledre!(y, M, L, randn(size(M.z, 1), length(M.f.pool)))

"""
    simulate!(m; β, σ, θ)

Simulate a response vector from model `m`, and refit `m`.

Args:

- `m`: a `LinearMixedModel`.
- `β`: the fixed-effects parameter vector to use; defaults to `coef(m)`
- `σ`: the standard deviation of the per-observation random noise term to use; defaults to `sdest(m)`
- `θ`: the covariance parameter vector to use; defaults to `m[:θ]`

Returns:
  `m` after having refit it to the simulated response vector
"""
function simulate!(m::LinearMixedModel; β = coef(m), σ = sdest(m), θ = m[:θ])
    m[:θ] = θ
    trms, Λ = unwttrms(m), m.Λ
    y = randn!(model_response(m)) # initialize to standard normal noise
    for j in eachindex(Λ)         # add the unscaled random effects
        unscaledre!(y, trms[j], Λ[j])
    end
    Base.LinAlg.BLAS.gemv!('N', 1.0, trms[end - 1], β, σ, y)
    m |> reevaluateAend! |> resetθ! |> fit!
end

"""
    refit!(m, y)
Refit the model `m` with response `y`

Args:

- `m`: a `MixedModel{T}`
- `y`: a `Vector{T}` of length `n`, the number of observations in `m`

Returns:
  `m` after refitting
"""
function refit!(m::LinearMixedModel,y)
    copy!(model_response(m),y)
    m |> reevaluateAend! |> resetθ! |> fit!
end

"""
extract the response (as a reference)
"""
StatsBase.model_response(m::LinearMixedModel) = vec(unwttrms(m)[end])
